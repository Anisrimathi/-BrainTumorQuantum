
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_metrics(prediction, target):
    """Calculate Dice, IoU, precision and recall."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    denominator = 2 * tp + fp + fn
    union = tp + fp + fn

    dice = 1.0 if denominator == 0 else 2 * tp / denominator
    iou = 1.0 if union == 0 else tp / union

    precision = (
        1.0 if tp + fp == 0 and tp + fn == 0
        else tp / (tp + fp) if tp + fp else 0.0
    )
    recall = (
        1.0 if tp + fn == 0 and tp + fp == 0
        else tp / (tp + fn) if tp + fn else 0.0
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def hd95(prediction, target, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate symmetric HD95 in physical distance units.

    voxel_spacing must follow the array-axis order of the masks,
    in millimetres for NIfTI MRI volumes.

    Returns:
        0.0 if both masks are empty.
        NaN if exactly one mask is empty.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    if prediction.ndim != 3:
        raise ValueError("HD95 expects 3D masks.")

    if len(voxel_spacing) != 3 or any(
        not np.isfinite(s) or s <= 0 for s in voxel_spacing
    ):
        raise ValueError("Provide three positive voxel spacings.")

    if not prediction.any() and not target.any():
        return 0.0

    if not prediction.any() or not target.any():
        return float("nan")

    structure = np.ones((3, 3, 3), dtype=bool)

    prediction_surface = prediction & ~binary_erosion(
        prediction, structure=structure, border_value=0
    )
    target_surface = target & ~binary_erosion(
        target, structure=structure, border_value=0
    )

    # Distance to the nearest target surface voxel.
    distance_to_target = distance_transform_edt(
        ~target_surface, sampling=voxel_spacing
    )
    distance_to_prediction = distance_transform_edt(
        ~prediction_surface, sampling=voxel_spacing
    )

    distances_prediction_to_target = distance_to_target[
        prediction_surface
    ]
    distances_target_to_prediction = distance_to_prediction[
        target_surface
    ]

    all_distances = np.concatenate([
        distances_prediction_to_target,
        distances_target_to_prediction,
    ])

    return float(np.percentile(all_distances, 95))


def make_brats_regions(label_volume):
    """
    BraTS labels:
      0 = background
      1 = necrotic/non-enhancing tumor core
      2 = edema
      3 = enhancing tumor

    WT = labels 1, 2, 3
    TC = labels 1, 3
    ET = label 3
    """
    labels = np.asarray(label_volume)

    if not np.isin(np.unique(labels), [0, 1, 2, 3]).all():
        raise ValueError("Expected segmentation labels 0, 1, 2, 3.")

    return {
        "WT": labels > 0,
        "TC": (labels == 1) | (labels == 3),
        "ET": labels == 3,
    }


def evaluate_brats_regions(
    prediction,
    target,
    voxel_spacing=(1.0, 1.0, 1.0),
):
    """Calculate overlap and HD95 metrics for WT, TC and ET."""
    prediction_regions = make_brats_regions(prediction)
    target_regions = make_brats_regions(target)

    results = {}

    for name in ("WT", "TC", "ET"):
        pred_mask = prediction_regions[name]
        true_mask = target_regions[name]

        results[name] = binary_metrics(pred_mask, true_mask)
        results[name]["hd95_mm"] = hd95(
            pred_mask,
            true_mask,
            voxel_spacing=voxel_spacing,
        )

    return results


if __name__ == "__main__":
    target = np.zeros((16, 16, 16), dtype=np.uint8)
    prediction = np.zeros_like(target)

    target[4:10, 4:10, 4:10] = 2
    target[6:8, 6:8, 6:8] = 3
    prediction[:] = target

    results = evaluate_brats_regions(
        prediction,
        target,
        voxel_spacing=(1.0, 1.0, 1.0),
    )

    for region, metrics in results.items():
        print(region, metrics)

    assert all(
        metrics["dice"] == 1.0
        and metrics["iou"] == 1.0
        and metrics["hd95_mm"] == 0.0
        for metrics in results.values()
    )

    print("All evaluation metric tests passed.")
