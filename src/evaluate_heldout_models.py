
from pathlib import Path
import json
import time

import numpy as np
import torch

from preprocessing.patch_dataset import (
    discover_cases,
    BraTSPatchDataset,
)
from models.attention_unet import AttentionUNet3D
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "results/checkpoints/attention_unet_pilot_best.pth"
HYBRID_PATH = ROOT / "results/checkpoints/hybrid_quantum_unet_ring_extended.pth"
HISTORY_PATH = ROOT / "results/hybrid_ring_extended_history.json"

TEST_PATIENTS_PATH = (
    ROOT / "results/comparison/heldout_test_patients.json"
)

OUTPUT_PATH = (
    ROOT / "results/comparison/baseline_vs_hybrid_heldout.json"
)

PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 4
NUM_CLASSES = 4
CLASS_NAMES = {
    1: "Label_1",
    2: "Label_2",
    3: "Label_3",
}
REGION_LABELS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


# ============================================================
# DEVICE SYNCHRONIZATION
# ============================================================

def synchronize(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target, class_id):
    """Calculate metrics for one individual segmentation class."""
    pred = prediction == class_id
    truth = target == class_id

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn

    return {
        "dice": float(2 * tp / dice_denominator if dice_denominator else 1.0),
        "iou": float(tp / union if union else 1.0),
        "precision": float(tp / (tp + fp) if tp + fp else 0.0),
        "recall": float(tp / (tp + fn) if tp + fn else 0.0),
    }


def calculate_region_metrics(prediction, target, labels):
    """Calculate metrics for a composite tumor region such as WT, TC, or ET."""
    pred = np.isin(prediction, labels)
    truth = np.isin(target, labels)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn

    return {
        "dice": float(2 * tp / dice_denominator if dice_denominator else 1.0),
        "iou": float(tp / union if union else 1.0),
        "precision": float(tp / (tp + fp) if tp + fp else 0.0),
        "recall": float(tp / (tp + fn) if tp + fn else 0.0),
    }


def average_metrics(items):
    if not items:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    return {
        key: float(np.mean([item[key] for item in items]))
        for key in ("dice", "iou", "precision", "recall")
    }


# ============================================================
# DATASET SAMPLE HANDLING
# ============================================================

def unpack_sample(sample):
    """
    Accept common dataset return formats:
      (image, mask)
      [image, mask]
      {"image": ..., "mask": ...}
      {"image": ..., "label": ...}
      {"image": ..., "segmentation": ...}

    If your dataset uses different dictionary keys, update this
    function to match src/preprocessing/patch_dataset.py.
    """
    if isinstance(sample, dict):
        image = sample.get("image", sample.get("x"))
        mask = sample.get(
            "mask",
            sample.get("label", sample.get("segmentation", sample.get("y")))
        )

        if image is None or mask is None:
            raise KeyError(
                "Could not identify image and mask in dataset sample. "
                f"Available keys: {list(sample.keys())}"
            )

        return image, mask

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], sample[1]

    raise TypeError(
        "Unsupported dataset sample format. Inspect "
        "BraTSPatchDataset.__getitem__() in patch_dataset.py."
    )


def prepare_tensor(tensor, device, is_image=False):
    tensor = torch.as_tensor(tensor)

    if is_image:
        # Expected image shape: [C, D, H, W].
        if tensor.ndim == 4 and tensor.shape[0] != 4:
            if tensor.shape[-1] == 4:
                tensor = tensor.permute(3, 0, 1, 2)

        if tensor.ndim != 4 or tensor.shape[0] != 4:
            raise ValueError(
                f"Expected image shape [4,D,H,W], got {tuple(tensor.shape)}"
            )

        tensor = tensor.float().unsqueeze(0)

    else:
        # Expected target shape: [D,H,W], optionally with one channel.
        tensor = tensor.squeeze()

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected mask shape [D,H,W], got {tuple(tensor.shape)}"
            )

        # Handle one-hot masks if supplied as [C,D,H,W].
        tensor = tensor.long()

    return tensor.to(device)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, patches, patient_ids, device, model_name):
    """Evaluate identical patches and report class and composite-region metrics."""
    per_patient = {
        pid: {
            "classes": {name: [] for name in CLASS_NAMES.values()},
            "regions": {name: [] for name in REGION_LABELS},
        }
        for pid in patient_ids
    }

    inference_times = []
    model.eval()

    with torch.inference_mode():
        for patch_index, sample in enumerate(patches):
            patient_index = patch_index // PATCHES_PER_CASE
            if patient_index >= len(patient_ids):
                raise RuntimeError(
                    "Patch count exceeds the configured test patient count."
                )
            patient_id = patient_ids[patient_index]

            image, mask = unpack_sample(sample)
            image = prepare_tensor(image, device, is_image=True)
            target = torch.as_tensor(mask).detach().cpu().numpy()
            target = np.squeeze(target)

            if target.ndim == 4:
                if target.shape[0] == NUM_CLASSES:
                    target = np.argmax(target, axis=0)
                elif target.shape[-1] == NUM_CLASSES:
                    target = np.argmax(target, axis=-1)
                else:
                    raise ValueError(f"Unsupported target shape: {target.shape}")
            if target.ndim != 3:
                raise ValueError(f"Expected 3D integer target, got shape {target.shape}")

            synchronize(device)
            start_time = time.perf_counter()
            logits = model(image)
            synchronize(device)
            inference_times.append(time.perf_counter() - start_time)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                if "out" not in logits:
                    raise ValueError(f"Unexpected model output dictionary: {list(logits)}")
                logits = logits["out"]
            if logits.ndim != 5:
                raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")

            prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: {prediction.shape} versus {target.shape}"
                )

            for class_id, class_name in CLASS_NAMES.items():
                per_patient[patient_id]["classes"][class_name].append(
                    calculate_metrics(prediction, target, class_id)
                )
            for region_name, labels in REGION_LABELS.items():
                per_patient[patient_id]["regions"][region_name].append(
                    calculate_region_metrics(prediction, target, labels)
                )

            if (patch_index + 1) % 20 == 0:
                print(f"{model_name}: evaluated {patch_index + 1}/{len(patches)} patches")

    patient_results = {}
    for patient_id, groups in per_patient.items():
        patient_results[patient_id] = {
            group: {
                name: average_metrics(values)
                for name, values in group_values.items()
            }
            for group, group_values in groups.items()
        }

    summary = {"classes": {}, "regions": {}}
    for group in ("classes", "regions"):
        names = CLASS_NAMES.values() if group == "classes" else REGION_LABELS.keys()
        for name in names:
            summary[group][name] = {
                metric: float(np.mean([
                    patient_results[pid][group][name][metric]
                    for pid in patient_ids
                ]))
                for metric in ("dice", "iou", "precision", "recall")
            }

    return {
        "summary": summary,
        "per_patient": patient_results,
        "mean_inference_seconds_per_patch": (
            float(np.mean(inference_times)) if inference_times else None
        ),
        "total_inference_seconds": float(sum(inference_times)),
        "patch_count": len(patches),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    required_paths = (
        BASELINE_PATH,
        HYBRID_PATH,
        HISTORY_PATH,
        TEST_PATIENTS_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...")

    baseline_ckpt = torch.load(
        BASELINE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    hybrid_ckpt = torch.load(
        HYBRID_PATH,
        map_location="cpu",
        weights_only=True,
    )

    history = json.loads(
        HISTORY_PATH.read_text(encoding="utf-8")
    )
    cfg = history["configuration"]

    if hybrid_ckpt.get("configuration") != cfg:
        raise ValueError(
            "Hybrid checkpoint/history configuration mismatch."
        )

    test_data = json.loads(
        TEST_PATIENTS_PATH.read_text(encoding="utf-8")
    )
    test_ids = test_data["patient_ids"]

    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Duplicate IDs in held-out patient list.")

    # Check overlap with both models' recorded training/validation sets.
    used_ids = (
        set(cfg["train_patient_ids"])
        | set(cfg["validation_patient_ids"])
        | set(baseline_ckpt.get("train_case_ids", []))
        | set(baseline_ckpt.get("val_case_ids", []))
    )

    overlap = set(test_ids) & used_ids

    if overlap:
        raise RuntimeError(
            "Held-out test patients overlap existing training/validation "
            f"splits: {sorted(overlap)}"
        )

    print(f"Checking {len(test_ids)} held-out patients...")

    all_cases = {
        case["id"]: case
        for case in discover_cases()
    }

    missing = [pid for pid in test_ids if pid not in all_cases]

    if missing:
        raise RuntimeError(f"Test patients not found in dataset: {missing}")

    test_ids = sorted(test_ids)
    cases = [all_cases[pid] for pid in test_ids]

    print("Creating fixed patch dataset...")

    dataset = BraTSPatchDataset(
        cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=2027,
    )

    fixed_patches = [
        dataset[index]
        for index in range(len(dataset))
    ]

    expected_patches = len(test_ids) * PATCHES_PER_CASE

    if len(fixed_patches) != expected_patches:
        raise RuntimeError(
            f"Expected {expected_patches} patches, "
            f"but dataset generated {len(fixed_patches)}."
        )

    print(f"Test patients: {len(test_ids)}")
    print(f"Prepared patches: {len(fixed_patches)}")

    device = select_device()
    print("Inference device:", device)

    print("Loading baseline model...")
    baseline = AttentionUNet3D(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base_channels=8,
    )
    baseline.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline = baseline.to(device).eval()

    print("Loading hybrid model...")
    hybrid = HybridQuantumUNet(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base_channels=8,
        strategy=hybrid_ckpt["strategy"],
    )
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    hybrid = hybrid.to(device).eval()

    print("\nEvaluating baseline Attention U-Net...")
    baseline_results = evaluate_model(
        baseline,
        fixed_patches,
        test_ids,
        device,
        "Baseline",
    )

    print("\nEvaluating hybrid quantum U-Net...")
    hybrid_results = evaluate_model(
        hybrid,
        fixed_patches,
        test_ids,
        device,
        "Hybrid",
    )

    output = {
        "evaluation_type": (
            "held-out patient patch-level evaluation; exploratory comparison"
        ),
        "test_patient_ids": test_ids,
        "test_patient_count": len(test_ids),
        "patch_size": list(PATCH_SIZE),
        "patches_per_patient": PATCHES_PER_CASE,
        "total_patches": len(fixed_patches),
        "training_cohorts_match": False,
        "baseline_checkpoint": str(BASELINE_PATH.relative_to(ROOT)),
        "hybrid_checkpoint": str(HYBRID_PATH.relative_to(ROOT)),
        "device": str(device),
        "metrics_are_patient_macro_averages": True,
        "baseline_attention_unet": baseline_results,
        "hybrid_quantum_unet": hybrid_results,
        "interpretation_note": (
            "Exploratory comparison only. The models used different "
            "training cohort sizes and training budgets; observed "
            "differences cannot be attributed to quantum components alone."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("HELD-OUT EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, results in (
       ("Baseline Attention U-Net", baseline_results),
        ("Hybrid Quantum U-Net", hybrid_results),
    ):
        print(f"\n{model_name}")

        for group_name in ("classes", "regions"):
            print(f"  {group_name.upper()}")

            for name, metrics in results["summary"][group_name].items():
                print(
                    f"    {name}: "
                    f"Dice={metrics['dice']:.4f}, "
                    f"IoU={metrics['iou']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}"
                )

        print(
            "  Mean inference time per patch: "
            f"{results['mean_inference_seconds_per_patch']:.4f} seconds"
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
