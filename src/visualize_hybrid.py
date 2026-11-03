
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.patch_dataset import discover_cases
from models.hybrid_quantum_unet import (
    HybridQuantumUNet,
    select_device,
)

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "hybrid_quantum_unet_ring_final_v2.pth"
)

MODALITIES = {
    "T1": "t1n",
    "T1ce": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
}


def normalize_for_display(volume):
    volume = np.asarray(volume, dtype=np.float32)
    nonzero = volume != 0

    if not nonzero.any():
        return np.zeros_like(volume)

    values = volume[nonzero]
    low, high = np.percentile(values, [1, 99])

    if high <= low:
        return np.zeros_like(volume)

    return np.clip((volume - low) / (high - low), 0, 1)


def main():
    cases = discover_cases()

    if not cases:
        raise RuntimeError("No complete BraTS-GLI cases found.")

    print("Complete cases:", len(cases))
    print("Enter a case number from 1 to", len(cases))
    case_number = int(input("Case number: "))

    if not 1 <= case_number <= len(cases):
        raise ValueError("Case number is outside the available range.")

    case = cases[case_number - 1]
    print("Selected case:", case["id"])

    volumes = {}
    for name, suffix in MODALITIES.items():
        path = case["images"][name]
        volumes[name] = nib.load(str(path)).get_fdata(
            dtype=np.float32
        )

    label_volume = nib.load(
        str(case["seg"])
    ).get_fdata(dtype=np.float32)

    if not np.allclose(label_volume, np.rint(label_volume)):
        raise ValueError("Segmentation contains non-integer labels.")

    label_volume = np.rint(label_volume).astype(np.uint8)

    # Select the axial slice containing the most tumor voxels.
    tumor_per_slice = np.count_nonzero(
        label_volume > 0, axis=(0, 1)
    )

    if tumor_per_slice.max() > 0:
        slice_index = int(np.argmax(tumor_per_slice))
    else:
        slice_index = label_volume.shape[2] // 2

    print("Selected axial slice:", slice_index)

    prediction_volume = None

    if CHECKPOINT.is_file():
        print("Loading hybrid checkpoint:", CHECKPOINT)
        device = select_device()

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        model = HybridQuantumUNet(
            in_channels=4,
            num_classes=4,
            base_channels=8,
            strategy=checkpoint.get("strategy", "ring"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # The pilot model accepts 32x32x32 patches. For this
        # visualization, use a centered patch and show its middle slice.
        center = [
            dimension // 2
            for dimension in label_volume.shape
        ]

        patch_size = 32
        starts = [
            max(0, min(c - patch_size // 2, dim - patch_size))
            for c, dim in zip(center, label_volume.shape)
        ]

        x, y, z = starts
        px, py, pz = patch_size, patch_size, patch_size

        patch_channels = []
        for name in MODALITIES:
            volume = volumes[name]
            mask = volume != 0

            if mask.any():
                mean = volume[mask].mean()
                std = volume[mask].std()

                normalized = volume.copy()
                if std > 1e-6:
                    normalized[mask] = (
                        volume[mask] - mean
                    ) / std
            else:
                normalized = volume.copy()

            patch_channels.append(
                normalized[
                    x:x + px,
                    y:y + py,
                    z:z + pz,
                ]
            )

        image_patch = np.stack(
            patch_channels, axis=0
        ).copy()

        with torch.no_grad():
            tensor = torch.from_numpy(
                image_patch
            ).float().unsqueeze(0).to(device)

            logits = model(tensor)
            prediction_patch = (
                torch.argmax(logits, dim=1)[0]
                .cpu()
                .numpy()
            )

        prediction_volume = prediction_patch

        # Display the central slice of the predicted patch.
        slice_index = pz // 2

        display_volumes = {
            name: volumes[name][
                x:x + px,
                y:y + py,
                z:z + pz,
            ]
            for name in MODALITIES
        }

        display_label = label_volume[
            x:x + px,
            y:y + py,
            z:z + pz,
        ]

    else:
        print(
            "No hybrid checkpoint found. "
            "Showing modalities and ground truth only."
        )
        display_volumes = volumes
        display_label = label_volume

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))

    for axis, name in zip(axes.flat[:4], MODALITIES):
        axis.imshow(
            normalize_for_display(
                display_volumes[name][:, :, slice_index]
            ).T,
            cmap="gray",
            origin="lower",
        )
        axis.set_title(name)
        axis.axis("off")

    axes.flat[4].imshow(
        display_label[:, :, slice_index].T,
        cmap="viridis",
        vmin=0,
        vmax=3,
        origin="lower",
        interpolation="nearest",
    )
    axes.flat[4].set_title("Ground-truth segmentation")
    axes.flat[4].axis("off")

    if prediction_volume is not None:
        axes.flat[5].imshow(
            prediction_volume[:, :, slice_index].T,
            cmap="viridis",
            vmin=0,
            vmax=3,
            origin="lower",
            interpolation="nearest",
        )
        axes.flat[5].set_title("Hybrid prediction (pilot)")
    else:
        axes.flat[5].axis("off")
        axes.flat[5].text(
            0.5,
            0.5,
            "No trained hybrid checkpoint",
            ha="center",
            va="center",
            transform=axes.flat[5].transAxes,
        )

    axes.flat[5].axis("off")

    figure.suptitle(
        f"BraTS-GLI case: {case['id']}",
        fontsize=14,
    )
    figure.tight_layout()

    output_dir = PROJECT_ROOT / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"{case['id']}_hybrid_visualization.png"
    )

    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print("Saved visualization:", output_path)

    plt.show()


if __name__ == "__main__":
    main()

