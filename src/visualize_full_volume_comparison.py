import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.patch_dataset import discover_cases, load_volume
from src.models.attention_unet import AttentionUNet3D
from src.models.hybrid_quantum_unet import HybridQuantumUNet, select_device
from src.evaluate_full_volume import (
    BASELINE_CKPT, HYBRID_CKPT, HELDOUT_FILE,
    load_models, predict_volume,
)

OUTPUT_DIR = ROOT / "results" / "visualizations" / "baseline_vs_hybrid_v2"
TARGET_PATIENT = "BraTS-GLI-00036-001"
MODALITY_INDEX = 3  # FLAIR, assuming load_volume order is T1, T1ce, T2, FLAIR

COLORS = ["black", "red", "green", "yellow"]
LABELS = ["Background", "Label 1", "Label 2", "Label 3"]


def display_image(volume):
    volume = np.asarray(volume, dtype=np.float32)
    mask = volume != 0
    if not mask.any():
        return np.zeros_like(volume)
    low, high = np.percentile(volume[mask], [1, 99])
    if high <= low:
        return np.zeros_like(volume)
    return np.clip((volume - low) / (high - low), 0, 1)


def overlay(axis, image, mask, title):
    axis.imshow(display_image(image).T, cmap="gray", origin="lower")
    colored = np.ma.masked_where(mask.T == 0, mask.T)
    axis.imshow(
        colored,
        cmap=ListedColormap(COLORS[1:]),
        vmin=1,
        vmax=3,
        alpha=0.55,
        origin="lower",
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")


def main():
    for path in (BASELINE_CKPT, HYBRID_CKPT, HELDOUT_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    import json
    heldout = json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))["patient_ids"]
    patient_id = TARGET_PATIENT if TARGET_PATIENT in heldout else heldout[0]

    cases = {case["id"]: case for case in discover_cases()}
    if patient_id not in cases:
        raise RuntimeError(f"Patient not found in dataset: {patient_id}")

    device = select_device()
    print("Device:", device, flush=True)
    print("Loading baseline and hybrid v2 models...", flush=True)
    baseline, hybrid = load_models(device)

    print("Loading patient:", patient_id, flush=True)
    image, truth = load_volume(cases[patient_id])
    print("Volume shape:", truth.shape, flush=True)

    print("Running full-volume baseline inference...", flush=True)
    base_pred, truth = predict_volume(baseline, image, device, truth)

    print("Running full-volume hybrid inference...", flush=True)
    hybrid_pred, truth = predict_volume(hybrid, image, device, truth)

    # Use the axial slice with the largest ground-truth tumor area.
    tumor_counts = np.count_nonzero(truth > 0, axis=(0, 1))
    slice_index = int(np.argmax(tumor_counts)) if tumor_counts.max() else truth.shape[2] // 2
    print("Display slice:", slice_index, flush=True)

    flair = image[MODALITY_INDEX, :, :, slice_index]
    gt = truth[:, :, slice_index]
    base = base_pred[:, :, slice_index]
    hybrid_mask = hybrid_pred[:, :, slice_index]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{patient_id}_slice_{slice_index}_comparison.png"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(display_image(flair).T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR MRI")
    axes[0].axis("off")
    overlay(axes[1], flair, gt, "Ground truth")
    overlay(axes[2], flair, base, "Baseline U-Net")
    overlay(axes[3], flair, hybrid_mask, "Hybrid quantum-classical U-Net v2")

    fig.suptitle(
        f"{patient_id} | Axial slice {slice_index} | Research comparison",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Save raw predictions too, so they can be reused in a later report/UI.
    np.savez_compressed(
        OUTPUT_DIR / f"{patient_id}_predictions.npz",
        ground_truth=truth.astype(np.uint8),
        baseline_prediction=base_pred.astype(np.uint8),
        hybrid_prediction=hybrid_pred.astype(np.uint8),
        flair_slice_index=np.int32(slice_index),
    )

    print("Saved comparison image:", output_path)
    print("Saved prediction volumes:", OUTPUT_DIR / f"{patient_id}_predictions.npz")
    print("Note: preliminary research output; not clinically validated.")


if __name__ == "__main__":
    main()
