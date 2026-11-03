
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

DATASET_ROOT = Path(
    r"C:\BrainTumorQuantum\dataset"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

MODALITIES = {
    "T1": "t1n",
    "T1ce": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
}


def main():
    cases = []

    for folder in sorted(DATASET_ROOT.glob("BraTS-GLI-*")):
        if not folder.is_dir():
            continue

        paths = {
            name: folder / f"{folder.name}-{suffix}.nii.gz"
            for name, suffix in MODALITIES.items()
        }
        seg_path = folder / f"{folder.name}-seg.nii.gz"

        if all(p.is_file() for p in paths.values()) and seg_path.is_file():
            cases.append((folder.name, paths, seg_path))

    if not cases:
        raise RuntimeError(
            f"No complete MRI cases found in {DATASET_ROOT}"
        )

    case_id, paths, seg_path = cases[0]
    print("Visualizing:", case_id)

    images = {
        name: nib.load(str(path)).get_fdata(dtype=np.float32)
        for name, path in paths.items()
    }
    seg = nib.load(str(seg_path)).get_fdata()

    shapes = [v.shape for v in images.values()] + [seg.shape]
    if len(set(shapes)) != 1:
        raise ValueError(f"Volume shapes do not match: {shapes}")

    # Select the axial slice containing the most labeled voxels.
    tumor_counts = np.count_nonzero(seg, axis=(0, 1))
    slice_idx = int(np.argmax(tumor_counts))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for ax, (name, volume) in zip(axes[:4], images.items()):
        ax.imshow(volume[:, :, slice_idx].T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")

    axes[4].imshow(
        images["FLAIR"][:, :, slice_idx].T,
        cmap="gray",
        origin="lower",
    )
    mask = seg[:, :, slice_idx].T
    axes[4].imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap="jet",
        alpha=0.55,
        origin="lower",
        interpolation="nearest",
    )
    axes[4].set_title("FLAIR + Ground Truth")
    axes[4].axis("off")

    fig.suptitle(
        f"{case_id} | Slice {slice_idx} | "
        f"Labels: {np.unique(seg).tolist()}"
    )
    fig.tight_layout()

    output_dir = Path(r"C:\BrainTumorQuantum\results\plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_visualization.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("Saved visualization:", output_path)
    print("Segmentation labels:", np.unique(seg))
    plt.show()


if __name__ == "__main__":
    main()
