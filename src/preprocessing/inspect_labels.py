
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(
    r"C:\BrainTumorQuantum\dataset"
    r"\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

case_ids = [
    "BraTS-GLI-00000-000",
    "BraTS-GLI-00001-000",
    "BraTS-GLI-00002-000",
]

for case_id in case_ids:
    seg_path = ROOT / case_id / f"{case_id}-seg.nii.gz"

    if not seg_path.exists():
        print(f"\n{case_id}: segmentation file not found")
        continue

    seg = nib.load(str(seg_path)).get_fdata()
    labels, counts = np.unique(seg, return_counts=True)

    print(f"\nCase: {case_id}")
    print("Shape:", seg.shape)
    print("Label voxel counts:", {
        str(label): int(count)
        for label, count in zip(labels, counts)
    })

    tumor_per_slice = np.count_nonzero(seg, axis=(0, 1))
    nonempty_slices = np.flatnonzero(tumor_per_slice)

    if len(nonempty_slices):
        best_slice = int(
            nonempty_slices[np.argmax(tumor_per_slice[nonempty_slices])]
        )
        print("Slices containing tumor:", len(nonempty_slices))
        print("Slice with most tumor:", best_slice)
        print("Tumor voxels on that slice:", int(tumor_per_slice[best_slice]))
    else:
        print("WARNING: No non-background voxels found.")
