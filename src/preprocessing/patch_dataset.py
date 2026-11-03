
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

MODALITIES = {
    "T1": "t1n",
    "T1ce": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
}

PATCH_SIZE = (32, 32, 32)


def discover_cases():
    cases = []

    for folder in sorted(DATASET_ROOT.glob("BraTS-GLI-*")):
        if not folder.is_dir():
            continue

        images = {
            name: folder / f"{folder.name}-{suffix}.nii.gz"
            for name, suffix in MODALITIES.items()
        }
        seg = folder / f"{folder.name}-seg.nii.gz"

        if all(p.is_file() for p in images.values()) and seg.is_file():
            cases.append({
                "id": folder.name,
                "images": images,
                "seg": seg,
            })

    return cases


def load_volume(case):
    channels = []

    for name in MODALITIES:
        volume = nib.load(str(case["images"][name])).get_fdata(
            dtype=np.float32
        )

        mask = volume != 0
        if mask.any():
            mean = volume[mask].mean()
            std = volume[mask].std()
            if std > 1e-6:
                volume[mask] = (volume[mask] - mean) / std

        channels.append(volume)

    image = np.stack(channels, axis=0)
    label = nib.load(str(case["seg"])).get_fdata(dtype=np.float32)

    if image.shape[1:] != label.shape:
        raise ValueError(
            f"Shape mismatch for {case['id']}: "
            f"{image.shape[1:]} vs {label.shape}"
        )

    rounded = np.rint(label)
    if not np.allclose(label, rounded):
        raise ValueError(f"Non-integer labels in {case['id']}")

    label = rounded.astype(np.int64)
    allowed = {0, 1, 2, 3}
    actual = set(np.unique(label).tolist())

    if not actual.issubset(allowed):
        raise ValueError(
            f"Unexpected labels in {case['id']}: {sorted(actual)}"
        )

    return image, label


class BraTSPatchDataset(Dataset):
    def __init__(
        self,
        cases,
        patches_per_case=8,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=42,
    ):
        self.cases = cases
        self.patches_per_case = patches_per_case
        self.patch_size = patch_size
        self.tumor_probability = tumor_probability
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cases) * self.patches_per_case

    def __getitem__(self, index):
        case_index = index // self.patches_per_case
        case = self.cases[case_index]
        image, label = load_volume(case)

        px, py, pz = self.patch_size
        shape = label.shape

        # With high probability, center the patch around a tumor voxel.
        tumor_coords = np.argwhere(label > 0)

        use_tumor = (
            len(tumor_coords) > 0
            and self.rng.random() < self.tumor_probability
        )

        if use_tumor:
            cx, cy, cz = tumor_coords[
                self.rng.integers(0, len(tumor_coords))
            ]
            starts = [
                int(np.clip(cx - px // 2, 0, shape[0] - px)),
                int(np.clip(cy - py // 2, 0, shape[1] - py)),
                int(np.clip(cz - pz // 2, 0, shape[2] - pz)),
            ]
        else:
            starts = [
                int(self.rng.integers(0, dim - patch + 1))
                for dim, patch in zip(shape, self.patch_size)
            ]

        x, y, z = starts
        image_patch = image[
            :, x:x + px, y:y + py, z:z + pz
        ].copy()
        label_patch = label[
            x:x + px, y:y + py, z:z + pz
        ].copy()

        return (
            torch.from_numpy(image_patch).float(),
            torch.from_numpy(label_patch).long(),
        )


if __name__ == "__main__":
    cases = discover_cases()
    print("Valid cases:", len(cases))

    if not cases:
        raise SystemExit(f"No complete cases found at {DATASET_ROOT}")

    dataset = BraTSPatchDataset(
        cases=cases[:2],
        patches_per_case=2,
        patch_size=PATCH_SIZE,
        seed=42,
    )

    for i in range(len(dataset)):
        image, label = dataset[i]
        print(
            f"Patch {i + 1}: image={tuple(image.shape)}, "
            f"label={tuple(label.shape)}, "
            f"labels={torch.unique(label).tolist()}, "
            f"tumor_voxels={int((label > 0).sum())}"
        )
