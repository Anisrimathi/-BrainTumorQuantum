from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_ROOT = Path(r"C:\BrainTumorQuantum\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
MODALITIES = {"T1": "t1n", "T1ce": "t1c", "T2": "t2w", "FLAIR": "t2f"}

def find_cases(root=DATASET_ROOT):
    cases = []
    for folder in sorted(root.glob("BraTS-GLI-*")):
        if not folder.is_dir():
            continue
        files = {key: folder / f"{folder.name}-{suffix}.nii.gz" for key, suffix in MODALITIES.items()}
        seg = folder / f"{folder.name}-seg.nii.gz"
        if all(path.is_file() for path in files.values()) and seg.is_file():
            cases.append({"id": folder.name, "images": files, "seg": seg})
    return cases

def load_case(case):
    images = [nib.load(str(case["images"][name])).get_fdata(dtype=np.float32) for name in MODALITIES]
    label = nib.load(str(case["seg"])).get_fdata(dtype=np.float32)
    shapes = [image.shape for image in images] + [label.shape]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent volume shapes for {case['id']}: {shapes}")
    image = np.stack(images, axis=0)
    for channel in range(image.shape[0]):
        volume = image[channel]
        nonzero = volume != 0
        if np.any(nonzero):
            mean = volume[nonzero].mean()
            std = volume[nonzero].std()
            if std > 0:
                image[channel][nonzero] = (volume[nonzero] - mean) / std
    return image, label

if __name__ == "__main__":
    cases = find_cases()
    print("Valid cases:", len(cases))
    if not cases:
        raise SystemExit("No complete cases found. Check DATASET_ROOT and filenames.")
    case = cases[0]
    image, label = load_case(case)
    print("First case:", case["id"])
    print("Image shape [4, X, Y, Z]:", image.shape)
    print("Label shape:", label.shape)
    print("Label values:", np.unique(label))
    print("Modality order:", list(MODALITIES))
