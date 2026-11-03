import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from preprocessing.patch_dataset import BraTSPatchDataset, discover_cases

ROOT = Path(r"C:\BrainTumorQuantum")
SPLIT_FILE = ROOT / "results" / "comparison" / "shared_patient_split.json"
OUTPUT_FILE = ROOT / "results" / "comparison" / "validation_sampling_diagnostic.json"

PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 2
MAX_CASES = 4
SEED = 42


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    val_ids = split["validation_patient_ids"]
    all_cases = discover_cases()

    # Match split IDs to the actual case objects using their path/name.
    case_by_id = {}
    for case in all_cases:
        if isinstance(case, dict):
            candidates = [
                str(case.get(k, "")) for k in
                ("id", "case_id", "patient_id", "name", "path")
            ]
        else:
            candidates = [str(case), Path(str(case)).name]

        for candidate in candidates:
            if candidate:
                case_by_id[Path(candidate).name] = case
                case_by_id[candidate] = case

    selected_cases = []
    unmatched = []

    for patient_id in val_ids:
        case = case_by_id.get(str(patient_id))
        if case is None:
            case = case_by_id.get(Path(str(patient_id)).name)
        if case is None:
            unmatched.append(str(patient_id))
        elif len(selected_cases) < MAX_CASES:
            selected_cases.append(case)

    if unmatched:
        print("Note: some validation IDs did not directly match discovered cases:")
        print(unmatched[:10])

    if not selected_cases:
        raise RuntimeError(
            "No validation cases matched. Inspect discover_cases() output "
            "and the IDs in shared_patient_split.json."
        )

    print(f"Matched validation patients: {len(selected_cases)}")
    print("No model training or checkpoint files will be changed.")

    def summarize(probability):
        dataset = BraTSPatchDataset(
            cases=selected_cases,
            patches_per_case=PATCHES_PER_CASE,
            patch_size=PATCH_SIZE,
            tumor_probability=probability,
            seed=SEED,
        )

        counts = Counter()
        total_voxels = 0
        tumor_free_patches = 0

        for i in range(len(dataset)):
            # Dataset owns its seeded RNG; resetting global RNGs is unnecessary.
            image, label = dataset[i]
            arr = label.cpu().numpy()
            counts.update(arr.reshape(-1).tolist())
            total_voxels += arr.size
            tumor_free_patches += int(not np.isin(arr, [1, 2, 3]).any())

        return {
            "tumor_probability": probability,
            "patches_checked": len(dataset),
            "total_voxels": total_voxels,
            "class_voxel_counts": {
                str(c): int(counts[c]) for c in range(4)
            },
            "class_percentages": {
                str(c): round(100 * counts[c] / max(total_voxels, 1), 3)
                for c in range(4)
            },
            "tumor_free_patches": tumor_free_patches,
        }

    # Use the same selected patients for both sampling strategies.
    results = {
        "seed": SEED,
        "patch_size": list(PATCH_SIZE),
        "validation_patients_requested": [
            str(x) for x in val_ids[:MAX_CASES]
        ],
        "matched_validation_patients": len(selected_cases),
        "tumor_focused_sampling": summarize(0.8),
        "uniform_sampling": summarize(0.0),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nDIAGNOSTIC RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
