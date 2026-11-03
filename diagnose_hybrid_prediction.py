import json
from pathlib import Path
from collections import Counter

import numpy as np

import evaluate_full_volume as ev

ROOT = Path(r"C:\BrainTumorQuantum")
HELDOUT_FILE = ROOT / "results" / "comparison" / "heldout_test_patients.json"

heldout = json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))["patient_ids"]
cases = {case["id"]: case for case in ev.discover_cases()}

patient_id = heldout[0]
if patient_id not in cases:
    raise RuntimeError(f"Held-out patient not found: {patient_id}")

device = ev.select_device()
print("Device:", device)
print("Patient:", patient_id)

_, hybrid = ev.load_models(device)
image, truth = ev.load_volume(cases[patient_id])

print("Volume shape:", tuple(truth.shape))
prediction, truth = ev.predict_volume(hybrid, image, device, truth)

def report(name, arr):
    values, counts = np.unique(arr, return_counts=True)
    total = arr.size
    print(f"\n{name}:")
    for value, count in zip(values, counts):
        print(f"  Class {int(value)}: {int(count):,} voxels ({100*count/total:.3f}%)")

report("Ground truth", truth)
report("Hybrid prediction", prediction)

print("\nPrediction diagnostic complete. No files were changed.")
