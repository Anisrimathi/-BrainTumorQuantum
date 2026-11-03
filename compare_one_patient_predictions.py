import json
from pathlib import Path
import numpy as np
import evaluate_full_volume as ev

root = Path(r"C:\BrainTumorQuantum")
heldout = json.loads(
    (root / "results" / "comparison" / "heldout_test_patients.json")
    .read_text(encoding="utf-8")
)["patient_ids"]

cases = {case["id"]: case for case in ev.discover_cases()}
patient_id = heldout[0]
device = ev.select_device()

print("Device:", device)
print("Patient:", patient_id)

baseline, hybrid = ev.load_models(device)
image, truth = ev.load_volume(cases[patient_id])

def report(name, pred, truth):
    print(f"\n{name}")
    for cls in range(4):
        gt_count = int(np.count_nonzero(truth == cls))
        pred_count = int(np.count_nonzero(pred == cls))
        print(
            f"Class {cls}: truth={gt_count:,}, "
            f"prediction={pred_count:,}"
        )

base_pred, truth = ev.predict_volume(baseline, image, device, truth)
report("BASELINE", base_pred, truth)

hybrid_pred, truth = ev.predict_volume(hybrid, image, device, truth)
report("HYBRID", hybrid_pred, truth)

print("\nComparison complete. No files were changed.")
