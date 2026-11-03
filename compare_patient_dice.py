import json
import numpy as np
import evaluate_full_volume as ev
from pathlib import Path

root = Path(r"C:\BrainTumorQuantum")
heldout = json.loads(
    (root / "results" / "comparison" / "heldout_test_patients.json")
    .read_text(encoding="utf-8")
)["patient_ids"]

cases = {c["id"]: c for c in ev.discover_cases()}
pid = heldout[0]
device = ev.select_device()

baseline, hybrid = ev.load_models(device)
image, truth = ev.load_volume(cases[pid])

for name, model in [("BASELINE", baseline), ("HYBRID", hybrid)]:
    pred, truth = ev.predict_volume(model, image, device, truth)
    print(f"\n{name} — {pid}")

    for cls in (1, 2, 3):
        p = pred == cls
        t = truth == cls
        denominator = int(p.sum() + t.sum())
        dice = 2.0 * np.logical_and(p, t).sum() / denominator if denominator else 1.0
        print(
            f"Class {cls}: Dice={dice:.4f}, "
            f"predicted={int(p.sum()):,}, truth={int(t.sum()):,}"
        )

    del pred

print("\nDone. No files were changed.")
