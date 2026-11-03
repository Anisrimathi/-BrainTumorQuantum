import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.patch_dataset import discover_cases, load_volume
from src.models.attention_unet import AttentionUNet3D
from src.models.hybrid_quantum_unet import HybridQuantumUNet, select_device

BASELINE_CKPT = ROOT / "results/comparison/fair_baseline_40/checkpoints/baseline_40_best.pth"
HYBRID_CKPT = ROOT / "results/comparison/fair_hybrid_40/checkpoints/hybrid_40_best.pth"
HELDOUT_FILE = ROOT / "results/comparison/heldout_test_patients.json"
OUTPUT_FILE = ROOT / "results/comparison/baseline_vs_hybrid_40_full_volume.json"

PATCH = 32
STRIDE = 24


def load_models(device):
    baseline_ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    hybrid_ckpt = torch.load(HYBRID_CKPT, map_location="cpu", weights_only=False)

    baseline_config = baseline_ckpt.get(
        "model_config",
        {"in_channels": 4, "num_classes": 4, "base_channels": 8},
    )
    baseline = AttentionUNet3D(**baseline_config)
    baseline.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline.to(device).eval()

    hybrid_config = hybrid_ckpt.get(
        "configuration",
        {"in_channels": 4, "num_classes": 4, "base_channels": 8},
    )
    hybrid_config = {
        key: hybrid_config[key]
        for key in ("in_channels", "num_classes", "base_channels")
        if key in hybrid_config
    }
    hybrid = HybridQuantumUNet(**hybrid_config, strategy=hybrid_ckpt.get("strategy", "ring"))
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    hybrid.to(device).eval()

    return baseline, hybrid


def starts(length):
    if length <= PATCH:
        return [0]
    values = list(range(0, length - PATCH + 1, STRIDE))
    if values[-1] != length - PATCH:
        values.append(length - PATCH)
    return values


def pad_volume(image, label):
    _, d, h, w = image.shape
    pd, ph, pw = max(0, PATCH-d), max(0, PATCH-h), max(0, PATCH-w)
    image = np.pad(image, ((0, 0), (0, pd), (0, ph), (0, pw)))
    label = np.pad(label, ((0, pd), (0, ph), (0, pw)))
    return image, label, (d, h, w)


@torch.inference_mode()
def predict_volume(model, image, device, label):
    image, label, original_shape = pad_volume(image, label)
    _, d, h, w = image.shape

    prob_sum = np.zeros((4, d, h, w), dtype=np.float32)
    count = np.zeros((d, h, w), dtype=np.uint16)

    total = len(starts(d)) * len(starts(h)) * len(starts(w))
    done = 0

    for z in starts(d):
        for y in starts(h):
            for x in starts(w):
                patch = image[:, z:z+PATCH, y:y+PATCH, x:x+PATCH]
                tensor = torch.from_numpy(
                    np.ascontiguousarray(patch[None])
                ).float().to(device)

                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].float().cpu().numpy()

                prob_sum[:, z:z+PATCH, y:y+PATCH, x:x+PATCH] += probabilities
                count[z:z+PATCH, y:y+PATCH, x:x+PATCH] += 1
                done += 1

                if done % 100 == 0 or done == total:
                    print(f"    Windows: {done}/{total}", flush=True)

                del tensor, logits

    prob_sum /= np.maximum(count[None], 1)
    prediction = np.argmax(prob_sum, axis=0)
    d0, h0, w0 = original_shape
    return prediction[:d0, :h0, :w0], label[:d0, :h0, :w0]


def binary_metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    denom = 2 * tp + fp + fn
    union = tp + fp + fn

    return {
        "dice": (2 * tp / denom) if denom else None,
        "iou": (tp / union) if union else None,
        "precision": (tp / (tp + fp)) if tp + fp else None,
        "recall": (tp / (tp + fn)) if tp + fn else None,
    }


def evaluate_patient(prediction, truth):
    metrics = {}
    for cls in (1, 2, 3):
        metrics[f"class_{cls}"] = binary_metrics(prediction == cls, truth == cls)

    regions = {
        "WT": (1, 2, 3),
        "TC": (1, 3),
        "ET": (3,),
    }
    for name, classes in regions.items():
        metrics[name] = binary_metrics(
            np.isin(prediction, classes),
            np.isin(truth, classes),
        )
    return metrics


def mean_metrics(patient_results):
    output = {}
    keys = patient_results[0].keys() if patient_results else []
    for key in keys:
        output[key] = {}
        for metric in ("dice", "iou", "precision", "recall"):
            values = [
                item[key][metric]
                for item in patient_results
                if item[key][metric] is not None
            ]
            output[key][metric] = float(np.mean(values)) if values else None
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patients", type=int, default=1)
    args = parser.parse_args()

    if args.max_patients < 1:
        raise ValueError("--max-patients must be at least 1")

    for path in (BASELINE_CKPT, HYBRID_CKPT, HELDOUT_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    heldout = json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))["patient_ids"]
    all_cases = {case["id"]: case for case in discover_cases()}
    missing = [pid for pid in heldout if pid not in all_cases]
    if missing:
        raise RuntimeError(f"Held-out cases not found in dataset: {missing}")

    selected = heldout[:args.max_patients]
    device = select_device()
    print("Device:", device, flush=True)
    print("Loading checkpoints...", flush=True)
    baseline, hybrid = load_models(device)
    print("Both checkpoints loaded.", flush=True)

    results = {
        "scope": "pilot subset" if len(selected) < len(heldout) else "full held-out set",
        "method": "whole-volume sliding-window inference; patch=32; stride=24; overlapping probabilities averaged",
        "heldout_count": len(heldout),
        "evaluated_count": len(selected),
        "patients": {},
    }

    for index, pid in enumerate(selected, 1):
        print(f"\nPatient {index}/{len(selected)}: {pid}", flush=True)
        image, truth = load_volume(all_cases[pid])
        print("  Volume shape:", tuple(truth.shape), flush=True)

        print("  Baseline inference...", flush=True)
        base_pred, truth = predict_volume(baseline, image, device, truth)
        results["patients"].setdefault(pid, {})["baseline"] = evaluate_patient(base_pred, truth)

        print("  Hybrid inference (quantum circuit may be slow)...", flush=True)
        hybrid_pred, truth = predict_volume(hybrid, image, device, truth)
        results["patients"][pid]["hybrid"] = evaluate_patient(hybrid_pred, truth)

        results["patients"][pid]["ground_truth_label_values"] = [
            int(v) for v in np.unique(truth)
        ]

        del image, truth, base_pred, hybrid_pred
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  Saved progress: {OUTPUT_FILE}", flush=True)

    baseline_rows = [v["baseline"] for v in results["patients"].values()]
    hybrid_rows = [v["hybrid"] for v in results["patients"].values()]
    results["summary"] = {
        "baseline_patient_macro_average": mean_metrics(baseline_rows),
        "hybrid_patient_macro_average": mean_metrics(hybrid_rows),
        "note": "Pilot results are preliminary. Metrics are averaged across patients; empty class/region cases are excluded from that metric's average.",
    }
    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nEvaluation finished.")
    print("Results:", OUTPUT_FILE)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
