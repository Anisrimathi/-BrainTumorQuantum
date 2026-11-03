
from pathlib import Path
import sys
import json
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.patch_dataset import discover_cases, BraTSPatchDataset
from models.attention_unet import AttentionUNet3D
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device

BASELINE_PATH = ROOT / "results/checkpoints/attention_unet_pilot_best.pth"
HYBRID_PATH = ROOT / "results/checkpoints/hybrid_quantum_unet_ring_extended.pth"
HISTORY_PATH = ROOT / "results/hybrid_ring_extended_history.json"
OUTPUT_PATH = ROOT / "results/comparison/baseline_vs_hybrid_extended.json"

PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 4


def metrics(pred, target, class_id):
    p = pred == class_id
    t = target == class_id
    intersection = int(np.logical_and(p, t).sum())
    pc = int(p.sum())
    tc = int(t.sum())
    union = int(np.logical_or(p, t).sum())

    return {
        "dice": 1.0 if pc + tc == 0 else 2 * intersection / (pc + tc),
        "iou": 1.0 if union == 0 else intersection / union,
        "precision": (
            1.0 if pc == 0 and tc == 0
            else 0.0 if pc == 0
            else intersection / pc
        ),
        "recall": (
            1.0 if pc == 0 and tc == 0
            else 0.0 if tc == 0
            else intersection / tc
        ),
    }


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def average(items):
    return {
        key: float(np.mean([item[key] for item in items]))
        for key in ("dice", "iou", "precision", "recall")
    }


def main():
    for path in (BASELINE_PATH, HYBRID_PATH, HISTORY_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_ckpt = torch.load(
        BASELINE_PATH, map_location="cpu", weights_only=True
    )
    hybrid_ckpt = torch.load(
        HYBRID_PATH, map_location="cpu", weights_only=True
    )
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    cfg = history["configuration"]

    if hybrid_ckpt.get("configuration") != cfg:
        raise ValueError("Hybrid checkpoint/history configuration mismatch.")

    validation_ids = cfg["validation_patient_ids"]
    all_cases = {case["id"]: case for case in discover_cases()}
    missing = [pid for pid in validation_ids if pid not in all_cases]
    if missing:
        raise RuntimeError(f"Missing validation patients: {missing}")

    cases = [all_cases[pid] for pid in validation_ids]

    # Build the patch list once. Both models consume the same tensors.
    dataset = BraTSPatchDataset(
        cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=cfg["seed"] + 1,
    )
    fixed_patches = [dataset[i] for i in range(len(dataset))]
    print(f"Prepared {len(fixed_patches)} fixed patches for both models.")

    device = select_device()
    print("Inference device:", device)

    baseline = AttentionUNet3D(
        in_channels=4, num_classes=4, base_channels=8
    )
    baseline.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline = baseline.to(device).eval()

    hybrid = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy=hybrid_ckpt["strategy"],
    )
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    hybrid = hybrid.to(device).eval()

    models = {"baseline_attention_unet": baseline, "hybrid_quantum_unet": hybrid}
    class_names = {1: "Label_1", 2: "Label_2", 3: "Label_3"}
    output = {
        "evaluation_type": "same fixed random patches; exploratory comparison",
        "validation_patient_ids": validation_ids,
        "patches_per_patient": PATCHES_PER_CASE,
        "total_patches": len(fixed_patches),
        "patch_size": list(PATCH_SIZE),
        "device": str(device),
        "models": {},
        "limitations": [
            "Patch-level metrics, not full-volume segmentation metrics.",
            "Models were trained on different training sets.",
            "This comparison cannot establish causal quantum advantage.",
            "Empty target and prediction masks receive Dice/IoU of 1.",
        ],
    }

    for model_name, model in models.items():
        per_class = {class_id: [] for class_id in class_names}
        per_patient = {pid: {class_id: [] for class_id in class_names}
                       for pid in validation_ids}
        timings = []

        with torch.no_grad():
            for index, (image, label) in enumerate(fixed_patches):
                patient_id = validation_ids[index // PATCHES_PER_CASE]
                image = image.unsqueeze(0).to(device)
                label_np = label.numpy()

                sync(device)
                start = time.perf_counter()
                logits = model(image)
                sync(device)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)

                pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy()

                for class_id in class_names:
                    item = metrics(pred_np, label_np, class_id)
                    per_class[class_id].append(item)
                    per_patient[patient_id][class_id].append(item)

                print(
                    f"{model_name}: patch {index + 1}/"
                    f"{len(fixed_patches)} ({elapsed:.3f}s)",
                    flush=True,
                )

        output["models"][model_name] = {
            "checkpoint": str(
                BASELINE_PATH if model_name == "baseline_attention_unet"
                else HYBRID_PATH
            ),
            "checkpoint_epoch": (
                baseline_ckpt["epoch"] if model_name == "baseline_attention_unet"
                else hybrid_ckpt["epoch"]
            ),
            "aggregate_patch_metrics": {
                class_names[cid]: average(items)
                for cid, items in per_class.items()
            },
            "per_patient_patch_metrics": {
                pid: {
                    class_names[cid]: average(items)
                    for cid, items in class_dict.items()
                }
                for pid, class_dict in per_patient.items()
            },
            "mean_inference_seconds_per_patch": float(np.mean(timings)),
            "total_inference_seconds": float(np.sum(timings)),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nAggregate comparison:")
    for name, details in output["models"].items():
        print(f"\n{name}")
        print(json.dumps(details["aggregate_patch_metrics"], indent=2))
        print("Mean inference seconds/patch:",
              details["mean_inference_seconds_per_patch"])
    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
