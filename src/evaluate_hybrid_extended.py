
from pathlib import Path
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.patch_dataset import (
    discover_cases,
    BraTSPatchDataset,
)
from models.hybrid_quantum_unet import (
    HybridQuantumUNet,
    select_device,
)

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "hybrid_quantum_unet_ring_extended.pth"
)
HISTORY_FILE = (
    PROJECT_ROOT / "results" / "hybrid_ring_extended_history.json"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "results"
    / "hybrid_ring_extended_evaluation.json"
)

PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 4


def calculate_metrics(prediction, target, class_id):
    predicted = prediction == class_id
    actual = target == class_id

    intersection = int(np.logical_and(predicted, actual).sum())
    predicted_count = int(predicted.sum())
    actual_count = int(actual.sum())
    union = int(np.logical_or(predicted, actual).sum())

    return {
        "dice": (
            1.0 if predicted_count + actual_count == 0
            else 2.0 * intersection / (predicted_count + actual_count)
        ),
        "iou": 1.0 if union == 0 else intersection / union,
        "precision": (
            1.0 if predicted_count == 0 and actual_count == 0
            else 0.0 if predicted_count == 0
            else intersection / predicted_count
        ),
        "recall": (
            1.0 if actual_count == 0 and predicted_count == 0
            else 0.0 if actual_count == 0
            else intersection / actual_count
        ),
    }


def synchronize(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    for path in (CHECKPOINT, HISTORY_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    config = history["configuration"]

    if checkpoint.get("strategy") != config["strategy"]:
        raise ValueError("Checkpoint and history strategies do not match.")

    if checkpoint.get("configuration") != config:
        raise ValueError(
            "Checkpoint and history configurations differ. "
            "Stop rather than evaluate a potentially mismatched model."
        )

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = select_device()
    print("Evaluation device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Quantum strategy:", checkpoint["strategy"])

    model = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy=checkpoint["strategy"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_cases = {case["id"]: case for case in discover_cases()}
    validation_ids = config["validation_patient_ids"]

    missing = [patient_id for patient_id in validation_ids
               if patient_id not in all_cases]
    if missing:
        raise RuntimeError(f"Validation cases missing from dataset: {missing}")

    val_cases = [all_cases[patient_id] for patient_id in validation_ids]

    dataset = BraTSPatchDataset(
        val_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        seed=config["seed"] + 1,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    class_names = {1: "Label_1", 2: "Label_2", 3: "Label_3"}
    per_class = {class_id: [] for class_id in class_names}
    patient_metrics = {
        patient_id: {class_id: [] for class_id in class_names}
        for patient_id in validation_ids
    }
    timings = []
    patch_results = []

    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            patient_index = index // PATCHES_PER_CASE
            patient_id = validation_ids[patient_index]

            images = images.to(device)
            labels = labels.to(device)

            synchronize(device)
            start = time.perf_counter()
            logits = model(images)
            synchronize(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()
            target = labels[0].cpu().numpy()

            patch_metrics = {}
            for class_id, name in class_names.items():
                metrics = calculate_metrics(
                    prediction, target, class_id
                )
                per_class[class_id].append(metrics)
                patient_metrics[patient_id][class_id].append(metrics)
                patch_metrics[name] = metrics

            patch_results.append({
                "patch_index": index + 1,
                "patient_id": patient_id,
                "inference_seconds": elapsed,
                "classes": patch_metrics,
            })

            print(
                f"Patch {index + 1}/{len(loader)} | "
                f"{patient_id} | {elapsed:.2f}s",
                flush=True,
            )

    def average_metrics(items):
        return {
            key: float(np.mean([item[key] for item in items]))
            for key in ("dice", "iou", "precision", "recall")
        }

    patient_results = {}
    for patient_id, class_metrics in patient_metrics.items():
        patient_results[patient_id] = {
            class_names[class_id]: average_metrics(values)
            for class_id, values in class_metrics.items()
        }

    results = {
        "evaluation_type": "random tumor-aware patch-level evaluation",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "strategy": checkpoint["strategy"],
        "seed": config["seed"],
        "validation_patient_ids": validation_ids,
        "validation_patients": len(validation_ids),
        "patches_per_patient": PATCHES_PER_CASE,
        "total_patches": len(dataset),
        "patch_size": list(PATCH_SIZE),
        "aggregate_patch_metrics": {
            class_names[class_id]: average_metrics(metrics)
            for class_id, metrics in per_class.items()
        },
        "per_patient_patch_metrics": patient_results,
        "mean_inference_seconds_per_patch": float(np.mean(timings)),
        "total_inference_seconds": float(np.sum(timings)),
        "patch_results": patch_results,
        "limitations": [
            "Metrics are averages over randomly sampled 32x32x32 patches.",
            "They are not full-volume segmentation metrics.",
            "They are not clinical performance or evidence of quantum advantage.",
            "Empty target/prediction masks receive a score of 1 for Dice and IoU."
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\nAggregate patch-level metrics:")
    print(json.dumps(results["aggregate_patch_metrics"], indent=2))
    print("\nSaved complete evaluation:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
