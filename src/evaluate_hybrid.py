
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
    / "hybrid_quantum_unet_ring_pilot.pth"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "hybrid_ring_pilot_evaluation.json"
)

SEED = 42
NUM_CASES = 10
PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 1


def calculate_metrics(prediction, target, class_id):
    predicted_mask = prediction == class_id
    target_mask = target == class_id

    intersection = np.logical_and(
        predicted_mask, target_mask
    ).sum()

    predicted_count = predicted_mask.sum()
    target_count = target_mask.sum()

    denominator = predicted_count + target_count

    dice = (
        1.0
        if denominator == 0
        else 2.0 * intersection / denominator
    )

    union = np.logical_or(
        predicted_mask, target_mask
    ).sum()

    iou = 1.0 if union == 0 else intersection / union

    precision = (
        1.0
        if predicted_count == 0 and target_count == 0
        else (
            0.0
            if predicted_count == 0
            else intersection / predicted_count
        )
    )

    recall = (
        1.0
        if target_count == 0 and predicted_count == 0
        else (
            0.0
            if target_count == 0
            else intersection / target_count
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Run src\\train_hybrid.py first."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = select_device()
    print("Evaluation device:", device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    strategy = checkpoint.get("strategy", "ring")

    model = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy=strategy,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    cases = discover_cases()
    if len(cases) < NUM_CASES:
        raise RuntimeError(
            f"Expected at least {NUM_CASES} complete cases."
        )

    # Reproduce the same deterministic case selection and split
    # used by the pilot training script.
    import random
    rng = random.Random(SEED)
    rng.shuffle(cases)
    selected = cases[:NUM_CASES]

    split_index = int(NUM_CASES * 0.8)
    val_cases = selected[split_index:]

    dataset = BraTSPatchDataset(
        val_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        seed=SEED + 1,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    per_class = {1: [], 2: [], 3: []}
    timings = []

    with torch.no_grad():
        for index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            start = time.perf_counter()
            logits = model(images)
            if device.type == "xpu":
                torch.xpu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            timings.append(elapsed)

            predictions = torch.argmax(logits, dim=1)[0]
            prediction_np = predictions.cpu().numpy()
            target_np = labels[0].cpu().numpy()

            for class_id in (1, 2, 3):
                per_class[class_id].append(
                    calculate_metrics(
                        prediction_np,
                        target_np,
                        class_id,
                    )
                )

            print(
                f"Validation patch {index}/{len(loader)} "
                f"inference={elapsed:.2f}s",
                flush=True,
            )

    results = {
        "checkpoint": str(CHECKPOINT),
        "evaluation_type": "pilot random patches",
        "validation_cases": len(val_cases),
        "patch_size": list(PATCH_SIZE),
        "strategy": strategy,
        "classes": {},
        "mean_inference_seconds_per_patch": float(
            np.mean(timings)
        ),
        "total_inference_seconds": float(np.sum(timings)),
    }

    for class_id, name in {
        1: "Label_1",
        2: "Label_2",
        3: "Label_3",
    }.items():
        metric_names = (
            "dice",
            "iou",
            "precision",
            "recall",
        )

        results["classes"][name] = {
            metric: float(
                np.mean([
                    item[metric]
                    for item in per_class[class_id]
                ])
            )
            for metric in metric_names
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation summary:")
    print(json.dumps(results, indent=2))
    print("\nSaved evaluation:", OUTPUT_FILE)
    print(
        "\nThese are preliminary patch-level metrics, "
        "not full-volume clinical evaluation results."
    )


if __name__ == "__main__":
    main()
