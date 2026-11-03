
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(r"C:\BrainTumorQuantum")
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.attention_unet import AttentionUNet3D
from models.hybrid_quantum_unet import HybridQuantumUNet
from preprocessing.patch_dataset import BraTSPatchDataset, discover_cases


CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
OUTPUT_DIR = ROOT / "results" / "comparison"
OUTPUT_FILE = OUTPUT_DIR / "baseline_vs_hybrid_pilot.json"

SEED = 42
NUM_CASES = 10
VAL_FRACTION = 0.20
PATCH_SIZE = (32, 32, 32)
BATCH_SIZE = 1
NUM_CLASSES = 4
BASE_CHANNELS = 8


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device):
    # These are checkpoints created by this project and are expected
    # to contain model_state_dict.
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {path}")

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise RuntimeError(
            f"No 'model_state_dict' in checkpoint: {path}"
        )

    return checkpoint, state


def calculate_dice(prediction, target, num_classes=4):
    scores = {}

    for class_id in range(1, num_classes):
        pred_mask = prediction == class_id
        target_mask = target == class_id

        pred_count = int(pred_mask.sum())
        target_count = int(target_mask.sum())

        if pred_count + target_count == 0:
            # Mark absent-in-both classes as undefined rather than
            # automatically awarding a perfect score.
            score = None
        else:
            intersection = int((pred_mask & target_mask).sum())
            score = (2.0 * intersection) / (
                pred_count + target_count
            )

        scores[f"Label_{class_id}"] = score

    valid_scores = [
        value for value in scores.values()
        if value is not None
    ]
    scores["mean_foreground_dice"] = (
        float(np.mean(valid_scores)) if valid_scores else None
    )

    return scores


def evaluate_model(name, model, loader, device):
    model.eval()
    per_class_scores = {i: [] for i in range(1, NUM_CLASSES)}
    mean_scores = []
    elapsed = 0.0
    patch_count = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "xpu":
                torch.xpu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            logits = model(images)

            if device.type == "xpu":
                torch.xpu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            elapsed += time.perf_counter() - start

            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            for batch_index in range(predictions.shape[0]):
                result = calculate_dice(
                    predictions[batch_index],
                    targets[batch_index],
                    NUM_CLASSES,
                )

                for class_id in range(1, NUM_CLASSES):
                    value = result[f"Label_{class_id}"]
                    if value is not None:
                        per_class_scores[class_id].append(value)

                if result["mean_foreground_dice"] is not None:
                    mean_scores.append(result["mean_foreground_dice"])

                patch_count += 1

    summary = {
        "model": name,
        "patch_count": patch_count,
        "mean_inference_seconds_per_patch": (
            elapsed / patch_count if patch_count else None
        ),
        "total_inference_seconds": elapsed,
        "dice": {},
    }

    for class_id, values in per_class_scores.items():
        summary["dice"][f"Label_{class_id}"] = (
            float(np.mean(values)) if values else None
        )

    summary["dice"]["mean_foreground_dice"] = (
        float(np.mean(mean_scores)) if mean_scores else None
    )

    return summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("Evaluation device:", device)

    baseline_path = CHECKPOINT_DIR / "attention_unet_pilot_best.pth"
    hybrid_path = CHECKPOINT_DIR / "hybrid_quantum_unet_ring_pilot.pth"

    for path in (baseline_path, hybrid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    cases = discover_cases()
    random.Random(SEED).shuffle(cases)
    cases = cases[:min(NUM_CASES, len(cases))]

    if len(cases) < 2:
        raise RuntimeError("At least two complete cases are required.")

    split = max(1, int(len(cases) * (1 - VAL_FRACTION)))
    if split >= len(cases):
        split = len(cases) - 1

    val_cases = cases[split:]
    print("Validation cases:", len(val_cases))
    print("Validation patient IDs:", [case["id"] for case in val_cases])

    # A deterministic, fixed patch dataset is shared by both models.
    validation_dataset = BraTSPatchDataset(
        val_cases,
        patches_per_case=1,
        patch_size=PATCH_SIZE,
        seed=SEED + 1,
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    baseline_checkpoint, baseline_state = load_checkpoint(
        baseline_path, device
    )
    baseline = AttentionUNet3D(
        in_channels=4,
        num_classes=4,
        base_channels=BASE_CHANNELS,
    )
    baseline.load_state_dict(baseline_state, strict=True)
    baseline = baseline.to(device)

    hybrid_checkpoint, hybrid_state = load_checkpoint(
        hybrid_path, device
    )
    strategy = hybrid_checkpoint.get("strategy", "ring")
    hybrid = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=BASE_CHANNELS,
        strategy=strategy,
    )
    hybrid.load_state_dict(hybrid_state, strict=True)
    hybrid = hybrid.to(device)

    results = {
        "evaluation_type": "pilot fixed random patches",
        "warning": (
            "Preliminary patch-level evaluation only. "
            "Not full-volume segmentation or clinical validation."
        ),
        "device": str(device),
        "seed": SEED,
        "patch_size": list(PATCH_SIZE),
        "validation_cases": len(val_cases),
        "models": [],
    }

    print("\nEvaluating baseline...")
    results["models"].append(
        evaluate_model("AttentionUNet3D", baseline, loader, device)
    )

    print("\nEvaluating hybrid...")
    results["models"].append(
        evaluate_model(
            f"HybridQuantumUNet ({strategy})", hybrid, loader, device
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nComparison results:")
    print(json.dumps(results, indent=2))
    print("\nSaved report:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
