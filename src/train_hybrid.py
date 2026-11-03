
from pathlib import Path
import json
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
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


# -------------------------
# Pilot configuration
# -------------------------

SEED = 42
NUM_CASES = 10
TRAIN_FRACTION = 0.8
PATCHES_PER_CASE = 1
PATCH_SIZE = (32, 32, 32)
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
ENTANGLEMENT = "ring"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "hybrid_quantum_unet_ring_pilot.pth"
)
HISTORY_PATH = RESULTS_DIR / "hybrid_ring_pilot_history.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def calculate_dice(logits, targets, num_classes=4):
    """Calculate foreground-class Dice on one batch."""
    predictions = torch.argmax(logits, dim=1)
    scores = []

    for class_id in range(1, num_classes):
        predicted_mask = predictions == class_id
        target_mask = targets == class_id

        denominator = (
            predicted_mask.sum() + target_mask.sum()
        ).float()

        if denominator.item() == 0:
            # No voxels of this class in either mask.
            scores.append(1.0)
        else:
            intersection = (
                predicted_mask & target_mask
            ).sum().float()

            scores.append(
                (2.0 * intersection / denominator).item()
            )

    return scores


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    dice_totals = np.zeros(3, dtype=np.float64)

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss in batch {batch_number}."
                )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        dice_values = calculate_dice(logits.detach(), labels)
        dice_totals += np.asarray(dice_values)

        print(
            f"  {'Train' if training else 'Val'} "
            f"batch {batch_number}/{len(loader)} "
            f"loss={loss.item():.4f} "
            f"Dice={np.mean(dice_values):.4f}",
            flush=True,
        )

    if total_batches == 0:
        raise RuntimeError("The data loader returned no batches.")

    return {
        "loss": total_loss / total_batches,
        "dice_class_1": float(dice_totals[0] / total_batches),
        "dice_class_2": float(dice_totals[1] / total_batches),
        "dice_class_3": float(dice_totals[2] / total_batches),
        "mean_foreground_dice": float(
            dice_totals.mean() / total_batches
        ),
    }


def main():
    set_seed(SEED)

    device = select_device()
    print("Selected device:", device)
    print("Quantum simulator: PennyLane default.qubit on CPU")
    print("Entanglement strategy:", ENTANGLEMENT)

    all_cases = discover_cases()
    print("Complete dataset cases:", len(all_cases))

    if len(all_cases) < NUM_CASES:
        raise RuntimeError(
            f"Need at least {NUM_CASES} complete cases."
        )

    # Select a reproducible subset, then split by patient.
    rng = random.Random(SEED)
    selected_cases = list(all_cases)
    rng.shuffle(selected_cases)
    selected_cases = selected_cases[:NUM_CASES]

    split_index = int(NUM_CASES * TRAIN_FRACTION)
    train_cases = selected_cases[:split_index]
    val_cases = selected_cases[split_index:]

    train_ids = {case["id"] for case in train_cases}
    val_ids = {case["id"] for case in val_cases}

    if train_ids & val_ids:
        raise RuntimeError("Patient leakage between train and validation.")

    print("Training patients:", len(train_cases))
    print("Validation patients:", len(val_cases))
    print("Patient overlap:", len(train_ids & val_ids))

    train_dataset = BraTSPatchDataset(
        train_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        seed=SEED,
    )

    val_dataset = BraTSPatchDataset(
        val_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        seed=SEED + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy=ENTANGLEMENT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "configuration": {
            "seed": SEED,
            "strategy": ENTANGLEMENT,
            "num_cases": NUM_CASES,
            "train_cases": len(train_cases),
            "validation_cases": len(val_cases),
            "patch_size": list(PATCH_SIZE),
            "patches_per_case": PATCHES_PER_CASE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
        },
        "epochs": [],
    }

    best_val_loss = float("inf")
    overall_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "seconds": round(epoch_seconds, 2),
        }
        history["epochs"].append(record)

        print("\nEpoch summary")
        print(json.dumps(record, indent=2))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val_loss,
                    "strategy": ENTANGLEMENT,
                    "configuration": history["configuration"],
                },
                CHECKPOINT_PATH,
            )

            print("Saved best checkpoint:", CHECKPOINT_PATH)

        HISTORY_PATH.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    history["total_seconds"] = round(
        time.time() - overall_start, 2
    )

    HISTORY_PATH.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\nPilot training completed.")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)
    print("Total seconds:", history["total_seconds"])
    print(
        "Important: these are pilot results, not final "
        "full-dataset evaluation results."
    )


if __name__ == "__main__":
    main()
