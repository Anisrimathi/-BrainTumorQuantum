
from pathlib import Path
import json
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(r"C:\BrainTumorQuantum")
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.patch_dataset import (
    discover_cases,
    BraTSPatchDataset,
)
from models.hybrid_quantum_unet import (
    HybridQuantumUNet,
    select_device,
)

# Extended experiment configuration
SEED = 42
NUM_CASES = 40
TRAIN_FRACTION = 0.80
PATCHES_PER_CASE = 4
PATCH_SIZE = (32, 32, 32)
EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
ENTANGLEMENT = "ring"

CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
RESULTS_DIR = ROOT / "results"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "hybrid_quantum_unet_ring_final_v2.pth"
)
HISTORY_PATH = RESULTS_DIR / "hybrid_ring_final_v2_history.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)

def dice_loss(logits, target, num_classes=4, eps=1e-6):
    probs = torch.softmax(logits, dim=1)

    one_hot = torch.nn.functional.one_hot(
        target.long(), num_classes
    ).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)

    dice = (2 * intersection + eps) / (denominator + eps)

    # Match the baseline: exclude background from Dice loss.
    return 1 - dice[1:].mean()


def calculate_loss(logits, labels, ce):
    return dice_loss(logits, labels) + ce(logits, labels)


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print("Device:", device)
    print("Quantum simulator: PennyLane default.qubit on CPU")
    print("Entanglement:", ENTANGLEMENT)
    print("Cases:", NUM_CASES)
    print("Epochs:", EPOCHS)
    print("Patches per case:", PATCHES_PER_CASE)

    if CHECKPOINT_PATH.exists() or HISTORY_PATH.exists():
        raise FileExistsError(
            "Extended-run output already exists. "
            "Rename or back up these files before rerunning:\n"
            f"{CHECKPOINT_PATH}\n{HISTORY_PATH}"
        )

    all_cases = list(discover_cases())
    if len(all_cases) < NUM_CASES:
        raise RuntimeError(
            f"Requested {NUM_CASES} cases but found {len(all_cases)}."
        )

    random.Random(SEED).shuffle(all_cases)
    selected_cases = all_cases[:NUM_CASES]

    split_index = int(NUM_CASES * TRAIN_FRACTION)
    train_cases = selected_cases[:split_index]
    val_cases = selected_cases[split_index:]

    train_ids = {case["id"] for case in train_cases}
    val_ids = {case["id"] for case in val_cases}

    if train_ids & val_ids:
        raise RuntimeError("Patient overlap detected.")

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

    ce = nn.CrossEntropyLoss()
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
            "train_patient_ids": sorted(train_ids),
            "validation_patient_ids": sorted(val_ids),
            "patch_size": list(PATCH_SIZE),
            "patches_per_case": PATCHES_PER_CASE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
        },
        "epochs": [],
    }

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = calculate_loss(logits, labels, ce)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Train {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = calculate_loss(logits, labels, ce)

                val_loss_sum += loss.item()
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "seconds": round(seconds, 2),
        }
        history["epochs"].append(record)

        print(json.dumps(record, indent=2), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

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

        history["total_seconds"] = round(
            time.time() - start_time, 2
        )
        history["best_validation_loss"] = best_val_loss

        HISTORY_PATH.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print("\nExtended training finished.")
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)
    print("Note: patch-level development experiment only.")


if __name__ == "__main__":
    main()




