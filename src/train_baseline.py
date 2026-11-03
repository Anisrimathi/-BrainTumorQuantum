
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.attention_unet import AttentionUNet3D
from preprocessing.patch_dataset import (
    BraTSPatchDataset,
    discover_cases,
)

ROOT = Path(r"C:\BrainTumorQuantum")
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"

SEED = 42
MAX_CASES = 20       # Pilot experiment; increase for full training
VAL_FRACTION = 0.20
PATCHES_PER_CASE = 2
PATCH_SIZE = (32, 32, 32)
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-4


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Device: Intel XPU")
        return torch.device("xpu")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def dice_loss(logits, target, num_classes=4, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(
        target, num_classes
    ).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)

    dice = (2 * intersection + eps) / (denominator + eps)

    # Exclude background from the Dice loss.
    return 1 - dice[1:].mean()


def calculate_loss(logits, labels, ce):
    # Equal-weight sum of Dice loss and multiclass cross-entropy.
    return dice_loss(logits, labels) + ce(logits, labels)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cases = discover_cases()
    if len(cases) < 2:
        raise RuntimeError("Need at least two complete patient cases.")

    random.shuffle(cases)
    cases = cases[:min(MAX_CASES, len(cases))]

    split = max(1, int(len(cases) * (1 - VAL_FRACTION)))
    if split >= len(cases):
        split = len(cases) - 1

    train_cases = cases[:split]
    val_cases = cases[split:]

    train_ids = {case["id"] for case in train_cases}
    val_ids = {case["id"] for case in val_cases}

    assert train_ids.isdisjoint(val_ids)

    print("Pilot cases:", len(cases))
    print("Training cases:", len(train_cases))
    print("Validation cases:", len(val_cases))
    print("Patient overlap:", len(train_ids & val_ids))

    train_data = BraTSPatchDataset(
        train_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=SEED,
    )
    val_data = BraTSPatchDataset(
        val_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=SEED + 1,
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = get_device()
    model = AttentionUNet3D(
        in_channels=4, num_classes=4, base_channels=8
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_path = CHECKPOINT_DIR / "attention_unet_pilot_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0

        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = calculate_loss(logits, labels, ce)
            loss.backward()
            optimizer.step()

            train_total += loss.item()

            if step % 5 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"train step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_total += calculate_loss(logits, labels, ce).item()

        train_loss = train_total / len(train_loader)
        val_loss = val_total / len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_case_ids": sorted(train_ids),
                    "val_case_ids": sorted(val_ids),
                    "model_config": {
                        "in_channels": 4,
                        "num_classes": 4,
                        "base_channels": 8,
                    },
                },
                best_path,
            )
            print("Saved best checkpoint:", best_path)

    print("Pilot finished. Best validation loss:", best_val_loss)
    print("This pilot is not a final research result.")


if __name__ == "__main__":
    main()
