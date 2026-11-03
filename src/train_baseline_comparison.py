
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.attention_unet import AttentionUNet3D
from preprocessing.patch_dataset import BraTSPatchDataset, discover_cases

ROOT = Path(r"C:\BrainTumorQuantum")
SPLIT_PATH = ROOT / "results" / "comparison" / "shared_patient_split.json"
OUTPUT_DIR = ROOT / "results" / "comparison" / "fair_baseline"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

SEED = 42
PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 2
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-4
MAX_TRAIN_PATIENTS = 8
MAX_VAL_PATIENTS = 2


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_loss(logits, target, num_classes=4, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(
        target.long(), num_classes
    ).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice[1:].mean()


def calculate_loss(logits, labels, ce):
    return dice_loss(logits, labels) + ce(logits, labels)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in discover_cases()}

    train_ids = split["train_patient_ids"][:MAX_TRAIN_PATIENTS]
    val_ids = split["validation_patient_ids"][:MAX_VAL_PATIENTS]
    test_ids = set(split["test_patient_ids"])

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Training/validation overlap.")
    if set(train_ids) & test_ids or set(val_ids) & test_ids:
        raise RuntimeError("Test-patient leakage detected.")

    missing = (set(train_ids) | set(val_ids)) - cases.keys()
    if missing:
        raise RuntimeError(f"Missing patient cases: {sorted(missing)}")

    train_cases = [cases[x] for x in train_ids]
    val_cases = [cases[x] for x in val_ids]

    train_data = BraTSPatchDataset(
        train_cases, patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE, tumor_probability=0.8, seed=SEED,
    )
    val_data = BraTSPatchDataset(
        val_cases, patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE, tumor_probability=0.8, seed=SEED + 1,
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    device = get_device()
    print("Device:", device)
    print("Training patients:", len(train_cases))
    print("Validation patients:", len(val_cases))
    print("Held-out test patients:", len(test_ids))
    print("Patches per patient:", PATCHES_PER_CASE)
    print("Epochs:", EPOCHS)

    model = AttentionUNet3D(
        in_channels=4, num_classes=4, base_channels=8
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ce = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "baseline_smoke_best.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0

        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = calculate_loss(model(images), labels, ce)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"train {step}/{len(train_loader)} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_sum += calculate_loss(model(images), labels, ce).item()

        train_loss = train_sum / len(train_loader)
        val_loss = val_sum / len(val_loader)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_loss": val_loss,
                "train_patient_ids": train_ids,
                "validation_patient_ids": val_ids,
                "configuration": {
                    "patch_size": list(PATCH_SIZE),
                    "patches_per_case": PATCHES_PER_CASE,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "seed": SEED,
                    "pilot_subset": True,
                },
            }, checkpoint_path)
            print("Saved:", checkpoint_path)

    (OUTPUT_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print("Baseline smoke test training finished.")
    print("This small-subset run is not a final result.")


if __name__ == "__main__":
    main()
