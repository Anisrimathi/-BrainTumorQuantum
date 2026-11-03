
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.attention_unet import AttentionUNet3D
from preprocessing.patch_dataset import (
    BraTSPatchDataset,
    discover_cases,
)

ROOT = Path(r"C:\BrainTumorQuantum")
CHECKPOINT = (
    ROOT / "results" / "checkpoints"
    / "attention_unet_pilot_best.pth"
)

PATCH_SIZE = (32, 32, 32)
PATCHES_PER_CASE = 4


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def update_counts(counts, prediction, target, name, pred_mask, true_mask):
    pred_mask = pred_mask(prediction)
    true_mask = true_mask(target)

    tp = int((pred_mask & true_mask).sum().item())
    fp = int((pred_mask & ~true_mask).sum().item())
    fn = int((~pred_mask & true_mask).sum().item())

    counts[name]["tp"] += tp
    counts[name]["fp"] += fp
    counts[name]["fn"] += fn


def make_counts():
    names = [
        "Class 1", "Class 2", "Class 3",
        "WT", "TC", "ET",
    ]
    return {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in names
    }


def update_all(counts, prediction, target):
    regions = {
        "Class 1": (lambda x: x == 1, lambda x: x == 1),
        "Class 2": (lambda x: x == 2, lambda x: x == 2),
        "Class 3": (lambda x: x == 3, lambda x: x == 3),
        "WT": (lambda x: x > 0, lambda x: x > 0),
        "TC": (
            lambda x: (x == 1) | (x == 3),
            lambda x: (x == 1) | (x == 3),
        ),
        "ET": (lambda x: x == 3, lambda x: x == 3),
    }

    for name, (pred_fn, true_fn) in regions.items():
        update_counts(
            counts, prediction, target, name, pred_fn, true_fn
        )


def report(counts):
    print("\nPRELIMINARY PATCH-LEVEL METRICS")
    print("Region       Dice       IoU     Precision    Recall")

    for name, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]

        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn
        denom_precision = tp + fp
        denom_recall = tp + fn

        dice = (2 * tp / denom_dice) if denom_dice else float("nan")
        iou = (tp / denom_iou) if denom_iou else float("nan")
        precision = (
            tp / denom_precision if denom_precision else float("nan")
        )
        recall = tp / denom_recall if denom_recall else float("nan")

        print(
            f"{name:<12} {dice:>7.4f} {iou:>9.4f} "
            f"{precision:>12.4f} {recall:>10.4f}"
        )


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = get_device()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    config = checkpoint["model_config"]
    model = AttentionUNet3D(**{
        "in_channels": config["in_channels"],
        "num_classes": config["num_classes"],
        "base_channels": config["base_channels"],
    })
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    validation_ids = set(checkpoint["val_case_ids"])
    all_cases = discover_cases()
    val_cases = [
        case for case in all_cases if case["id"] in validation_ids
    ]

    if not val_cases:
        raise RuntimeError("Could not find checkpoint validation cases.")

    print("Evaluation device:", device)
    print("Validation cases:", len(val_cases))

    dataset = BraTSPatchDataset(
        val_cases,
        patches_per_case=PATCHES_PER_CASE,
        patch_size=PATCH_SIZE,
        tumor_probability=0.8,
        seed=123,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    counts = make_counts()

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(device)
            logits = model(images)
            prediction = logits.argmax(dim=1).cpu()
            labels = labels.cpu()

            update_all(counts, prediction, labels)

            if step % 4 == 0 or step == len(loader):
                print(f"Evaluated patches: {step}/{len(loader)}")

    report(counts)
    print(
        "\nNote: These are preliminary random-patch metrics, "
        "not full-volume clinical or research evaluation."
    )


if __name__ == "__main__":
    main()
