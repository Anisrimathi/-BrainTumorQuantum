
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.preprocessing.patch_dataset import BraTSPatchDataset, discover_cases
from src.models.hybrid_quantum_unet import HybridQuantumUNet, select_device

ROOT = Path(".")
split = json.loads(
    (ROOT / "results/comparison/shared_patient_split.json").read_text()
)

cases = {case["id"]: case for case in discover_cases()}
patient_ids = split["validation_patient_ids"][:2]
selected_cases = [cases[pid] for pid in patient_ids]

dataset = BraTSPatchDataset(
    selected_cases,
    patch_size=(32, 32, 32),
    patches_per_case=2,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

checkpoint = torch.load(
    ROOT / "results/comparison/fair_hybrid_40/checkpoints/hybrid_40_best.pth",
    map_location="cpu",
    weights_only=False,
)

model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy=checkpoint.get("strategy", "ring"),
)
model.load_state_dict(checkpoint["model_state_dict"])

device = select_device()
model.to(device).eval()

pred_counts = np.zeros(4, dtype=np.int64)
true_counts = np.zeros(4, dtype=np.int64)
total = 0

print("Device:", device)
print("Validation patients:", patient_ids)

with torch.inference_mode():
    for images, labels in loader:
                logits = model(images.to(device))

        print("Mean logits by class:", logits.mean(dim=(0, 2, 3, 4)).cpu().tolist())
        print("Mean logit std by class:", logits.std(dim=(0, 2, 3, 4)).cpu().tolist())

        predictions = logits.argmax(dim=1).cpu().numpy()
        truth = labels.numpy()

        pred_counts += np.bincount(predictions.ravel(), minlength=4)
        true_counts += np.bincount(truth.ravel(), minlength=4)
        total += truth.size

print("Total voxels:", total)
print("Ground truth counts:", true_counts.tolist())
print("Ground truth percentages:", np.round(true_counts / total * 100, 3).tolist())
print("Hybrid prediction counts:", pred_counts.tolist())
print("Hybrid prediction percentages:", np.round(pred_counts / total * 100, 3).tolist())
