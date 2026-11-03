import torch
from preprocessing.patch_dataset import discover_cases, BraTSPatchDataset
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device

ROOT = r"C:\BrainTumorQuantum"
device = select_device()
ckpt = torch.load(
    ROOT + r"\results\checkpoints\hybrid_quantum_unet_ring_extended.pth",
    map_location="cpu",
    weights_only=False,
)

cases = {c["id"]: c for c in discover_cases()}
train_ids = ckpt["configuration"]["train_patient_ids"]
case = cases[train_ids[0]]

dataset = BraTSPatchDataset(
    [case],
    patches_per_case=1,
    patch_size=(32, 32, 32),
    seed=42,
)
image, label = dataset[0]
image = image.unsqueeze(0).to(device)
label = label.unsqueeze(0).to(device).long()

model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy=ckpt["strategy"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.train()
model.zero_grad(set_to_none=True)

logits = model(image)
ce = torch.nn.functional.cross_entropy(logits, label)

probs = torch.softmax(logits, dim=1)
one_hot = torch.nn.functional.one_hot(label, 4).permute(0, 4, 1, 2, 3).float()
dims = (0, 2, 3, 4)
intersection = (probs * one_hot).sum(dims)
denominator = probs.sum(dims) + one_hot.sum(dims)
dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
dice_loss = 1 - dice[1:].mean()
loss = ce + dice_loss
loss.backward()

print("Device:", device)
print("Training patient:", train_ids[0])
print("Patch shape:", tuple(image.shape))
print("Label counts:", torch.bincount(label.flatten(), minlength=4).cpu().tolist())
print("CE loss:", round(ce.item(), 6))
print("Dice loss:", round(dice_loss.item(), 6))
print("Total loss:", round(loss.item(), 6))

for name, params in [
    ("Quantum circuit", list(model.quantum_layer.parameters())),
    ("Quantum input projection", list(model.quantum_projection.parameters())),
    ("Bottleneck", list(model.bottleneck.parameters())),
    ("Decoder", list(model.dec1.parameters())),
]:
    grads = [p.grad for p in params if p.grad is not None]
    grad_sum = sum(g.abs().sum().item() for g in grads)
    grad_max = max((g.abs().max().item() for g in grads), default=0.0)
    print(f"{name}: grad_sum={grad_sum:.8g}, grad_max={grad_max:.8g}")

print("Diagnostic only; no optimizer step or checkpoint save.")
