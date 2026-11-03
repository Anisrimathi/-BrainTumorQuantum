import torch
from preprocessing.patch_dataset import discover_cases, BraTSPatchDataset
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device

root = r"C:\BrainTumorQuantum"
device = select_device()
path = root + r"\results\checkpoints\hybrid_quantum_unet_ring_extended.pth"

ckpt = torch.load(path, map_location="cpu", weights_only=False)
cases = {c["id"]: c for c in discover_cases()}
patient_id = ckpt["configuration"]["train_patient_ids"][0]

dataset = BraTSPatchDataset(
    [cases[patient_id]],
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
qparams = list(model.quantum_layer.parameters())
before = [p.detach().clone() for p in qparams]

optimizer.zero_grad(set_to_none=True)
logits = model(image)
ce = torch.nn.functional.cross_entropy(logits, label)

probs = torch.softmax(logits, dim=1)
one_hot = torch.nn.functional.one_hot(label, 4).permute(0, 4, 1, 2, 3).float()
dims = (0, 2, 3, 4)
intersection = (probs * one_hot).sum(dims)
denominator = probs.sum(dims) + one_hot.sum(dims)
dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
loss = ce + (1 - dice[1:].mean())

loss.backward()

print("Device:", device)
print("Loss before update:", loss.item())

for i, p in enumerate(qparams):
    if p.grad is None:
        print(f"Quantum parameter {i}: gradient is None")
    else:
        print(
            f"Quantum parameter {i}: "
            f"gradient_abs_sum={p.grad.abs().sum().item():.8g}, "
            f"gradient_abs_max={p.grad.abs().max().item():.8g}"
        )

optimizer.step()

for i, (old, p) in enumerate(zip(before, qparams)):
    delta = (p.detach() - old).abs()
    print(
        f"Quantum parameter {i}: "
        f"update_abs_sum={delta.sum().item():.8g}, "
        f"update_abs_max={delta.max().item():.8g}"
    )

print("One-step diagnostic finished. No checkpoint was saved.")
