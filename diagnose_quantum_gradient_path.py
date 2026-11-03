import torch
from preprocessing.patch_dataset import discover_cases, BraTSPatchDataset
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device

root = r"C:\BrainTumorQuantum"
device = select_device()
ckpt = torch.load(
    root + r"\results\checkpoints\hybrid_quantum_unet_ring_extended.pth",
    map_location="cpu",
    weights_only=False,
)

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
model.zero_grad(set_to_none=True)

captured = {}

def quantum_hook(module, inputs, output):
    for i, value in enumerate(inputs):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            value.retain_grad()
            captured[f"input_{i}"] = value
    if isinstance(output, torch.Tensor):
        output.retain_grad()
        captured["output"] = output

handle = model.quantum_layer.register_forward_hook(quantum_hook)

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
handle.remove()

print("Device:", device)
print("Loss:", loss.item())

for name, tensor in captured.items():
    grad = tensor.grad
    if grad is None:
        print(f"{name}: gradient is None")
    else:
        print(
            f"{name}: shape={tuple(tensor.shape)}, "
            f"grad_abs_sum={grad.abs().sum().item():.8g}, "
            f"grad_abs_max={grad.abs().max().item():.8g}, "
            f"finite={torch.isfinite(grad).all().item()}"
        )

for name, param in model.quantum_layer.named_parameters():
    grad = param.grad
    print(f"Parameter: {name}")
    print("  Device:", param.device)
    print("  Dtype:", param.dtype)
    print("  Gradient is None:", grad is None)
    if grad is not None:
        print("  Gradient sum:", grad.abs().sum().item())
        print("  Gradient max:", grad.abs().max().item())

print("Diagnostic only; no optimizer step or checkpoint save.")
