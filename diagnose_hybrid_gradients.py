
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\BrainTumorQuantum")
sys.path.insert(0, str(ROOT / "src"))

from models.hybrid_quantum_unet import HybridQuantumUNet, select_device
from preprocessing.patch_dataset import discover_cases, BraTSPatchDataset

CHECKPOINT = ROOT / "results/checkpoints/hybrid_quantum_unet_ring_extended.pth"

device = select_device()
print("Device:", device)

# Load the trained model without changing the checkpoint.
model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy="ring",
).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

if isinstance(checkpoint, dict):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
else:
    raise TypeError("Unexpected checkpoint format.")

model.load_state_dict(state)
model.train()

# Use one real BraTS training patch.
cases = list(discover_cases())
dataset = BraTSPatchDataset(
    cases[:1],
    patches_per_case=1,
    patch_size=(32, 32, 32),
    seed=123,
)

image, label = dataset[0]
image = image.unsqueeze(0).to(device)
label = label.unsqueeze(0).long().to(device)

# Capture intermediate activations and their gradients.
captured = {}

def save_activation(name):
    def hook(module, inputs, output):
        captured[name] = output
        if output.requires_grad:
            output.retain_grad()
    return hook

handles = [
    model.quantum_projection.register_forward_hook(save_activation("projection")),
    model.quantum_layer.register_forward_hook(save_activation("quantum")),
    model.quantum_expansion.register_forward_hook(save_activation("expansion")),
]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad(set_to_none=True)

logits = model(image)
loss = torch.nn.functional.cross_entropy(logits, label)
loss.backward()

print("\nLoss:", loss.item())

for name, tensor in captured.items():
    print(f"\n{name}:")
    print("  mean abs:", tensor.detach().abs().mean().item())
    print("  max abs:", tensor.detach().abs().max().item())
    print("  grad exists:", tensor.grad is not None)
    if tensor.grad is not None:
        print("  grad mean abs:", tensor.grad.abs().mean().item())
        print("  grad max abs:", tensor.grad.abs().max().item())

for name, parameter in model.named_parameters():
    if any(key in name for key in (
        "quantum_projection",
        "quantum_layer",
        "quantum_expansion",
    )):
        grad = parameter.grad
        print(f"\nParameter: {name}")
        print("  gradient missing:", grad is None)
        if grad is not None:
            print("  gradient mean abs:", grad.abs().mean().item())
            print("  gradient max abs:", grad.abs().max().item())

for handle in handles:
    handle.remove()

print("\nDiagnostic completed. No checkpoint was saved.")
