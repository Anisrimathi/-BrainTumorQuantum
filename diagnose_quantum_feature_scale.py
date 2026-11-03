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
image, _ = dataset[0]
image = image.unsqueeze(0).to(device)

model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy=ckpt["strategy"],
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

captured = {}

def capture_bottleneck(module, inputs, output):
    captured["bottleneck"] = output.detach()

def capture_quantum(module, inputs, output):
    captured["quantum"] = output.detach()

h1 = model.bottleneck.register_forward_hook(capture_bottleneck)
h2 = model.quantum_expansion.register_forward_hook(capture_quantum)

with torch.no_grad():
    _ = model(image)

h1.remove()
h2.remove()

for name, tensor in captured.items():
    t = tensor.float()
    print(
        f"{name}: shape={tuple(t.shape)}, "
        f"mean_abs={t.abs().mean().item():.8g}, "
        f"std={t.std().item():.8g}, "
        f"max_abs={t.abs().max().item():.8g}"
    )

print("Device:", device)
print("Diagnostic complete; no weights changed.")
