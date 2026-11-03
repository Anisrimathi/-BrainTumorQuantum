import torch
from models.hybrid_quantum_unet import HybridQuantumUNet, select_device

torch.manual_seed(42)
device = select_device()
checkpoint_path = "results/checkpoints/hybrid_quantum_unet_ring_extended.pth"

checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)

model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy=checkpoint["strategy"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.train()

quantum_params = list(model.quantum_layer.parameters())
projection_params = list(model.quantum_projection.parameters())

before = [p.detach().clone() for p in quantum_params]

x = torch.randn(1, 4, 32, 32, 32, device=device)
target = torch.randint(0, 4, (1, 32, 32, 32), device=device)

model.zero_grad(set_to_none=True)
logits = model(x)
loss = torch.nn.functional.cross_entropy(logits, target)
loss.backward()

print("Device:", device)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Loss:", round(loss.item(), 6))

for name, params in [
    ("Quantum circuit", quantum_params),
    ("Quantum input projection", projection_params),
]:
    grads = [p.grad for p in params]
    present = [g for g in grads if g is not None]
    finite = all(torch.isfinite(g).all().item() for g in present)
    total_grad = sum(g.abs().sum().item() for g in present)

    print(f"\n{name}:")
    print(f"  Parameters: {len(params)}")
    print(f"  Parameters with gradients: {len(present)}/{len(params)}")
    print(f"  Gradients finite: {finite}")
    print(f"  Sum absolute gradient: {total_grad:.8g}")

print("\nQuantum parameter values loaded from checkpoint:")
for i, p in enumerate(quantum_params):
    print(f"  Tensor {i}: shape={tuple(p.shape)}, mean={p.detach().mean().item():.6f}")

print("\nDiagnostic complete; no optimizer step was performed.")
