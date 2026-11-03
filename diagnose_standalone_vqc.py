import torch
import pennylane as qml
from quantum.quantum_vqc import QuantumVQC

torch.manual_seed(123)
torch.set_default_dtype(torch.float32)

model = QuantumVQC(strategy="ring", n_layers=2).cpu()
model.train()

x = torch.tensor(
    [[-0.7, -0.4, -0.1, 0.2, 0.4, 0.6, 0.8, 1.0]],
    dtype=torch.float32,
    requires_grad=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
before = [p.detach().clone() for p in model.parameters()]

optimizer.zero_grad(set_to_none=True)
y = model(x)

# A deliberately non-constant weighted objective.
weights = torch.tensor(
    [[1., 2., 3., 4., 5., 6., 7., 8.]],
    dtype=torch.float32,
)
loss = (y * weights).sum()
loss.backward()

print("PennyLane:", qml.__version__)
print("PyTorch:", torch.__version__)
print("Output:", y.detach().tolist())
print("Loss:", loss.item())
print("Input gradient abs sum:", x.grad.abs().sum().item())
print("Input gradient abs max:", x.grad.abs().max().item())

for name, p in model.named_parameters():
    print(f"\nParameter: {name}")
    print("Gradient is None:", p.grad is None)
    if p.grad is not None:
        print("Gradient abs sum:", p.grad.abs().sum().item())
        print("Gradient abs max:", p.grad.abs().max().item())
        print("Gradient finite:", torch.isfinite(p.grad).all().item())

optimizer.step()

for (name, p), old in zip(model.named_parameters(), before):
    delta = (p.detach() - old).abs()
    print(f"\n{name} update abs sum:", delta.sum().item())
    print(f"{name} update abs max:", delta.max().item())

print("\nDirect circuit diagnostic complete.")
