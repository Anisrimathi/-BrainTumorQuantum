
import torch
from src.models.hybrid_quantum_unet import HybridQuantumUNet, select_device

device = select_device()
model = HybridQuantumUNet(
    in_channels=4,
    num_classes=4,
    base_channels=8,
    strategy="ring",
).to(device)
model.train()

images = torch.randn(2, 4, 32, 32, 32, device=device)
labels = torch.randint(0, 4, (2, 32, 32, 32), device=device)

logits = model(images)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()

print("Device:", device)
print("Loss:", loss.item())

for name, parameter in model.named_parameters():
    if "quantum" in name.lower() or "vqc" in name.lower():
        grad = parameter.grad
        print(
            name,
            "requires_grad =", parameter.requires_grad,
            "gradient is None =", grad is None,
            "gradient norm =", None if grad is None else grad.detach().norm().item(),
        )
