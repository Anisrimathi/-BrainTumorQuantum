
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.attention_unet import ConvBlock
from quantum.quantum_vqc import QuantumVQC


class HybridQuantumUNet(nn.Module):
    """
    Hybrid 3D quantum-classical U-Net.

    Classical layers run on the selected PyTorch device.
    PennyLane default.qubit simulates the quantum circuit on CPU.
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy="ring",
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        cb = base_channels * 16

        self.pool = nn.MaxPool3d(kernel_size=2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, cb)

        # Classical-to-quantum feature projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.quantum_projection = nn.Linear(cb, 8)

        # IMPORTANT: This matches your actual QuantumVQC constructor.
        self.quantum_layer = QuantumVQC(
            strategy=strategy,
            n_layers=2,
        )

        # Quantum-to-classical feature projection and fusion
        self.quantum_expansion = nn.Linear(8, cb)

        self.fusion = nn.Sequential(
            nn.Conv3d(cb * 2, cb, kernel_size=1),
            nn.BatchNorm3d(cb),
            nn.ReLU(inplace=True),
        )

        # Decoder stage 4
        self.up4 = nn.ConvTranspose3d(
            cb, c4, kernel_size=2, stride=2
        )
        self.gate4 = nn.Sequential(
            nn.Conv3d(c4 * 2, c4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dec4 = ConvBlock(c4 * 2, c4)

        # Decoder stage 3
        self.up3 = nn.ConvTranspose3d(
            c4, c3, kernel_size=2, stride=2
        )
        self.gate3 = nn.Sequential(
            nn.Conv3d(c3 * 2, c3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dec3 = ConvBlock(c3 * 2, c3)

        # Decoder stage 2
        self.up2 = nn.ConvTranspose3d(
            c3, c2, kernel_size=2, stride=2
        )
        self.gate2 = nn.Sequential(
            nn.Conv3d(c2 * 2, c2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dec2 = ConvBlock(c2 * 2, c2)

        # Decoder stage 1
        self.up1 = nn.ConvTranspose3d(
            c2, c1, kernel_size=2, stride=2
        )
        self.gate1 = nn.Sequential(
            nn.Conv3d(c1 * 2, c1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.output = nn.Conv3d(
            c1, num_classes, kernel_size=1
        )

    @staticmethod
    def match_size(tensor, reference):
        """Match decoder spatial dimensions to the skip connection."""
        if tensor.shape[2:] != reference.shape[2:]:
            tensor = F.interpolate(
                tensor,
                size=reference.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return tensor

    def decode_stage(
        self, decoder, skip, upsample, gate, block
    ):
        decoder = upsample(decoder)
        decoder = self.match_size(decoder, skip)

        gate_weights = gate(
            torch.cat([decoder, skip], dim=1)
        )
        attended_skip = skip * gate_weights

        return block(
            torch.cat([decoder, attended_skip], dim=1)
        )

    def forward(self, x):
        # Input shape: [B, 4, D, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        # Generate eight features for the quantum circuit.
        classical_features = self.global_pool(
            bottleneck
        ).flatten(1)

        quantum_inputs = torch.tanh(
            self.quantum_projection(classical_features)
        )

        # Quantum simulator executes on CPU.
        quantum_outputs = self.quantum_layer(
            quantum_inputs.to("cpu")
        )

        # Return quantum features to the classical device.
        quantum_outputs = quantum_outputs.to(
            device=bottleneck.device,
            dtype=bottleneck.dtype,
        )

        quantum_features = self.quantum_expansion(
            quantum_outputs
        )

        quantum_features = quantum_features[:, :, None, None, None]
        quantum_features = quantum_features.expand_as(bottleneck)

        fused = self.fusion(
            torch.cat(
                [bottleneck, quantum_features],
                dim=1,
            )
        )

        d4 = self.decode_stage(
            fused, e4, self.up4, self.gate4, self.dec4
        )
        d3 = self.decode_stage(
            d4, e3, self.up3, self.gate3, self.dec3
        )
        d2 = self.decode_stage(
            d3, e2, self.up2, self.gate2, self.dec2
        )
        d1 = self.decode_stage(
            d2, e1, self.up1, self.gate1, self.dec1
        )

        # Output logits: [B, 4 classes, D, H, W]
        return self.output(d1)


def select_device():
    """Prefer Intel XPU, then CUDA, then CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def run_smoke_test():
    """Check forward pass, loss, gradients, and optimizer update."""
    torch.manual_seed(42)

    device = select_device()

    print("Classical model device:", device)
    print("Quantum simulator: PennyLane default.qubit (CPU)")

    model = HybridQuantumUNet(
        in_channels=4,
        num_classes=4,
        base_channels=8,
        strategy="ring",
    ).to(device)

    model.train()

    # A 32x32x32 input keeps enough spatial elements at the bottleneck
    # for BatchNorm with batch size 1.
    inputs = torch.randn(
        1, 4, 32, 32, 32,
        device=device,
    )

    targets = torch.randint(
        0, 4, (1, 32, 32, 32),
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = criterion(logits, targets)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is not finite.")

    loss.backward()

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    gradients_available = sum(
        p.grad is not None
        for p in trainable_parameters
    )

    gradients_finite = all(
        p.grad is None
        or torch.isfinite(p.grad).all().item()
        for p in trainable_parameters
    )

    optimizer.step()

    print("Input shape:", tuple(inputs.shape))
    print("Output shape:", tuple(logits.shape))
    print("Loss:", round(loss.item(), 6))
    print(
        "Parameters receiving gradients:",
        f"{gradients_available}/{len(trainable_parameters)}",
    )
    print("Gradients finite:", gradients_finite)
    print("Elapsed seconds:", round(time.time() - start, 2))

    expected_shape = (1, 4, 32, 32, 32)

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(logits.shape)}"
        )

    if not gradients_finite:
        raise RuntimeError("Non-finite gradients detected.")

    if gradients_available == 0:
        raise RuntimeError(
            "No trainable parameters received gradients."
        )

    print("Forward and backward pass completed.")
    print("Hybrid smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
