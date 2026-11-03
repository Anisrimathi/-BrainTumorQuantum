
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for filtering encoder features."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels),
        )

        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels),
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        attention_map = self.attention(
            self.gate_conv(gate) + self.skip_conv(skip)
        )
        return skip * attention_map


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for four-channel brain MRI segmentation.

    Input:  [B, 4, D, H, W]
    Output: [B, num_classes, D, H, W] logits
    """

    def __init__(self, in_channels=4, num_classes=4, base_channels=8):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)

        self.bottleneck = ConvBlock(c4, c4 * 2)

        self.up4 = nn.ConvTranspose3d(c4 * 2, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4, c4, c4 // 2)
        self.dec4 = ConvBlock(c4 * 2, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3, c3, c3 // 2)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, c2 // 2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1, c1, c1 // 2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.output = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        s4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.output(d1)


if __name__ == "__main__":
    model = AttentionUNet3D(
        in_channels=4,
        num_classes=4,
        base_channels=8,
    )

    test_input = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        output = model(test_input)

    print("Input shape:", tuple(test_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
