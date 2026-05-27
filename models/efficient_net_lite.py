import torch
import torch.nn as nn
from typing import Optional

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv).
    Core building block of EfficientNet.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 expand_ratio: int = 6,
                 stride: int = 1,
                 se_ratio: float = 0.25):
        super().__init__()
        mid_ch  = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ]
        # SE
        se_ch = max(1, int(in_ch * se_ratio))
        layers.append(SEBlock(mid_ch, reduction=max(1, mid_ch // se_ch)))
        # Project
        layers += [
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self.dropout(out) + x
        return out


class EfficientNetLite(nn.Module):
    """
    EfficientNet-Lite variant for audio deepfake detection.
    Lightweight (~380K params) version without SE blocks in early layers.

    Input:  (B, 1, freq_bins, time_frames)
    Output: (B, 2) — logits for [real, fake]
    """

    def __init__(self, in_channels: int = 1,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super(EfficientNetLite, self).__init__()

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            # MBConv blocks
            MBConvBlock(32,  16,  expand_ratio=1, stride=1),
            MBConvBlock(16,  24,  expand_ratio=6, stride=2),
            MBConvBlock(24,  24,  expand_ratio=6, stride=1),
            MBConvBlock(24,  40,  expand_ratio=6, stride=2),
            MBConvBlock(40,  40,  expand_ratio=6, stride=1),
            MBConvBlock(40,  80,  expand_ratio=6, stride=2),
            MBConvBlock(80,  112, expand_ratio=6, stride=1),

            # Head
            nn.Conv2d(112, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(320, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EfficientNetLite()
    x = torch.randn(4, 1, 80, 400)
    out = model(x)
    print(f"EfficientNetLite output: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")