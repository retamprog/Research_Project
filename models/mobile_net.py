# models/mobilenet_cnn.py — MobileNet-style depthwise separable CNN

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (MobileNet building block)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MobileNetCNN(nn.Module):
    """
    MobileNet-inspired lightweight CNN for deepfake detection.
    Uses depthwise separable convolutions for efficiency.
    ~280K parameters.

    Input:  (B, 1, freq_bins, time_frames)
    Output: (B, 2) — logits for [real, fake]
    """

    def __init__(self, in_channels: int = 1,
                 num_classes: int = 2,
                 width_mult: float = 0.5,
                 dropout: float = 0.3):
        super(MobileNetCNN, self).__init__()

        def c(n): return max(1, int(n * width_mult))

        self.features = nn.Sequential(
            # Stem (standard conv)
            nn.Conv2d(in_channels, c(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU6(inplace=True),

            # DW-Sep blocks
            DepthwiseSeparableConv(c(32),  c(64),  stride=1),
            DepthwiseSeparableConv(c(64),  c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128), stride=1),
            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256), stride=1),
            DepthwiseSeparableConv(c(256), c(512), stride=2),

            # Final pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c(512), num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MobileNetCNN()
    x = torch.randn(4, 1, 80, 400)
    out = model(x)
    print(f"MobileNetCNN output: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")