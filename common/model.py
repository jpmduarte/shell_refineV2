"""A small 3D U-Net (1.40M parameters) that regresses the SDF, used by both passes of the method.

Input: the z-scored image, one channel. Output: tanh, the SDF normalised by its 10 mm truncation.

  96^3 x16  ─enc1───────────────────────────────── dec1─ 96^3 x16 ─ head ─ tanh
     48^3 x32  ─enc2───────────────────── dec2─┘
        24^3 x64  ─enc3─────────── dec3─┘
            12^3 x128  ─bottleneck─┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH = 3   # pooling levels; a fourth (5.6M parameters) gave nothing on plain x4
SPATIAL_DIMS = slice(2, None)   # (N, C, D, H, W) -> (D, H, W)


class ConvBlock(nn.Module):
    """Two size-preserving convolutions + batch norm + ReLU; nnU-Net's instance norm and leaky ReLU were worse here."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = DEPTH, kernel: int = 3):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2 ** level for level in range(depth)]

        # Created by name and in this order: it is what the checkpoints hold and what a seeded run initialises.
        for level in range(depth):
            in_ch = in_channels if level == 0 else channels[level - 1]
            setattr(self, f"enc{level + 1}", ConvBlock(in_ch, channels[level], kernel))
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, kernel)

        for level in reversed(range(depth)):
            setattr(self, f"up{level + 1}",
                    nn.ConvTranspose3d(channels[level] * 2, channels[level], kernel_size=2, stride=2))
            setattr(self, f"dec{level + 1}", ConvBlock(channels[level] * 2, channels[level], kernel))

        self.head = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        return torch.tanh(self.head(self.features(x)))

    def features(self, x):
        """The last decoder level's features."""
        skips = []
        for level in range(self.depth):
            x = getattr(self, f"enc{level + 1}")(x if level == 0 else self.pool(x))
            skips.append(x)

        d = self.bottleneck(self.pool(x))

        for level in reversed(range(self.depth)):
            skip = skips[level]
            up = self._match(getattr(self, f"up{level + 1}")(d), skip)
            d = getattr(self, f"dec{level + 1}")(torch.cat([up, skip], dim=1))

        return d

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize x to ref's spatial shape; a safety net that never fires when every axis divides by 8."""
        if x.shape[SPATIAL_DIMS] != ref.shape[SPATIAL_DIMS]:
            x = F.interpolate(x, size=ref.shape[SPATIAL_DIMS], mode="trilinear", align_corners=False)
        return x
