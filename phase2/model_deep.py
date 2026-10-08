"""Four-level 3D U-Net for phase 2, otherwise identical to the shared three-level one.

The shared model pools three times. This pools four, which doubles the receptive field
at the bottleneck: on a 128x128x48 slab the deepest features see 16 voxels of stride
rather than 8, so a voxel's decision draws on roughly 6mm more of the boundary either
side at 0.4mm spacing.

Depth rather than width, because width has already been measured and was null. Widening
the polar refiner from 177k to 1.58M parameters moved validation MAE by 0.02mm and
training loss by 5%; keep_r=6, which has *fewer* parameters, did the same. Reach is the
axis that has shown signal: enlarging the polar patch from a sixteenth of the sphere to
a quarter was worth +0.0006 Dice, and the slab thickness sweep found its optimum at the
thickness that covered the measured displacement tail.

A fourth level needs the input to divide by 16. A 128x128x48 slab does (48/16 = 3) and
a 64^3 cube does (64/16 = 4). A fifth level would not, which is why this stops at four.

Separate from the shared model.py deliberately. That one is loaded by five phase 1
checkpoints and every phase 2 arm measured so far, and changing its structure would
either break those or quietly alter what "the same architecture" means across a
comparison that has been running for weeks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeepUNet3D(nn.Module):

    def __init__(self, in_channels: int = 2, base_channels: int = 16,
                 out_activation: str = "sigmoid"):
        super().__init__()
        c = base_channels
        self.out_activation = out_activation

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c,     c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4  = nn.ConvTranspose3d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3  = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2  = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1  = nn.ConvTranspose3d(c * 2, c,     kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._match(self.up4(b),  e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._match(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))

        out = self.head(d1)

        if self.out_activation == "tanh":
            return torch.tanh(out)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(out)
        if self.out_activation == "half_tanh":
            return 0.5 * torch.tanh(out)
        return out

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x
