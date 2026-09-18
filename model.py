"""Small 3D U-Net. Phase 1 regresses a signed field in [-1, 1] (tanh); phase 2 (once
built) outputs a probability (sigmoid).
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


class UNet3D(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 16,
                 out_activation: str = "tanh"):
        super().__init__()
        c = base_channels
        self.out_activation = out_activation

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c,     c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(c * 4, c * 8)

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

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))

        out = self.head(d1)

        if self.out_activation == "tanh":
            return torch.tanh(out)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(out)
        return out

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x
