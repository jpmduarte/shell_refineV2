# The refiner's network: a 3D U-Net like ../../model.py, adapted to refine.
#
#   - 2 input channels: the image and the first pass's prior (its SDF), so the network
#     sees where the first pass put the surface and only has to correct it.
#   - sigmoid output: the probability that each voxel is inside the head (trained with
#     Dice + BCE in train.py), instead of an SDF.
#
# Size is configurable, for the architecture arms:
#   depth          pooling levels (3: 16-32-64, bottleneck 128; 4 adds a 128 level, 256 bottleneck)
#   kernel         convolution kernel size (3, or 5 for a wider receptive field per layer)
#   base_channels  channels at the top level, doubling per level (16, or 32: 32-64-128)
# The default (3, 3, 16) is the measured refiner: 1.40M parameters.

import torch
import torch.nn as nn
import torch.nn.functional as F


# Two kxkxk convolutions, each followed by batch norm and ReLU; padding keeps the size.
class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RefinerUNet(nn.Module):

    def __init__(self, in_channels: int = 2, base_channels: int = 16, depth: int = 3,
                 kernel: int = 3):
        super().__init__()
        self.depth = depth
        ch = [base_channels * 2 ** i for i in range(depth)]   # e.g. 16, 32, 64

        # Encoder: one block per level, halving the resolution between levels.
        for i in range(depth):
            setattr(self, f"enc{i + 1}",
                    ConvBlock(in_channels if i == 0 else ch[i - 1], ch[i], kernel))
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(ch[-1], ch[-1] * 2, kernel)

        # Decoder: upsample, join the skip connection, one block per level.
        for i in reversed(range(depth)):
            setattr(self, f"up{i + 1}", nn.ConvTranspose3d(ch[i] * 2, ch[i], kernel_size=2,
                                                           stride=2))
            setattr(self, f"dec{i + 1}", ConvBlock(ch[i] * 2, ch[i], kernel))

        self.head = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = getattr(self, f"enc{i + 1}")(x if i == 0 else self.pool(x))
            skips.append(x)

        d = self.bottleneck(self.pool(x))

        for i in reversed(range(self.depth)):
            skip = skips[i]
            up = getattr(self, f"up{i + 1}")(d)
            if up.shape[2:] != skip.shape[2:]:
                up = F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=False)
            d = getattr(self, f"dec{i + 1}")(torch.cat([up, skip], dim=1))

        return torch.sigmoid(self.head(d))
