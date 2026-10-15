# The network: a small 3D U-Net that regresses the signed distance field.
#
# Input: one channel, the z-scored image on a 96^3 grid. Output: one channel in [-1, 1]
# (tanh), the SDF normalised by its 10 mm truncation; the head is where it is negative.
# Three pooling levels, 16 channels at the top: 1.40M parameters. Both passes of the
# method (the x4 volume estimate and the size-normalised segmentation) use it.
#
#   96^3 x16  ─enc1───────────────────────────────── dec1─ 96^3 x16 ─ head ─ tanh
#      48^3 x32  ─enc2───────────────────── dec2─┘
#         24^3 x64  ─enc3─────────── dec3─┘
#             12^3 x128  ─bottleneck─┘

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH = 3   # pooling levels; a fourth (5.6M parameters) gave nothing on plain x4


# Two 3x3x3 convolutions, each followed by batch norm and ReLU. The nnU-Net's choices,
# instance norm and leaky ReLU, were tried here and were worse.
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

    # contour_head adds a second output for training only: a Gaussian contour map
    # (exp(-d^2 / 2 sigma^2), 1 on the surface, 0 far from it) predicted from the same last
    # features (Torres et al., EMBC 2023). Created after every other layer, so with it off
    # the network, its checkpoints and a seeded initialisation are exactly as before.
    # depth = pooling levels (3: 16-32-64, bottleneck 128; 4 adds a 128 level, bottleneck
    # 256, and needs inputs divisible by 16).
    def __init__(self, in_channels: int = 1, base_channels: int = 16, contour_head: bool = False,
                 depth: int = DEPTH):
        super().__init__()
        self.depth = depth
        # Channels per level, doubling at each pooling: 16, 32, 64.
        ch = [base_channels * 2 ** i for i in range(depth)]

        # The layers are created by name (enc1, enc2, ..., up3, dec3, ...) and in this
        # order, because that is what the trained checkpoints contain and what a seeded
        # run initialises from.

        # Encoder: one block per level, halving the resolution between levels.
        for i in range(depth):
            setattr(self, f"enc{i + 1}", ConvBlock(in_channels if i == 0 else ch[i - 1], ch[i]))
        self.pool = nn.MaxPool3d(2)

        # Bottleneck, at 1/8 of the input resolution.
        self.bottleneck = ConvBlock(ch[-1], ch[-1] * 2)

        # Decoder: upsample x2, join with the encoder output of the same level (skip
        # connection), one block.
        for i in reversed(range(depth)):
            setattr(self, f"up{i + 1}", nn.ConvTranspose3d(ch[i] * 2, ch[i], kernel_size=2,
                                                           stride=2))
            setattr(self, f"dec{i + 1}", ConvBlock(ch[i] * 2, ch[i]))

        # 1x1x1 convolution down to the single SDF channel.
        self.head = nn.Conv3d(base_channels, 1, kernel_size=1)

        # Optional contour output (sigmoid), from the same features.
        self.contour_head = nn.Conv3d(base_channels, 1, kernel_size=1) if contour_head else None

    # The SDF only: what prediction uses, with or without a contour head.
    def forward(self, x):
        return torch.tanh(self.head(self.features(x)))

    # The head's raw output, without tanh: logits for a network trained on a mask
    # (train.py --loss mask), where sigmoid gives the probability of "inside".
    def logits(self, x):
        return self.head(self.features(x))

    # Training with a contour head: (SDF, contour map).
    def forward_with_contour(self, x):
        d = self.features(x)
        return torch.tanh(self.head(d)), torch.sigmoid(self.contour_head(d))

    # The last decoder level's features, shared by both heads.
    def features(self, x):
        # Encoder, keeping each level's output for its skip connection.
        skips = []
        for i in range(self.depth):
            x = getattr(self, f"enc{i + 1}")(x if i == 0 else self.pool(x))
            skips.append(x)

        d = self.bottleneck(self.pool(x))

        # Decoder, from the deepest level up.
        for i in reversed(range(self.depth)):
            skip = skips[i]
            up = self._match(getattr(self, f"up{i + 1}")(d), skip)
            d = getattr(self, f"dec{i + 1}")(torch.cat([up, skip], dim=1))

        return d

    # Resize x to ref's spatial shape if they differ. Never triggers when every axis is
    # divisible by 8; kept as a safety net.
    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x
