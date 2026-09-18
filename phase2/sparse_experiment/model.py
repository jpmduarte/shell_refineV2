"""Phase 2 network: a stack of submanifold sparse convs over the band's active set.
No downsampling (the band is already thin — see sparse.py's docstring for why a
dense-conv-style encoder/decoder isn't needed here).
"""

import torch
import torch.nn as nn

from sparse import SparseBatchNorm, SubMConv3d, build_neighbor_index


class SparseConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = SubMConv3d(in_ch, out_ch, kernel_size=3)
        self.norm = SparseBatchNorm(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, coords, features, neighbor_idx):
        x = self.conv(coords, features, neighbor_idx=neighbor_idx)
        x = self.norm(x)
        return self.act(x)


class SparseRefiner(nn.Module):
    """in_channels=2 (image, sdf_prior) -> per-voxel logit. Sigmoid applied outside
    (BCEWithLogitsLoss expects raw logits for numerical stability)."""

    def __init__(self, in_channels: int = 2, width: int = 32, depth: int = 4):
        super().__init__()
        channels = [in_channels] + [width] * depth
        self.blocks = nn.ModuleList([
            SparseConvBlock(cin, cout) for cin, cout in zip(channels[:-1], channels[1:])
        ])
        self.head = SubMConv3d(width, 1, kernel_size=1)

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        neighbor_idx = build_neighbor_index(coords, self.blocks[0].conv.offsets)
        x = features
        for block in self.blocks:
            x = block(coords, x, neighbor_idx)
        # 1x1 "conv" has no neighbors to gather — kernel_size=1 means its only
        # offset is (0,0,0), so build_neighbor_index isn't needed for the head.
        logit = self.head(coords, x)
        return logit.squeeze(-1)
