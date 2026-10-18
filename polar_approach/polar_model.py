# The baseline U-Net (../model.py, unchanged) plus a polar branch: from the head's centre,
# the head's radius in each of polar.N_DIRS directions.
#
# Why not the first version (a 1x1x1 convolution on the last decoder features, at every
# voxel; v1/results.md): there the 64 radii were a linear function of the 16 features the
# SDF also needs, read at full resolution with little context, and asked of 128 arbitrary
# interior points. Here the radii are asked only from the centre, and read from four levels
# of the network, the bottleneck's 12^3 view of the whole head included, through a small
# MLP:
#
#   features at the centre (grid_sample):  dec1 16 + dec2 32 + dec3 64 + bottleneck 128
#   + the bottleneck averaged over the volume:                                     128
#   -> Linear 368->256 -> ReLU -> Linear 256->64 = log radius (voxels), offset by log(mean)
#
# The U-Net is built first and the branch after it, so with the same seed the U-Net starts
# from the same weights as the baseline; its state_dict is saved on its own
# (phase1_last.pth), which infer.py and predict.py read unchanged.

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model import UNet3D  # noqa: E402

import polar  # noqa: E402


# Sample a feature map (1, C, d, h, w) covering the cube at voxel points (K, 3) of the full
# grid (cube^3): every level spans the same physical extent, so one mapping fits all.
def sample_at(fmap, pts_vox, cube):
    norm = (2 * pts_vox + 1) / cube - 1                      # (K, 3) in [-1, 1], z y x
    grid = norm.flip(-1).view(1, -1, 1, 1, 3).to(fmap.dtype)  # grid_sample wants x y z
    return F.grid_sample(fmap, grid, align_corners=False).view(fmap.shape[1], -1).T  # (K, C)


class PolarNet(nn.Module):

    def __init__(self, base_channels: int = 16, n_dirs: int = polar.N_DIRS, hidden: int = 256,
                 mean_radius_vox: float = polar.MEAN_RADIUS_VOX):
        super().__init__()
        self.net = UNet3D(base_channels=base_channels)
        ch = [base_channels * 2 ** i for i in range(self.net.depth)]
        n_in = sum(ch) + 2 * ch[-1] * 2       # dec1..dec3, the bottleneck at c, its mean
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, n_dirs))
        # Start every radius at the mean head radius: the last layer's bias, zero weights
        # would freeze it, so small ones.
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.constant_(self.mlp[-1].bias, math.log(mean_radius_vox))

    # The U-Net, keeping every decoder level and the bottleneck (as UNet3D.features does).
    def _levels(self, x):
        net, skips = self.net, []
        for i in range(net.depth):
            x = getattr(net, f"enc{i + 1}")(x if i == 0 else net.pool(x))
            skips.append(x)
        d = net.bottleneck(net.pool(x))
        levels = {"bottleneck": d}
        for i in reversed(range(net.depth)):
            up = net._match(getattr(net, f"up{i + 1}")(d), skips[i])
            d = getattr(net, f"dec{i + 1}")(torch.cat([up, skips[i]], dim=1))
            levels[f"dec{i + 1}"] = d
        return torch.tanh(net.head(d)), levels

    # x (B, 1, cube^3), centres (B, 3) voxel coordinates. Returns (SDF (B, 1, cube^3),
    # log radii (B, n_dirs), in voxels).
    def forward(self, x, centres):
        sdf, lv = self._levels(x)
        cube = x.shape[-1]
        out = []
        for b in range(x.shape[0]):
            c = centres[b:b + 1].float()
            feats = [sample_at(lv[k][b:b + 1], c, cube)
                     for k in ("dec1", "dec2", "dec3", "bottleneck")]
            feats.append(lv["bottleneck"][b].mean(dim=(1, 2, 3))[None])
            out.append(self.mlp(torch.cat(feats, dim=1)))
        return sdf, torch.cat(out, dim=0)
