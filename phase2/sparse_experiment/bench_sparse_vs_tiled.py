"""Benchmark: dense overlapping-tile conv (v1's phase 2 approach) vs. sparse conv on
the real band, same small network depth/channels, timed on this GPU.

    python phase2/bench_sparse_vs_tiled.py
"""

import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import case_ids  # noqa: E402
from make_sdf import load as load_sdf  # noqa: E402
from phase2.sparse import SubMConv3d, build_neighbor_index  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = [2, 16, 32, 16, 1]  # matches v1 phase 2's rough depth/width
PATCH = 64
STRIDE = 32
BAND_MM = 5.0
TRUNC_MM = 10.0


# --------------------------------------------------------------------------- dense

class DenseStack(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.ReLU(inplace=True)]
        layers = layers[:-1]  # no relu after the last layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def axis_origins(dim, patch, stride):
    if dim <= patch:
        return [0]
    last = dim - patch
    stops = list(range(0, last + 1, stride))
    if stops[-1] != last:
        stops.append(last)
    return stops


def tile_origins(shape, patch, stride):
    return list(itertools.product(*(axis_origins(d, patch, stride) for d in shape)))


@torch.no_grad()
def bench_dense_tiled(bbox_shape, channels, patch, stride, tile_batch=4):
    model = DenseStack(channels).to(DEVICE).eval()
    x = torch.randn(1, channels[0], *bbox_shape, device=DEVICE)

    origins = tile_origins(bbox_shape, patch, stride)
    n_tiles = len(origins)
    total_tile_voxels = n_tiles * patch ** 3

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(0, n_tiles, tile_batch):
        batch_origins = origins[i:i + tile_batch]
        tiles = torch.stack([
            x[0, :, oz:oz + patch, oy:oy + patch, ox:ox + patch]
            for oz, oy, ox in batch_origins
        ], dim=0)
        _ = model(tiles)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, n_tiles, total_tile_voxels


# -------------------------------------------------------------------------- sparse

class SparseStack(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            SubMConv3d(cin, cout, kernel_size=3)
            for cin, cout in zip(channels[:-1], channels[1:])
        ])

    def forward(self, coords, features):
        neighbor_idx = build_neighbor_index(coords, self.layers[0].offsets)
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(coords, x, neighbor_idx=neighbor_idx)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


@torch.no_grad()
def bench_sparse(coords, channels):
    model = SparseStack(channels).to(DEVICE).eval()
    features = torch.randn(coords.shape[0], channels[0], device=DEVICE)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(coords, features)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed


# --------------------------------------------------------------------------- main

def main():
    cid = case_ids("cache")[0]
    sdf, affine = load_sdf("cache", cid, 1)
    sdf = sdf[0] if sdf.ndim == 4 else sdf
    band = np.abs(sdf) < (BAND_MM / TRUNC_MM)

    coords_np = np.argwhere(band)
    mins, maxs = coords_np.min(0), coords_np.max(0)
    bbox_shape = tuple((maxs - mins + 1).tolist())

    print(f"Case: {cid}")
    print(f"Native shape: {sdf.shape}   Band voxels: {band.sum():,} ({100*band.mean():.2f}%)")
    print(f"Band bbox shape: {bbox_shape}   bbox voxels: {np.prod(bbox_shape):,}")
    print(f"Device: {DEVICE}\n")

    # sparse: coords relative to bbox origin, batch column = 0
    rel_coords = coords_np - mins
    coords = torch.from_numpy(rel_coords).long()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.long), coords], dim=1).to(DEVICE)

    # warmup (first CUDA call pays context/cudnn-autotune cost, exclude from timing)
    _ = bench_dense_tiled(bbox_shape, CHANNELS, PATCH, STRIDE, tile_batch=2)
    _ = bench_sparse(coords, CHANNELS)

    dense_t, n_tiles, tile_voxels = bench_dense_tiled(bbox_shape, CHANNELS, PATCH, STRIDE)
    sparse_t = bench_sparse(coords, CHANNELS)

    print(f"Dense tiled  : {dense_t*1000:8.2f} ms   {n_tiles} tiles   "
         f"{tile_voxels:,} voxel-passes ({tile_voxels/np.prod(bbox_shape):.1f}x the bbox)")
    print(f"Sparse (band): {sparse_t*1000:8.2f} ms   {coords.shape[0]:,} active voxels")
    print(f"\nSpeedup: {dense_t/sparse_t:.2f}x")


if __name__ == "__main__":
    main()
