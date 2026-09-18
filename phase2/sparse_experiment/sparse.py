"""Submanifold sparse 3D convolution built from plain torch ops — no CUDA extension.

Exists because spconv/torchsparse/MinkowskiEngine all predate Blackwell (sm_120):
their prebuilt wheels have no kernel for this GPU, and building from source hits the
same wall (their CUDA codegen was last touched ~2024/2025, before sm_120 existed).
This runs on whatever torch already supports, at the cost of a hand-tuned kernel's
speed — but compute here still scales with active voxel count, not volume, which is
the property that actually matters for the band-only refinement this exists for.

A "sparse tensor" here is just two parallel tensors:
    coords:   (N, 4) int64   [batch, z, y, x]
    features: (N, C) float

Submanifold: the output active set is always identical to the input active set (never
grows or shrinks) — the only thing that changes through a stack of these is the
feature dimension, exactly like a stride-1 dense conv with padding.
"""

import torch
import torch.nn as nn

# Large odd primes, chosen so batch/z/y/x collisions are astronomically unlikely for
# any volume this pipeline will ever see (native fetal head volumes top out ~400 vox/axis).
_HASH_PRIMES = torch.tensor([1_000_000_007, 2_654_435_761, 40_503, 1], dtype=torch.int64)


def _hash_coords(coords: torch.Tensor) -> torch.Tensor:
    """(N, 4) int64 [b,z,y,x] -> (N,) int64 unique key, order-independent of magnitude."""
    return (coords.to(torch.int64) * _HASH_PRIMES.to(coords.device)).sum(dim=1)


def build_neighbor_index(coords: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """For each active voxel and each of the kernel's offsets, find the row index of
    that neighbor in `coords` if it's active, else -1.

    coords:  (N, 4) int64 [b, z, y, x]
    offsets: (K, 3) int64 relative (dz, dy, dx), e.g. the 27 offsets of a 3x3x3 kernel
    returns: (K, N) int64, neighbor_idx[k, i] = row of coords for voxel i's k-th
             neighbor, or -1 if that neighbor isn't in the active set.
    """
    device = coords.device
    n = coords.shape[0]
    k = offsets.shape[0]
    keys = _hash_coords(coords)  # (N,) — a voxel's key is linear in its coords, so a
    order = torch.argsort(keys)   # shifted voxel's key is just keys + a per-offset
    sorted_keys = keys[order]     # constant. Avoids ever materializing a (K, N, 4)
                                   # broadcasted coordinate tensor (was the actual
                                   # memory blowup: ~2.9GB per intermediate at N=3.4M).
    zero_batch_col = torch.zeros(k, 1, dtype=torch.int64, device=device)
    offset_keys = _hash_coords(torch.cat([zero_batch_col, offsets.to(device)], dim=1))  # (K,)

    neighbor_idx = torch.empty((k, n), dtype=torch.int64, device=device)
    for i in range(k):
        shifted_keys = keys + offset_keys[i]  # (N,) — one offset at a time, no (K,N) blowup
        pos = torch.searchsorted(sorted_keys, shifted_keys)
        pos_clamped = pos.clamp(max=n - 1)
        found = (pos < n) & (sorted_keys[pos_clamped] == shifted_keys)
        neighbor_idx[i] = torch.where(found, order[pos_clamped], torch.full_like(pos, -1))

    return neighbor_idx


def cube_offsets(kernel_size: int = 3) -> torch.Tensor:
    """All (dz, dy, dx) offsets of a cubic kernel, kernel_size odd, centred on 0."""
    r = kernel_size // 2
    axis = torch.arange(-r, r + 1)
    dz, dy, dx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([dz.reshape(-1), dy.reshape(-1), dx.reshape(-1)], dim=1)


class SubMConv3d(nn.Module):
    """Submanifold 3D conv: active set in == active set out. Bias-free, like spconv's
    default (a norm layer typically follows and absorbs the bias)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.offsets = nn.Parameter(cube_offsets(kernel_size), requires_grad=False)
        k = self.offsets.shape[0]
        # One weight matrix per kernel offset, matching a dense Conv3d's per-tap weight.
        self.weight = nn.Parameter(torch.empty(k, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weight.view(k * in_channels, out_channels), a=5 ** 0.5)

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
               neighbor_idx: torch.Tensor = None) -> torch.Tensor:
        """coords: (N,4), features: (N,C_in). neighbor_idx: precomputed by
        build_neighbor_index if the caller wants to reuse it across layers on the same
        active set (submanifold conv never changes the active set, so this is normally
        computed once per forward pass of the whole network, not once per layer).
        """
        if neighbor_idx is None:
            neighbor_idx = build_neighbor_index(coords, self.offsets)

        n = features.shape[0]
        out = features.new_zeros(n, self.weight.shape[-1])
        zero_row = features.new_zeros(1, features.shape[1])
        padded = torch.cat([features, zero_row], dim=0)  # index -1 -> the zero row

        for k in range(self.offsets.shape[0]):
            gathered = padded[neighbor_idx[k]]           # (N, C_in), 0 where absent
            out = out + gathered @ self.weight[k]         # (N, C_out)

        return out


def SparseBatchNorm(num_features: int) -> nn.Module:
    """A sparse tensor's features here are already (N, C) — plain BatchNorm1d is
    exactly the right op, no sparse-specific logic needed."""
    return nn.BatchNorm1d(num_features)
