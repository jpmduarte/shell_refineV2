# Resampling between the native grid and the coarse ones.
#
# Every volume carries an affine: the 4x4 matrix that maps a voxel index to its position
# in millimetres. Two grids of the same scan differ only in their affine and shape, so
# resampling means: for each output voxel, find where it lies in the input and
# interpolate there.
#
#   downsample                      builds the caches (MONAI's Spacingd)
#   resample_grid + apply_resample  the same map as an explicit per-voxel grid (the
#                                   slower "geometry" resampler of infer.py)
#   spacing_grid + affine_resample  both directions at inference, on the GPU
#                                   (size_normalization/infer.py): the same
#                                   voxels as the two above, at a fraction of the cost
#
# About `cval`: grid_sample has no "fill with a constant" mode, only "repeat the border".
# So the constant is subtracted before sampling and added back after. An SDF needs +1
# (far outside the head) beyond the volume; padding it with 0 would invent a surface
# along the border.

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Spacingd

# The network's grid, defined once here for every script.
FACTOR = 4        # the x4 grid: 4 x the native spacing (1.6 mm before size normalisation)
CUBE = 96         # network input per axis: a multiple of 8 (three poolings) fitting every case


# Resample item[key] to `factor` times its current spacing (factor 4: 0.4 mm -> 1.6 mm).
# Trilinear interpolation. Used for images and SDFs; never for the label, which would
# pin the surface to the coarse grid.
def downsample(item, factor, key="image"):
    pixdim = tuple(float(p) * factor for p in item[key].pixdim)
    return Spacingd(keys=[key], pixdim=pixdim, mode="bilinear",
                    padding_mode="border", align_corners=False)(item)


# For every voxel of the destination grid, its position in the source grid, in source
# voxel coordinates. Shape (D', H', W', 3). Built in float64 on `device`.
#
# Resampling by voxel index instead (a plain F.interpolate) would stretch each axis by its
# own ratio of shapes, which differ per axis and per case, and so distort the anatomy.
def resample_grid(src_affine, dst_affine, dst_shape, device=None):
    dev = torch.device(device) if device is not None else torch.device("cpu")
    src = torch.as_tensor(src_affine, dtype=torch.float64).to(dev)
    dst = torch.as_tensor(dst_affine, dtype=torch.float64).to(dev)

    # The index of every destination voxel, as homogeneous coordinates (i, j, k, 1).
    zz, yy, xx = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float64, device=dev) for s in dst_shape],
        indexing="ij")
    ones = torch.ones_like(zz)
    dst_vox = torch.stack([zz, yy, xx, ones], dim=-1)

    # destination index -> mm (dst_affine) -> source index (inverse of src_affine).
    to_src = torch.linalg.solve(src, dst)
    return torch.einsum("ij,...j->...i", to_src, dst_vox)[..., :3]


# Sample img (N, C, D, H, W) at the positions resample_grid gave (in a src_shape volume).
def apply_resample(img, src_vox, src_shape, cval=0.0):
    # grid_sample wants positions in [-1, 1], where -1 and 1 are the outer edges of the
    # first and last voxel (align_corners=False).
    shape = torch.tensor(src_shape, dtype=src_vox.dtype, device=src_vox.device)
    norm = (2.0 * src_vox + 1.0) / shape - 1.0
    # ... and in (x, y, z) order, the reverse of the array's axis order; one per sample.
    grid = norm.flip(-1)[None].to(img.dtype)
    grid = grid.expand(img.shape[0], *grid.shape[1:])

    return F.grid_sample(img - cval, grid, mode="bilinear",
                         padding_mode="border", align_corners=False) + cval


# The affine and shape that downsample (Spacingd) produces for a given isotropic spacing,
# computed without resampling anything: same first voxel centre, voxel size `spacing`,
# and enough voxels for the centres to span the source's.
def spacing_grid(affine, shape, spacing):
    src_spacing = np.abs(np.diag(affine[:3, :3]))
    out_shape = tuple(int(round((n - 1) * s / spacing + 1)) for n, s in zip(shape, src_spacing))
    out = np.array(affine, dtype=np.float64)
    # Keep each axis' direction (the sign), change only the voxel size.
    out[:3, :3] = np.diag(np.sign(np.diag(affine[:3, :3])) * spacing)
    return out, out_shape


# Resample img (D, H, W) from one grid to another, on img's device.
#
# Same result as resample_grid + apply_resample (checked against the stored caches to
# 7e-5), but between our grids the map is only a scale and a shift per axis, so
# affine_grid can generate the sampling grid from a 3x4 matrix in float32, instead of
# computing 19 million coordinates in float64 (~200 ms on a native volume).
def affine_resample(img, src_affine, dst_affine, dst_shape, cval=0.0):

    # Matrix taking a voxel index (z, y, x) of a volume of this shape to grid_sample's
    # [-1, 1] coordinates in (x, y, z) order.
    def to_unit(shape):
        n = np.array(shape, dtype=np.float64)
        scale = np.eye(4)
        scale[:3, :3] = np.diag(2.0 / n)
        scale[:3, 3] = 1.0 / n - 1
        reverse_axes = np.eye(4)
        reverse_axes[:3, :3] = np.eye(3)[::-1]
        return reverse_axes @ scale

    # destination index -> source index, as one matrix.
    dst_to_src = np.linalg.solve(np.asarray(src_affine, np.float64),
                                 np.asarray(dst_affine, np.float64))
    # The same map expressed in grid_sample's units: destination [-1, 1] -> source [-1, 1].
    theta = to_unit(img.shape) @ dst_to_src @ np.linalg.inv(to_unit(dst_shape))
    theta = torch.tensor(theta[:3], dtype=torch.float32, device=img.device)[None]

    grid = F.affine_grid(theta, (1, 1, *dst_shape), align_corners=False)
    return F.grid_sample(img[None, None].float() - cval, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0, 0] + cval
