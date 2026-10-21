"""Resampling between grids of one scan; each grid is an affine (voxel index -> mm) and a shape."""
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Spacingd

FACTOR = 4   # the network's grid: 4 x the native spacing
CUBE = 96    # network input size per axis; a multiple of 8 for the three poolings

SPATIAL = slice(0, 3)   # rows/columns of an affine that hold the voxel axes
MM3_PER_ML = 1000


def voxel_spacing(affine) -> np.ndarray:
    return np.abs(np.diag(affine[SPATIAL, SPATIAL]))


def isotropic_spacing(affine) -> float:
    return float(np.abs(affine[0, 0]))


def add_batch_axes(volume):
    """(D, H, W) -> (1, 1, D, H, W), the shape a network or grid_sample expects."""
    return volume[None, None]


def drop_batch_axes(volume):
    """(1, 1, D, H, W) -> (D, H, W)."""
    return volume[0, 0]


def crop_to_shape(volume, shape):
    """Cut the padding off the last three axes."""
    depth, height, width = shape
    return volume[..., :depth, :height, :width]


def downsample(item, factor, key="image"):
    """MONAI's Spacingd to `factor` x the current spacing; never use it on a label."""
    pixdim = tuple(float(p) * factor for p in item[key].pixdim)
    return Spacingd(keys=[key], pixdim=pixdim, mode="bilinear",
                    padding_mode="border", align_corners=False)(item)


def resample_grid(src_affine, dst_affine, dst_shape, device=None):
    """For every destination voxel, its position in source voxel coordinates, (D, H, W, 3)."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    src = torch.as_tensor(src_affine, dtype=torch.float64).to(device)
    dst = torch.as_tensor(dst_affine, dtype=torch.float64).to(device)

    zz, yy, xx = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64, device=device) for n in dst_shape], indexing="ij")
    dst_index = torch.stack([zz, yy, xx, torch.ones_like(zz)], dim=-1)   # homogeneous

    dst_index_to_src_index = torch.linalg.solve(src, dst)
    return torch.einsum("ij,...j->...i", dst_index_to_src_index, dst_index)[..., SPATIAL]


def apply_resample(volumes, src_positions, src_shape, cval=0.0):
    """Sample (N, C, D, H, W) at the positions from resample_grid; outside the volume: cval."""
    shape = torch.tensor(src_shape, dtype=src_positions.dtype, device=src_positions.device)
    unit = (2.0 * src_positions + 1.0) / shape - 1.0          # grid_sample's [-1, 1]
    grid = unit.flip(-1)[None].to(volumes.dtype)                # (z, y, x) -> (x, y, z)
    grid = grid.expand(volumes.shape[0], *grid.shape[1:])
    # grid_sample can only repeat the border, so the constant is subtracted and added back.
    return F.grid_sample(volumes - cval, grid, mode="bilinear",
                         padding_mode="border", align_corners=False) + cval


def spacing_grid(affine, shape, spacing):
    """The affine and shape Spacingd would produce for an isotropic `spacing`, without resampling."""
    src_spacing = voxel_spacing(affine)
    out_shape = tuple(int(round((n - 1) * s / spacing + 1)) for n, s in zip(shape, src_spacing))
    out = np.array(affine, dtype=np.float64)
    axis_directions = np.sign(np.diag(affine[SPATIAL, SPATIAL]))
    out[SPATIAL, SPATIAL] = np.diag(axis_directions * spacing)
    return out, out_shape


def _index_to_unit(shape):
    """Matrix from a voxel index (z, y, x) to grid_sample's [-1, 1] coordinates in (x, y, z)."""
    n = np.array(shape, dtype=np.float64)
    scale = np.eye(4)
    scale[SPATIAL, SPATIAL] = np.diag(2.0 / n)
    scale[SPATIAL, 3] = 1.0 / n - 1
    reverse_axes = np.eye(4)
    reverse_axes[SPATIAL, SPATIAL] = np.eye(3)[::-1]
    return reverse_axes @ scale


def affine_resample(volume, src_affine, dst_affine, dst_shape, cval=0.0):
    """resample_grid + apply_resample for a (D, H, W) volume, from one 3x4 matrix (our grids only scale and shift)."""
    dst_index_to_src_index = np.linalg.solve(np.asarray(src_affine, np.float64),
                                             np.asarray(dst_affine, np.float64))
    theta = _index_to_unit(volume.shape) @ dst_index_to_src_index @ np.linalg.inv(_index_to_unit(dst_shape))
    theta = torch.tensor(theta[SPATIAL], dtype=torch.float32, device=volume.device)[None]

    grid = F.affine_grid(theta, (1, 1, *dst_shape), align_corners=False)
    resampled = F.grid_sample(add_batch_axes(volume).float() - cval, grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
    return drop_batch_axes(resampled) + cval
