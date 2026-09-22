"""Down and up between spacings. The MetaTensor carries its own affine."""

import torch
from monai.transforms import Spacingd, SpatialResample


def downsample(item, factor, key="image"):
    """Resampling the label would pin the surface to a coarse grid, so leave it out."""
    pixdim = tuple(float(p) * factor for p in item[key].pixdim)
    return Spacingd(keys=[key], pixdim=pixdim, mode="bilinear",
                    padding_mode="border", align_corners=False)(item)


def upsample(img, affine, shape, cval=0.0):
    """Back to a stored affine and shape, so the factor's shape rounding cancels.

    grid_sample has no constant padding mode, so shift by cval and back. An SDF needs
    +1 outside, not 0, which would invent a surface at the border.
    """
    resample = SpatialResample(mode="bilinear", align_corners=False, dtype=torch.float64)
    return resample(img - cval, dst_affine=affine, spatial_size=shape) + cval


def resample_grid(src_affine, dst_affine, dst_shape, device=None):
    """The sampling grid SpatialResample would use, built once from the affines.

    Joint training needs to resample inside the autograd graph, and SpatialResample does
    not pass gradients. What it does underneath is grid_sample, which does — only the
    grid construction is opaque, and that depends solely on the affines, which are
    constants. So the grid can be built here and applied in the graph.

    Interpolating by voxel index instead, as a plain F.interpolate would, divides each
    axis by its own ratio of shapes. With x8 volumes varying per case (45x46x31 against
    24x28x19) those ratios differ per axis, which reintroduces exactly the anisotropy
    the cached grid exists to remove.

    Returns a grid for grid_sample with align_corners=False, in (x, y, z) order.
    """
    src = torch.as_tensor(src_affine, dtype=torch.float64)
    dst = torch.as_tensor(dst_affine, dtype=torch.float64)

    # Destination voxel centres -> world -> source voxel coordinates. Built wherever
    # `device` says: on a native volume this is 19M coordinates, and doing it on the CPU
    # in float64 measured 203ms against 4ms for the network it serves.
    dev = torch.device(device) if device is not None else torch.device("cpu")
    src, dst = src.to(dev), dst.to(dev)

    zz, yy, xx = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float64, device=dev) for s in dst_shape],
        indexing="ij")
    ones = torch.ones_like(zz)
    dst_vox = torch.stack([zz, yy, xx, ones], dim=-1)

    to_src = torch.linalg.solve(src, dst)
    return torch.einsum("ij,...j->...i", to_src, dst_vox)[..., :3]


def apply_resample(img, src_vox, src_shape, cval=0.0):
    """Sample img at src_vox (from resample_grid), differentiably.

    img: (N, C, D, H, W). src_vox: (D', H', W', 3) in source voxel coordinates.
    """
    import torch.nn.functional as F

    shape = torch.tensor(src_shape, dtype=src_vox.dtype, device=src_vox.device)
    norm = (2.0 * src_vox + 1.0) / shape - 1.0          # align_corners=False
    grid = norm.flip(-1)[None].to(img.dtype)            # grid_sample wants (x, y, z)
    grid = grid.expand(img.shape[0], *grid.shape[1:])

    return F.grid_sample(img - cval, grid, mode="bilinear",
                         padding_mode="border", align_corners=False) + cval
