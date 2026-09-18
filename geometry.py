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
