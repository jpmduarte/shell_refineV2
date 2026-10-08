# The nnU-Net's training augmentation (v2.8, nnUNetTrainer.get_training_transforms),
# reimplemented in numpy for an image and its signed distance field (SDF). Shared by every
# approach in this repo.
#
# The same transforms, order, ranges and probabilities as nnU-Net, with two changes for an
# SDF target:
#   - Scaling a volume by k scales every distance by k, so the SDF is rescaled with the
#     image. That is why the SDF comes in millimetres and untruncated (caches/cache/sdf/x4mm) and
#     is truncated only at the end.
#   - The SDF is interpolated linearly, where nnU-Net resamples a label.
#
# Pieces, usable on their own:
#   mirror               flip image and SDF along each axis with probability 0.5
#   spatial_transform    rotation and scaling, sampled straight into a cube
#   intensity_transform  noise, blur, brightness, contrast, low resolution, gamma
#                        (image only; works on any volume)
#   augment_nnunet       all of the above in nnU-Net's order, plus padding and truncation
#   spatial_patch        rotation and scaling of a patch about its own centre, sampled
#                        from a larger volume, for any number of aligned arrays (used by
#                        the boundary refiner's cubes)
#
# Every random draw comes from the numpy Generator passed in, in a fixed order, so a seeded
# run reproduces exactly.

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates

TRUNC_MM = 10.0   # the SDF's truncation (make_sdf.py)
FAR_MM = 1e3      # SDF value used outside the scan: far outside, so +1 once truncated


# Pad a volume at the far end of each axis up to cube^3, with a constant.
def pad_to_cube(vol: np.ndarray, cube: int, cval: float) -> np.ndarray:
    pads = [(0, cube - s) for s in vol.shape]
    if any(p[1] < 0 for p in pads):
        raise ValueError(f"shape {vol.shape} exceeds cube={cube}")
    return np.pad(vol, pads, mode="constant", constant_values=cval)


# A random factor in [lo, hi], drawn as batchgeneratorsv2's BGContrast does: half the
# time below 1, half above.
def bg_contrast(rng, lo, hi):
    if rng.random() < 0.5 and lo < 1:
        return rng.uniform(lo, 1)
    return rng.uniform(max(lo, 1), hi)


# A random rotation matrix: an angle in [-max_rad, max_rad] about each axis.
def random_rotation(rng, max_rad):
    ax, ay, az = rng.uniform(-max_rad, max_rad, 3)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


# Mirroring: each axis flipped with probability 0.5, image and SDF together.
def mirror(img, sdf_mm, rng):
    for axis in range(3):
        if rng.random() < 0.5:
            img, sdf_mm = np.flip(img, axis), np.flip(sdf_mm, axis)
    # np.flip returns views with negative strides; make them ordinary arrays again.
    return np.ascontiguousarray(img), np.ascontiguousarray(sdf_mm)


# Rotation (up to +-30 degrees per axis, probability 0.2) and isotropic scaling (0.7 to
# 1.4, probability 0.2) about the volume's centre, sampled straight into the cube.
# Returns (image, SDF in mm), both cube-shaped, or None when neither was drawn.
#
# As in batchgeneratorsv2, the scale multiplies the sampling coordinates: 1.4 shrinks the
# head and 0.7 enlarges it. Every distance changes by 1/scale, hence the division below.
def spatial_transform(img, sdf_mm, rng, cube):
    do_rot   = rng.random() < 0.2
    do_scale = rng.random() < 0.2
    if not (do_rot or do_scale):
        return None
    m = random_rotation(rng, np.deg2rad(30)) if do_rot else np.eye(3)
    s = rng.uniform(0.7, 1.4) if do_scale else 1.0
    m = m @ (np.eye(3) * s)

    # For every voxel of the output cube, the point of the input it samples: offset from
    # the centre, rotated and scaled, then placed back around the input's centre.
    centre = (np.array(img.shape) - 1) / 2
    grid = np.indices((cube,) * 3, dtype=np.float64).reshape(3, -1) - centre[:, None]
    coords = (m @ grid + centre[:, None]).reshape(3, cube, cube, cube)

    # Linear interpolation. Outside the scan: 0 in the image (its background), far
    # outside in the SDF.
    img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    sdf_mm = map_coordinates(sdf_mm, coords, order=1, mode="constant", cval=FAR_MM) / s
    return img.astype(np.float32), sdf_mm.astype(np.float32)


# The same rotation and scaling as spatial_transform (same probabilities, ranges and
# draw order), for a patch: a cube of side `cube` whose corner is `origin` in the given
# arrays, rotated and scaled about the cube's own centre and sampled from the arrays
# around it, so a rotated patch is filled with real data rather than padding. Every array
# is transformed identically (linear interpolation); `cvals` gives each one's value
# beyond the arrays' edge.
#
# Returns (list of cube-shaped arrays, scale). Distances in an SDF change by 1/scale, so
# the caller divides SDFs by it. When neither transform is drawn, the patch is cut out
# exactly, with no interpolation.
def spatial_patch(arrays, cvals, origin, rng, cube):
    do_rot   = rng.random() < 0.2
    do_scale = rng.random() < 0.2
    if not (do_rot or do_scale):
        sl = tuple(slice(o, o + cube) for o in origin)
        return [a[sl].astype(np.float32) for a in arrays], 1.0
    m = random_rotation(rng, np.deg2rad(30)) if do_rot else np.eye(3)
    s = rng.uniform(0.7, 1.4) if do_scale else 1.0
    m = m @ (np.eye(3) * s)

    # Offsets of the cube's voxels from its centre, rotated and scaled, placed around the
    # same centre in the arrays. With no transform this is exactly the plain cut.
    half = (cube - 1) / 2
    pivot = np.array(origin, dtype=np.float64) + half
    grid = np.indices((cube,) * 3, dtype=np.float64).reshape(3, -1) - half
    coords = (m @ grid + pivot[:, None]).reshape(3, cube, cube, cube)
    out = [map_coordinates(a, coords, order=1, mode="constant", cval=cv).astype(np.float32)
           for a, cv in zip(arrays, cvals)]
    return out, s


# Gamma correction (exponent 0.7 to 1.5) that keeps the image's mean and std: it reshapes
# the histogram only. With invert=True it acts on the negated image.
def gamma(x, rng, invert):
    if invert:
        x = -x
    mean, std = x.mean(), x.std()
    lo, value_range = x.min(), x.max() - x.min()
    # Map to [0, 1], apply the exponent, map back.
    x = np.power((x - lo) / (value_range + 1e-7), bg_contrast(rng, 0.7, 1.5)) * (value_range + 1e-7) + lo
    # Restore the original mean and std.
    x = (x - x.mean()) / (x.std() + 1e-8) * std + mean
    return -x if invert else x


# Intensity augmentation, each with its own probability, in the nnU-Net's order.
def intensity_transform(x, rng):
    # Gaussian noise.
    if rng.random() < 0.1:
        x = x + rng.normal(0.0, np.sqrt(rng.uniform(0, 0.1)), x.shape)
    # Gaussian blur.
    if rng.random() < 0.2 and rng.random() < 0.5:
        x = gaussian_filter(x, sigma=rng.uniform(0.5, 1.0, 3))
    # Brightness: multiply by a factor.
    if rng.random() < 0.15:
        x = x * bg_contrast(rng, 0.75, 1.25)
    # Contrast: stretch around the mean, keeping the original value range.
    if rng.random() < 0.15:
        lo, hi, mean = x.min(), x.max(), x.mean()
        x = np.clip((x - mean) * bg_contrast(rng, 0.75, 1.25) + mean, lo, hi)
    # Simulated low resolution: shrink (nearest) and enlarge back (trilinear).
    if rng.random() < 0.25 and rng.random() < 0.5:
        shape, zoom = x.shape, rng.uniform(0.5, 1.0)
        small = [max(1, round(n * zoom)) for n in shape]
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
        t = F.interpolate(t, size=small, mode="nearest-exact")
        x = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)[0, 0].numpy()
    # Gamma on the inverted image, then on the image itself.
    if rng.random() < 0.1:
        x = gamma(x, rng, invert=True)
    if rng.random() < 0.3:
        x = gamma(x, rng, invert=False)
    return x


# The full augmentation: unpadded image and SDF in mm in, cube-shaped image and
# normalised SDF ([-1, 1]) out.
def augment_nnunet(img, sdf_mm, rng, cube, trunc_mm=TRUNC_MM):
    # Mirroring first (the nnU-Net does it last), on the unpadded arrays, so the scan stays
    # at the origin of the cube, where prediction puts it. Rotations are symmetric, so the
    # order changes nothing else.
    img, sdf_mm = mirror(img, sdf_mm, rng)

    # Rotation and scaling, which also pad to the cube; if neither was drawn, just pad.
    out = spatial_transform(img, sdf_mm, rng, cube)
    if out is None:
        img = pad_to_cube(img, cube, cval=0.0)
        sdf_mm = pad_to_cube(sdf_mm, cube, cval=FAR_MM)
    else:
        img, sdf_mm = out

    # Truncate and normalise the target only now, after any rescaling.
    sdf = np.clip(sdf_mm, -trunc_mm, trunc_mm) / trunc_mm
    return intensity_transform(img, rng).astype(np.float32), sdf.astype(np.float32)
