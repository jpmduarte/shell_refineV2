"""nnU-Net v2.8's training augmentation (nnUNetTrainer.get_training_transforms) in numpy, for an image and its SDF.

Same transforms, order, ranges and probabilities, with two changes for an SDF target: the SDF comes
in mm and untruncated, is rescaled with the image (zooming by k scales distances by k) and truncated
only at the end; and it is interpolated linearly where nnU-Net resamples a label.
Every random draw comes from the Generator passed in, in a fixed order, so a seeded run reproduces exactly.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates

from common.geometry import add_batch_axes, drop_batch_axes
from common.sdf_targets import TRUNC_MM

FAR_MM = 1e3   # SDF outside the scan: far outside, so +1 once truncated

MIRROR_P = 0.5
ROTATION_P = 0.2
MAX_ROTATION_DEG = 30
SCALE_P = 0.2
SCALE_RANGE = (0.7, 1.4)   # multiplies the sampling coordinates: 1.4 shrinks the head, 0.7 enlarges it
GAMMA_RANGE = (0.7, 1.5)
EPS_RANGE = 1e-7
EPS_STD = 1e-8


def pad_to_cube(vol: np.ndarray, cube: int, cval: float) -> np.ndarray:
    """Pad at the far end of each axis up to cube^3."""
    pads = [(0, cube - s) for s in vol.shape]
    if any(after < 0 for _, after in pads):
        raise ValueError(f"shape {vol.shape} exceeds cube={cube}")
    return np.pad(vol, pads, mode="constant", constant_values=cval)


def bg_contrast(rng, lo, hi):
    """A factor in [lo, hi] drawn as batchgeneratorsv2's BGContrast: half the time below 1, half above."""
    if rng.random() < 0.5 and lo < 1:
        return rng.uniform(lo, 1)
    return rng.uniform(max(lo, 1), hi)


def random_rotation(rng, max_rad):
    """An angle in [-max_rad, max_rad] about each axis."""
    ax, ay, az = rng.uniform(-max_rad, max_rad, 3)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def random_rotation_and_scale(rng, scale_p=SCALE_P, scale_range=SCALE_RANGE):
    """(matrix, scale) of the drawn rotation and isotropic scaling, or None when neither was drawn."""
    do_rotation = rng.random() < ROTATION_P
    do_scale = rng.random() < scale_p
    if not (do_rotation or do_scale):
        return None
    matrix = random_rotation(rng, np.deg2rad(MAX_ROTATION_DEG)) if do_rotation else np.eye(3)
    scale = rng.uniform(*scale_range) if do_scale else 1.0
    return matrix @ (np.eye(3) * scale), scale


def sampling_coords(matrix, cube, output_centre, input_centre):
    """For every voxel of the output cube, the input point it samples: its offset from the centre, transformed."""
    offsets = np.indices((cube,) * 3, dtype=np.float64).reshape(3, -1) - output_centre
    return (matrix @ offsets + input_centre[:, None]).reshape(3, cube, cube, cube)


def mirror(img, sdf_mm, rng):
    for axis in range(3):
        if rng.random() < MIRROR_P:
            img, sdf_mm = np.flip(img, axis), np.flip(sdf_mm, axis)
    return np.ascontiguousarray(img), np.ascontiguousarray(sdf_mm)   # np.flip leaves negative strides


def spatial_transform(img, sdf_mm, rng, cube, scale_p=SCALE_P, scale_range=SCALE_RANGE):
    """Rotation and scaling about the volume's centre, sampled into the cube: (image, SDF mm), or None if not drawn."""
    drawn = random_rotation_and_scale(rng, scale_p, scale_range)
    if drawn is None:
        return None
    matrix, scale = drawn
    centre = (np.array(img.shape) - 1) / 2
    coords = sampling_coords(matrix, cube, centre[:, None], centre)

    img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    sdf_mm = map_coordinates(sdf_mm, coords, order=1, mode="constant", cval=FAR_MM) / scale
    return img.astype(np.float32), sdf_mm.astype(np.float32)


def spatial_patch(arrays, cvals, origin, rng, cube):
    """spatial_transform for a patch: the cube at `origin`, turned about its own centre and filled with real data.

    Every array is transformed alike, `cvals` being each one's value beyond the edge. Returns
    (cube-shaped arrays, scale); the caller divides SDFs by the scale. Undrawn, it is the exact cut.
    """
    drawn = random_rotation_and_scale(rng)
    if drawn is None:
        patch = tuple(slice(o, o + cube) for o in origin)
        return [a[patch].astype(np.float32) for a in arrays], 1.0
    matrix, scale = drawn
    half = (cube - 1) / 2
    pivot = np.array(origin, dtype=np.float64) + half
    coords = sampling_coords(matrix, cube, half, pivot)
    out = [map_coordinates(a, coords, order=1, mode="constant", cval=cv).astype(np.float32)
           for a, cv in zip(arrays, cvals)]
    return out, scale


def gamma(x, rng, invert):
    """Gamma correction that keeps the mean and std, reshaping the histogram only; `invert` acts on -x."""
    if invert:
        x = -x
    mean, std = x.mean(), x.std()
    lo, value_range = x.min(), x.max() - x.min()
    in_unit_range = (x - lo) / (value_range + EPS_RANGE)
    x = np.power(in_unit_range, bg_contrast(rng, *GAMMA_RANGE)) * (value_range + EPS_RANGE) + lo
    x = (x - x.mean()) / (x.std() + EPS_STD) * std + mean
    return -x if invert else x


def simulate_low_resolution(x, zoom):
    """Shrink (nearest) and enlarge back (trilinear)."""
    shape = x.shape
    small = [max(1, round(n * zoom)) for n in shape]
    t = add_batch_axes(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
    t = F.interpolate(t, size=small, mode="nearest-exact")
    return drop_batch_axes(F.interpolate(t, size=shape, mode="trilinear", align_corners=False)).numpy()


def intensity_transform(x, rng):
    """Noise, blur, brightness, contrast, low resolution and gamma, each with its own probability, in nnU-Net's order."""
    if rng.random() < 0.1:
        x = x + rng.normal(0.0, np.sqrt(rng.uniform(0, 0.1)), x.shape)
    if rng.random() < 0.2 and rng.random() < 0.5:
        x = gaussian_filter(x, sigma=rng.uniform(0.5, 1.0, 3))
    if rng.random() < 0.15:
        x = x * bg_contrast(rng, 0.75, 1.25)   # brightness
    if rng.random() < 0.15:
        lo, hi, mean = x.min(), x.max(), x.mean()
        x = np.clip((x - mean) * bg_contrast(rng, 0.75, 1.25) + mean, lo, hi)   # contrast, within the range
    if rng.random() < 0.25 and rng.random() < 0.5:
        x = simulate_low_resolution(x, rng.uniform(0.5, 1.0))
    if rng.random() < 0.1:
        x = gamma(x, rng, invert=True)
    if rng.random() < 0.3:
        x = gamma(x, rng, invert=False)
    return x


def augment_nnunet(img, sdf_mm, rng, cube, trunc_mm=TRUNC_MM, scale_p=SCALE_P, scale_range=SCALE_RANGE):
    """Unpadded image and SDF in mm in; cube-shaped image and normalised SDF out."""
    # Mirroring first (nnU-Net does it last) so the scan stays at the cube's origin, where prediction puts it.
    img, sdf_mm = mirror(img, sdf_mm, rng)

    transformed = spatial_transform(img, sdf_mm, rng, cube, scale_p, scale_range)
    if transformed is None:
        img = pad_to_cube(img, cube, cval=0.0)
        sdf_mm = pad_to_cube(sdf_mm, cube, cval=FAR_MM)
    else:
        img, sdf_mm = transformed

    sdf = np.clip(sdf_mm, -trunc_mm, trunc_mm) / trunc_mm   # only now, after any rescaling
    return intensity_transform(img, rng).astype(np.float32), sdf.astype(np.float32)
