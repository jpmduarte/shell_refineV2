# Radial (star-convex) targets for the polar head: from a point inside the head, the
# distance to the head's surface along each of a fixed set of directions, as in StarDist
# 3D (Weigert et al., WACV 2020).
#
# The SDF tells a voxel how far the nearest surface is; the radii tell it where the
# surface is in every direction, which needs the whole head in view. Trained as an
# auxiliary output (train.py here), they push the shared features towards that global
# context; prediction still uses the SDF alone.
#
# v1 (v1/results.md) supervised 128 random interior points per sample (`targets`); v2
# supervises the head's centre only (`centre_targets`).

import numpy as np
from scipy.ndimage import map_coordinates

N_DIRS = 64          # directions on the sphere
N_POINTS = 128       # v1: interior points supervised per training sample
RADIUS_SCALE = 20.0  # v1: voxels; radii were predicted and compared divided by this
MEAN_RADIUS_VOX = 19.6   # a 130 mL head (V0) as a sphere: 31.4 mm, on the 1.6 mm x4 grid
CENTRE_JITTER_VOX = 1.0  # training centres are jittered by this (std per axis): pass 1's
                         # centre is 1.26 mm (~0.8 voxel) from the label's at inference


# n directions spread evenly over the sphere (a Fibonacci lattice), as unit vectors in
# voxel axes (z, y, x order of the arrays). Fixed: the head's channel i always means the
# same direction.
def directions(n: int = N_DIRS) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z ** 2)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([z, r * np.cos(phi), r * np.sin(phi)], axis=1)


# Distance (voxels) from each point to the surface {sdf = 0} along each direction, by
# sphere tracing: inside the head the SDF is minus the distance to the nearest surface, so
# stepping by it never crosses the surface; the last step is refined by bisection.
# sdf_vox: the SDF in voxels (negative inside); beyond the array it counts as outside.
# points (K, 3) voxel coordinates inside the head; dirs (D, 3). Returns (K, D).
def ray_radii(sdf_vox: np.ndarray, points: np.ndarray, dirs: np.ndarray,
              steps: int = 40, min_step: float = 0.5) -> np.ndarray:
    k, d = len(points), len(dirs)
    origin = np.repeat(points[:, None, :], d, axis=1).reshape(-1, 3)
    ray = np.tile(dirs, (k, 1))

    def sample(t):
        pos = origin + ray * t[:, None]
        return map_coordinates(sdf_vox, pos.T, order=1, mode="constant", cval=1e3)

    t = np.zeros(k * d)
    inside = np.ones(k * d, dtype=bool)
    for _ in range(steps):
        if not inside.any():
            break
        v = sample(t)
        inside = v < 0
        t = np.where(inside, t + np.maximum(-v, min_step), t)
    # The surface lies within the last step: bisect between the last inside point and t.
    lo = np.maximum(t - np.maximum(min_step, 1.0), 0.0)
    hi = t
    for _ in range(6):
        mid = (lo + hi) / 2
        inside_mid = sample(mid) < 0
        lo = np.where(inside_mid, mid, lo)
        hi = np.where(inside_mid, hi, mid)
    return ((lo + hi) / 2).reshape(k, d).astype(np.float32)


# v2 training target for one sample: the head's centre (the centroid of SDF < 0, jittered
# like pass 1's estimate) and the radii from it, in voxels. Returns (centre (3,) float32,
# radii (D,) float32, weight 1 or 0 when no head is in view).
def centre_targets(sdf: np.ndarray, spacing_mm: float, trunc_mm: float, rng, dirs: np.ndarray):
    inside = np.argwhere(sdf < 0)
    if len(inside) == 0:
        return np.full(3, sdf.shape[0] / 2, np.float32), np.ones(len(dirs), np.float32), 0.0
    c = inside.mean(0) + rng.normal(0.0, CENTRE_JITTER_VOX, 3)
    sdf_vox = sdf.astype(np.float32) * trunc_mm / spacing_mm
    r = ray_radii(sdf_vox, c[None], dirs)[0]
    # A jittered centre can land outside a thin or cut head (every ray then has radius 0):
    # that sample gets no polar loss.
    w = 1.0 if r.min() > 0.5 else 0.0
    return c.astype(np.float32), np.maximum(r, 0.5).astype(np.float32), w


# v1 training targets for one sample: N_POINTS random voxels inside the head and their radii.
# sdf: the normalised SDF on the cube (x trunc_mm = mm); spacing_mm: the grid's voxel size.
# Returns (points (K, 3) int64, radii (K, D) in voxels, weight 1 or 0 when no head is in view).
def targets(sdf: np.ndarray, spacing_mm: float, trunc_mm: float, rng, dirs: np.ndarray):
    inside = np.argwhere(sdf < 0)
    if len(inside) == 0:
        return np.zeros((N_POINTS, 3), np.int64), np.zeros((N_POINTS, len(dirs)), np.float32), 0.0
    pts = inside[rng.integers(0, len(inside), N_POINTS)]
    # Truncation (+-10 mm) only shortens the steps deep inside, never makes them cross.
    sdf_vox = sdf.astype(np.float32) * trunc_mm / spacing_mm
    return pts.astype(np.int64), ray_radii(sdf_vox, pts.astype(np.float64), dirs), 1.0
