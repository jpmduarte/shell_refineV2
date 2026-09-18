"""Anisotropic patches shaped like the band instead of like a cube.

The band is a shell of roughly constant thickness (~25 voxels at 0.4mm for band_mm=5),
so it is locally a slab: extended along the surface, thin across it. A cube spends most
of its volume on interior and exterior the refiner is not scored on — measured, a 64^3
band-centred cube is 44% band and a 128^3 one only 31%.

A slab of long x long x thin, with the thin axis along the surface normal, matches that
geometry: far more tangential context for the same voxel count, and a much higher
fraction of it supervised.

The normal varies over the surface, so an axis-aligned slab cannot match it everywhere.
Rather than rotate the volume into the local frame — which would mean interpolating, in
a pipeline whose whole point is not to disturb the metric — this picks, per patch,
whichever of the three axes is closest to the local normal and makes that one the thin
one. Worst case is a normal along the cube diagonal, ~54.7 degrees off, where the
effective thickness is 25/cos(54.7) ~ 43 voxels: worse than aligned, still far better
than a cube.

Patches are returned with the thin axis last, so the network always sees one shape and
can learn that the boundary runs across the last axis.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import PAD_VALUES  # noqa: E402

LONG = 128
THIN = 32
PATCHES_PER_CASE = 2


def pad_to_shape(arrays: dict, minimum: int) -> dict:
    """Pad every axis up to `minimum`, each array with its own fill value. The slab's
    thin axis is not known per patch until the normal is read, so pad for the worst
    case (long on every axis)."""
    shape = next(iter(arrays.values())).shape
    pads = [(0, max(0, minimum - s)) for s in shape]
    if all(p == (0, 0) for p in pads):
        return arrays
    return {name: np.pad(a, pads, mode="constant", constant_values=PAD_VALUES[name])
            for name, a in arrays.items()}


def local_normal_axis(sdf: np.ndarray, centre, radius: int = 4) -> int:
    """Which axis is most aligned with the SDF's gradient (the surface normal) here.

    Averaged over a small window: a single voxel's finite difference is noisy, and the
    prior is a network output, not an exact distance field.
    """
    sl = tuple(slice(max(0, c - radius), min(s, c + radius + 1))
               for c, s in zip(centre, sdf.shape))
    window = sdf[sl]
    if min(window.shape) < 3:
        return 2
    grads = np.gradient(window.astype(np.float32))
    return int(np.argmax([abs(float(g.mean())) for g in grads]))


def slab_origins(band: np.ndarray, sdf: np.ndarray, count: int, long: int, thin: int,
                 rng: np.random.Generator, min_sep: int = None, max_tries: int = 40):
    """Band-centred slab placements, spread apart as in the cube sampler.

    Returns (origin, thin_axis, shape) per patch: the thin axis is chosen from the
    local normal, so the extracted box differs in orientation from patch to patch.
    """
    flat = np.flatnonzero(band)
    if min_sep is None:
        min_sep = long // 2
    sep_sq = float(min_sep) ** 2

    if flat.size == 0:
        centres = [tuple(s // 2 for s in band.shape)] * count
    else:
        centres = []
        for _ in range(count):
            candidate = None
            for _attempt in range(max_tries):
                idx = flat[rng.integers(0, flat.size)]
                candidate = tuple(int(c) for c in np.unravel_index(idx, band.shape))
                if not centres:
                    break
                d_sq = min(sum((a - b) ** 2 for a, b in zip(candidate, c)) for c in centres)
                if d_sq >= sep_sq:
                    break
            centres.append(candidate)

    out = []
    for centre in centres:
        axis = local_normal_axis(sdf, centre)
        shape = [long, long, long]
        shape[axis] = thin
        origin = tuple(int(np.clip(c - e // 2, 0, max(0, s - e)))
                       for c, e, s in zip(centre, shape, band.shape))
        out.append((origin, axis, tuple(shape)))
    return out


def _augment_slab(arrays: list, rng: np.random.Generator):
    """Flips on every axis, plus 90-degree rotation only in the plane of the two long
    axes. A rotation involving the thin axis would move it out of last position and
    break the single-input-shape invariant."""
    for axis in range(3):
        if rng.random() < 0.5:
            arrays = [np.flip(a, axis) for a in arrays]

    k = rng.integers(0, 4)
    if k:
        arrays = [np.rot90(a, k, (0, 1)) for a in arrays]

    img = arrays[0] * rng.uniform(0.9, 1.1)
    img = img + rng.normal(0.0, 0.05, size=img.shape)
    arrays[0] = img
    return [np.ascontiguousarray(a) for a in arrays]


class SlabDataset(Dataset):
    """One item = PATCHES_PER_CASE band-centred slabs, thin axis last."""

    def __init__(self, crops_dir: str, case_ids: list[str], long: int = LONG,
                 thin: int = THIN, patches_per_case: int = PATCHES_PER_CASE,
                 augment: bool = False, seed: int = None):
        self.crops_dir = crops_dir
        self.case_ids = case_ids
        self.long = long
        self.thin = thin
        self.patches_per_case = patches_per_case
        self.augment = augment
        self._seed_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        data = np.load(os.path.join(self.crops_dir, f"{cid}.npz"))

        arrays = pad_to_shape({
            "img":       data["img"].astype(np.float32),
            "sdf_prior": data["sdf_prior"].astype(np.float32),
            "band":      data["band"].astype(np.float32),
            "mask":      data["mask"].astype(np.float32),
        }, self.long)
        img, prior, band, mask = (arrays[k] for k in ("img", "sdf_prior", "band", "mask"))

        rng = np.random.default_rng(int(self._seed_rng.integers(0, 2**31 - 1)))
        placements = slab_origins(band, prior, self.patches_per_case,
                                  self.long, self.thin, rng)

        xs, ys, bs = [], [], []
        for origin, axis, shape in placements:
            sl = tuple(slice(o, o + e) for o, e in zip(origin, shape))
            patch = [img[sl], prior[sl], mask[sl], band[sl]]
            # Thin axis last, so every patch reaches the network with one shape.
            patch = [np.moveaxis(a, axis, -1) for a in patch]

            if self.augment:
                patch = _augment_slab(patch, rng)

            img_p, prior_p, mask_p, band_p = patch
            xs.append(np.stack([img_p, prior_p], axis=0))
            ys.append(mask_p[np.newaxis])
            bs.append(band_p[np.newaxis])

        return (torch.from_numpy(np.stack(xs).astype(np.float32)),
                torch.from_numpy(np.stack(ys).astype(np.float32)),
                torch.from_numpy(np.stack(bs).astype(np.float32)))
