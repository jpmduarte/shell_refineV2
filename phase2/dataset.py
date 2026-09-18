"""Phase 2 training dataset: random 64^3 patches centred on band voxels, drawn fresh
each epoch (diversity, not coverage — see phase2/make_crops.py for what a crop holds).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

PATCH_SIZE       = 64
PATCHES_PER_CASE = 8

# Zero is right for the binary masks, but for the signed distance zero means "on the
# surface" — padding with it would invent a boundary along the volume edge. +1 is the
# truncated "far outside" value.
PAD_VALUES = {"img": 0.0, "sdf_prior": 1.0, "band": 0.0, "mask": 0.0, "coarse_mask": 0.0}


def _augment(img, prior, mask, band, rng: np.random.Generator):
    for axis in range(3):
        if rng.random() < 0.5:
            img, prior, mask, band = (np.flip(a, axis) for a in (img, prior, mask, band))

    axes = [(0, 1), (0, 2), (1, 2)][rng.integers(0, 3)]
    k = rng.integers(0, 4)
    img, prior, mask, band = (np.rot90(a, k, axes) for a in (img, prior, mask, band))

    img = img * rng.uniform(0.9, 1.1)
    img = img + rng.normal(0.0, 0.05, size=img.shape)

    return (np.ascontiguousarray(img), np.ascontiguousarray(prior),
           np.ascontiguousarray(mask), np.ascontiguousarray(band))


def pad_to_patch(arrays: dict, patch: int) -> dict:
    shape = next(iter(arrays.values())).shape
    pads = [(0, max(0, patch - s)) for s in shape]
    if all(p == (0, 0) for p in pads):
        return arrays
    return {name: np.pad(a, pads, mode="constant", constant_values=PAD_VALUES[name])
           for name, a in arrays.items()}


def band_patch_origins(band: np.ndarray, count: int, patch: int,
                       rng: np.random.Generator, min_sep: int = None,
                       max_tries: int = 40) -> list[tuple[int, ...]]:
    """Pick `count` patch origins whose centre sits on a band voxel, spread apart.

    Independent uniform draws clump — with 8 draws it is easy to get three patches
    centred a few voxels apart (near-identical content, wasted gradient) while a whole
    region of the surface goes unsampled that epoch. Rejecting candidates closer than
    min_sep to an accepted centre (dart-throwing / Poisson-disk) fixes that while
    keeping the sampling fully random epoch to epoch.

    min_sep defaults to patch // 2, so two accepted patches can overlap by at most half
    their extent. Falls back to accepting a clustered candidate after max_tries so a
    small or fragmented band can never hang the loop.
    """
    flat = np.flatnonzero(band)
    if flat.size == 0:
        centres = [tuple(s // 2 for s in band.shape)] * count
    else:
        if min_sep is None:
            min_sep = patch // 2
        sep_sq = float(min_sep) ** 2

        centres = []
        for _ in range(count):
            candidate = None
            for attempt in range(max_tries):
                idx = flat[rng.integers(0, flat.size)]
                candidate = tuple(int(c) for c in np.unravel_index(idx, band.shape))
                if not centres:
                    break
                d_sq = min(sum((a - b) ** 2 for a, b in zip(candidate, c)) for c in centres)
                if d_sq >= sep_sq:
                    break
            centres.append(candidate)

    half = patch // 2
    return [tuple(int(np.clip(c - half, 0, max(0, s - patch))) for c, s in zip(centre, band.shape))
           for centre in centres]


class Phase2Dataset(Dataset):
    """One item = a stack of PATCHES_PER_CASE band-centred patches from a single case."""

    def __init__(self, crops_dir: str, case_ids: list[str], patch_size: int = PATCH_SIZE,
                 patches_per_case: int = PATCHES_PER_CASE, augment: bool = False,
                 seed: int = None):
        self.crops_dir        = crops_dir
        self.case_ids         = case_ids
        self.patch_size        = patch_size
        self.patches_per_case  = patches_per_case
        self.augment           = augment
        # Persistent stream: each __getitem__ draws the next seed, so patch sampling
        # varies epoch to epoch under a fixed base seed (not frozen to one fixed set).
        self._seed_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        data = np.load(os.path.join(self.crops_dir, f"{cid}.npz"))

        arrays = pad_to_patch({
            "img":       data["img"].astype(np.float32),
            "sdf_prior": data["sdf_prior"].astype(np.float32),
            "band":      data["band"].astype(np.float32),
            "mask":      data["mask"].astype(np.float32),
        }, self.patch_size)
        img, prior, band, mask = arrays["img"], arrays["sdf_prior"], arrays["band"], arrays["mask"]

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        origins = band_patch_origins(band, self.patches_per_case, self.patch_size, rng)

        xs, ys, bs = [], [], []
        for origin in origins:
            sl = tuple(slice(o, o + self.patch_size) for o in origin)
            img_p, prior_p, band_p, mask_p = img[sl], prior[sl], band[sl], mask[sl]

            if self.augment:
                img_p, prior_p, mask_p, band_p = _augment(img_p, prior_p, mask_p, band_p, rng)

            xs.append(np.stack([img_p, prior_p], axis=0))
            ys.append(mask_p[np.newaxis])
            bs.append(band_p[np.newaxis])

        x = torch.from_numpy(np.stack(xs).astype(np.float32))  # (K, 2, P, P, P)
        y = torch.from_numpy(np.stack(ys).astype(np.float32))  # (K, 1, P, P, P)
        # The band travels with the patch so the loss can be restricted to it. The
        # network still sees the whole patch as input — it needs the surrounding
        # context to decide — it is only scored where refinement is allowed to act.
        b = torch.from_numpy(np.stack(bs).astype(np.float32))  # (K, 1, P, P, P)
        return x, y, b


def collate_patches(items):
    return (torch.cat([x for x, _, _ in items], dim=0),
           torch.cat([y for _, y, _ in items], dim=0),
           torch.cat([b for _, _, b in items], dim=0))
