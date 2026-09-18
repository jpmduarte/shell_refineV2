"""Phase 1 dataset: x8 image + x8 SDF target, padded to a fixed cube.

x8 shapes vary per case (native-proportional, isotropic mm/voxel — see cache.py); the
largest is 45x46x31, so every case fits in a 48^3 cube with only pad, never crop.
Padding keeps the cache's isotropic grid exactly as built — no second resample, so no
new anisotropy on top of the one cache.py already removed.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import load as load_image  # noqa: E402
from make_sdf import load as load_sdf  # noqa: E402

CUBE = 48


def pad_to_cube(vol: np.ndarray, cube: int, cval: float) -> np.ndarray:
    pads = [(0, cube - s) for s in vol.shape]
    if any(p[1] < 0 for p in pads):
        raise ValueError(f"shape {vol.shape} exceeds cube={cube}")
    return np.pad(vol, pads, mode="constant", constant_values=cval)


def _augment(img: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    for axis in range(3):
        if rng.random() < 0.5:
            img    = np.flip(img,    axis)
            target = np.flip(target, axis)

    axes = [(0, 1), (0, 2), (1, 2)][rng.integers(0, 3)]
    k    = rng.integers(0, 4)
    img    = np.rot90(img,    k, axes)
    target = np.rot90(target, k, axes)

    img = img * rng.uniform(0.9, 1.1)                    # random intensity scale
    img = img + rng.normal(0.0, 0.05, size=img.shape)    # additive noise

    # np.flip/rot90 return negative-stride views — make contiguous for torch
    img    = np.ascontiguousarray(img)
    target = np.ascontiguousarray(target)
    return img, target


class Phase1Dataset(Dataset):
    """cache_dir/x8/<cid>.npz + cache_dir/sdf/x8/<cid>.npz -> (image, sdf) at a fixed cube."""

    def __init__(self, cache_dir: str, case_ids: list[str], augment: bool = False,
                 cube: int = CUBE, seed: int = None):
        self.cache_dir = cache_dir
        self.case_ids  = case_ids
        self.augment   = augment
        self.cube      = cube
        # None (default): unseeded. With num_workers>0, seed_worker (train.py) reseeds
        # each forked copy independently — see train.py for why that step is needed.
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        img, _ = load_image(self.cache_dir, cid, 8)
        sdf, _ = load_sdf(self.cache_dir, cid, 8)
        img, sdf = img.astype(np.float32), sdf.astype(np.float32)

        if self.augment:
            img, sdf = _augment(img, sdf, self._rng)

        # Image padded with 0 (its own normalised background); SDF padded with +1
        # (outside), never 0, which would invent a surface at the pad boundary.
        img = pad_to_cube(img, self.cube, cval=0.0)
        sdf = pad_to_cube(sdf, self.cube, cval=1.0)

        x = img[np.newaxis].astype(np.float32)  # (1, D, H, W)
        y = sdf[np.newaxis].astype(np.float32)  # (1, D, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)
