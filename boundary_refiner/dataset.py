"""Refiner training data: per case, `patches_per_case` cubes centred on random band voxels of its crop.

    x     (P, 2, c, c, c)   the image and the first pass's prior (its SDF)
    y     (P, 1, c, c, c)   the true mask
    band  (P, 1, c, c, c)   where the refiner is scored (|prior| < BAND_MM)

Training cubes are drawn anew every epoch and get nnU-Net's augmentation, in its order: rotation and
scaling of the cube (image and both SDFs alike, the SDFs divided by the zoom and re-truncated, mask and
band re-derived from them), intensity on the image only, then mirroring of everything. Validation cubes
are the same every epoch (seeded per case) and not augmented.
"""
import os
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

from common.augmentation import MIRROR_P, intensity_transform, spatial_patch
from common.sdf_targets import TRUNC_MM

CUBE = 64
PATCHES_PER_CASE = 12
BAND_MM = 5.0   # the band's half-width, as make_crops.py builds it
CROP_ARRAYS = ("img", "sdf_prior", "sdf_true", "band", "mask")
PAD_VALUES = {"img": 0.0, "sdf_prior": 1.0, "sdf_true": 1.0, "band": 0.0, "mask": 0.0}   # SDFs: far outside


def pad_to_shape(arrays: dict, minimum: int) -> dict:
    """Pad every axis up to at least `minimum`, each array with its own fill value."""
    shape = next(iter(arrays.values())).shape
    pads = [(0, max(0, minimum - s)) for s in shape]
    if all(p == (0, 0) for p in pads):
        return arrays
    return {name: np.pad(a, pads, mode="constant", constant_values=PAD_VALUES[name])
            for name, a in arrays.items()}


def load_crop(path: str, cube: int) -> dict:
    with np.load(path) as crop:
        arrays = {k: crop[k].astype(np.float32) for k in CROP_ARRAYS}
    return pad_to_shape(arrays, cube)


def cube_slices(origin, cube) -> tuple:
    return tuple(slice(o, o + cube) for o in origin)


def random_origin(band: np.ndarray, cube: int, rng: np.random.Generator) -> tuple:
    """The corner of a cube centred on a random band voxel, kept inside the volume."""
    band_voxels = np.flatnonzero(band)
    centre = np.unravel_index(band_voxels[rng.integers(len(band_voxels))], band.shape)
    return tuple(int(np.clip(c - cube // 2, 0, s - cube)) for c, s in zip(centre, band.shape))


def augmented_cube(crop: dict, origin: tuple, cube: int, rng: np.random.Generator):
    """(image, prior, mask, band) of one training cube."""
    (img, prior, sdf_true), scale = spatial_patch(
        [crop["img"], crop["sdf_prior"], crop["sdf_true"]],
        [PAD_VALUES["img"], PAD_VALUES["sdf_prior"], PAD_VALUES["sdf_true"]], origin, rng, cube)
    if scale != 1.0:
        prior = np.clip(prior / scale, -1.0, 1.0)
        sdf_true = np.clip(sdf_true / scale, -1.0, 1.0)
    mask = (sdf_true < 0).astype(np.float32)
    band = (np.abs(prior) * TRUNC_MM < BAND_MM).astype(np.float32)

    img = intensity_transform(img, rng)

    for axis in range(3):
        if rng.random() < MIRROR_P:
            img, prior = np.flip(img, axis), np.flip(prior, axis)
            mask, band = np.flip(mask, axis), np.flip(band, axis)
    return img, prior, mask, band


def stacked(arrays) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrays), dtype=np.float32))


class CubeDataset(Dataset):

    def __init__(self, crops_dir: str, case_ids: list[str], augment: bool = False,
                 cube: int = CUBE, patches_per_case: int = PATCHES_PER_CASE, seed: int = None):
        self.paths = [os.path.join(crops_dir, f"{c}.npz") for c in case_ids]
        self.case_ids = case_ids
        self.augment = augment
        self.cube = cube
        self.patches_per_case = patches_per_case
        self.seed = seed or 0
        self._rng = np.random.default_rng(seed)   # replaced per loader worker by seeding.seed_worker

    def __len__(self) -> int:
        return len(self.paths)

    def _rng_for(self, idx):
        """Training draws from the running generator; validation from one seeded by the case id."""
        if self.augment:
            return self._rng
        return np.random.default_rng([self.seed, zlib.crc32(self.case_ids[idx].encode())])

    def __getitem__(self, idx: int):
        crop = load_crop(self.paths[idx], self.cube)
        rng = self._rng_for(idx)

        inputs, masks, bands = [], [], []
        for _ in range(self.patches_per_case):
            origin = random_origin(crop["band"], self.cube, rng)
            if self.augment:
                img, prior, mask, band = augmented_cube(crop, origin, self.cube, rng)
            else:
                patch = cube_slices(origin, self.cube)
                img, prior, mask, band = (crop[k][patch] for k in ("img", "sdf_prior", "mask", "band"))

            inputs.append(np.stack([img, prior]))
            masks.append(mask[None])
            bands.append(band[None])

        return stacked(inputs), stacked(masks), stacked(bands)


def collate_cases(items):
    """Batch several cases by concatenating their cubes: (cases x P, ...)."""
    inputs, masks, bands = zip(*items)
    return torch.cat(inputs), torch.cat(masks), torch.cat(bands)
