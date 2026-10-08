# Phase 2 training data: cubes cut from the band crops (make_crops.py).
#
# Each item is one case, as `patches_per_case` cubes of side `cube` centred on random band
# voxels, stacked on the batch axis:
#     x     (P, 2, c, c, c)   channel 0: the image; channel 1: phase 1's prior (its SDF)
#     y     (P, 1, c, c, c)   the true mask
#     band  (P, 1, c, c, c)   where the refiner is scored (|prior| < band_mm)
# The refiner sees the whole cube as context but is scored only inside the band, where it
# is allowed to change the answer (train.py).
#
# Training cubes are drawn anew every epoch and get the nnU-Net's augmentation
# (../../nnunet_augmentation.py), in its order:
#   1. rotation and scaling of the cube about its centre (spatial_patch), applied alike to
#      the image, the prior SDF and the true SDF. Both SDFs are divided by the zoom, since
#      distances scale with the image, and re-truncated. The target mask is re-derived as
#      true SDF < 0 (exact, rather than an interpolated mask), and the band is recomputed
#      from the transformed prior, so it still means "near phase 1's surface".
#   2. the intensity transforms, on the image only (the prior is not an image).
#   3. mirroring, of everything together.
# The prior is a network output already saturated at the truncation, so a shrinking zoom
# makes its values beyond ~7 mm slightly too small; inside the 5 mm band it is exact.
#
# Validation cubes are the same every epoch (seeded per case) and not augmented, so the
# logged curve compares like with like.

import os
import sys
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from nnunet_augmentation import intensity_transform, spatial_patch  # noqa: E402

CUBE = 64
PATCHES_PER_CASE = 12
BAND_MM = 5.0      # the band's half-width, as make_crops.py builds it
TRUNC_MM = 10.0    # the SDFs' truncation

# Padding values per array: 0 for the image and the masks; +1 ("far outside") for the
# SDFs, since 0 would mean "on the surface".
PAD_VALUES = {"img": 0.0, "sdf_prior": 1.0, "sdf_true": 1.0, "band": 0.0, "mask": 0.0}


# Pad every axis up to at least `minimum`, each array with its own fill value.
def pad_to_shape(arrays: dict, minimum: int) -> dict:
    shape = next(iter(arrays.values())).shape
    pads = [(0, max(0, minimum - s)) for s in shape]
    if all(p == (0, 0) for p in pads):
        return arrays
    return {name: np.pad(a, pads, mode="constant", constant_values=PAD_VALUES[name])
            for name, a in arrays.items()}


# Read the arrays a cube needs from one crop file, padded to at least one cube.
def load_crop(path: str, cube: int) -> dict:
    with np.load(path) as z:
        arrays = {k: z[k].astype(np.float32)
                  for k in ("img", "sdf_prior", "sdf_true", "band", "mask")}
    return pad_to_shape(arrays, cube)


# The corner of a cube centred on a random band voxel, kept inside the volume.
def random_origin(band: np.ndarray, cube: int, rng: np.random.Generator) -> tuple:
    band_voxels = np.flatnonzero(band)
    centre = np.unravel_index(band_voxels[rng.integers(len(band_voxels))], band.shape)
    return tuple(int(np.clip(c - cube // 2, 0, s - cube)) for c, s in zip(centre, band.shape))


# One training cube with the full augmentation: (image, prior, mask, band).
def augmented_cube(a: dict, origin: tuple, cube: int, rng: np.random.Generator):
    # 1. Rotation and scaling, the same for the image and both SDFs.
    (img, prior, sdf_true), s = spatial_patch(
        [a["img"], a["sdf_prior"], a["sdf_true"]], [0.0, 1.0, 1.0], origin, rng, cube)
    if s != 1.0:
        prior = np.clip(prior / s, -1.0, 1.0)
        sdf_true = np.clip(sdf_true / s, -1.0, 1.0)
    mask = (sdf_true < 0).astype(np.float32)
    band = (np.abs(prior) * TRUNC_MM < BAND_MM).astype(np.float32)

    # 2. Intensity, on the image only.
    img = intensity_transform(img, rng)

    # 3. Mirroring: the same flips for everything, so it stays aligned.
    for axis in range(3):
        if rng.random() < 0.5:
            img, prior = np.flip(img, axis), np.flip(prior, axis)
            mask, band = np.flip(mask, axis), np.flip(band, axis)
    return img, prior, mask, band


class CubeDataset(Dataset):

    def __init__(self, crops_dir: str, case_ids: list[str], augment: bool = False,
                 cube: int = CUBE, patches_per_case: int = PATCHES_PER_CASE, seed: int = None):
        self.paths = [os.path.join(crops_dir, f"{c}.npz") for c in case_ids]
        self.case_ids = case_ids
        self.augment = augment
        self.cube = cube
        self.patches_per_case = patches_per_case
        self.seed = seed or 0
        # Random generator for training cubes and augmentation; train_cube.py's
        # seed_worker gives each loader worker its own.
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        a = load_crop(self.paths[idx], self.cube)
        c = self.cube

        # Validation: the same cubes every epoch, from a generator seeded by the case id.
        if self.augment:
            rng = self._rng
        else:
            rng = np.random.default_rng([self.seed, zlib.crc32(self.case_ids[idx].encode())])

        xs, ys, bs = [], [], []
        for _ in range(self.patches_per_case):
            o = random_origin(a["band"], c, rng)
            if self.augment:
                img, prior, mask, band = augmented_cube(a, o, c, rng)
            else:
                sl = tuple(slice(oi, oi + c) for oi in o)
                img, prior, mask, band = a["img"][sl], a["sdf_prior"][sl], a["mask"][sl], a["band"][sl]

            xs.append(np.stack([img, prior]))
            ys.append(mask[None])
            bs.append(band[None])

        x = torch.from_numpy(np.ascontiguousarray(np.stack(xs), dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(np.stack(ys), dtype=np.float32))
        b = torch.from_numpy(np.ascontiguousarray(np.stack(bs), dtype=np.float32))
        return x, y, b


# Batch several cases by concatenating their cubes: (cases x P, ...).
def collate_cases(items):
    xs, ys, bs = zip(*items)
    return torch.cat(xs), torch.cat(ys), torch.cat(bs)
