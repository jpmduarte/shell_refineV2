"""Training data: each case's x4 image and SDF target, padded to the 96^3 cube (never resized or cropped)."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common.augmentation import SCALE_P, SCALE_RANGE, augment_nnunet, pad_to_cube
from common.geometry import CUBE, FACTOR
from common.image_cache import load as load_image
from common.sdf_targets import load as load_sdf, load_mm as load_sdf_mm

IMAGE_PAD = 0.0   # the z-scored mean, not the background (-0.6 to -0.8); inference pads alike, so keep it

SDF_PAD = 1.0   # outside: 0 would put a fake surface along the padding


def add_channel_axis(volume: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(volume[np.newaxis].astype(np.float32))


class SDFDataset(Dataset):
    """(image, SDF) pairs of shape (1, 96, 96, 96) from <cache>/x4 and <cache>/sdf/x4 (sdf/x4mm when augmenting)."""

    def __init__(self, cache_dir: str, case_ids: list[str], augment: bool = False, seed: int = None,
                 scale_p: float = SCALE_P, scale_range: tuple = SCALE_RANGE):
        self.cache_dir = cache_dir
        self.case_ids = case_ids
        self.augment = augment
        self.scale_p, self.scale_range = scale_p, tuple(scale_range)
        mm_dir = os.path.join(cache_dir, "sdf", f"x{FACTOR}mm")
        if augment and not os.path.isdir(mm_dir):
            raise ValueError(f"augmentation needs the untruncated field {mm_dir} "
                             f"(common/sdf_targets.py --untruncated, or size_normalization/build_cache.py)")
        self._rng = np.random.default_rng(seed)   # replaced per loader worker by seeding.seed_worker

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        img, _ = load_image(self.cache_dir, cid, FACTOR)
        img = img.astype(np.float32)

        if self.augment:
            sdf_mm, _ = load_sdf_mm(self.cache_dir, cid, FACTOR)
            img, sdf = augment_nnunet(img, sdf_mm.astype(np.float32), self._rng, CUBE,
                                      scale_p=self.scale_p, scale_range=self.scale_range)
        else:
            sdf, _ = load_sdf(self.cache_dir, cid, FACTOR)
            img = pad_to_cube(img, CUBE, cval=IMAGE_PAD)
            sdf = pad_to_cube(sdf.astype(np.float32), CUBE, cval=SDF_PAD)

        return add_channel_axis(img), add_channel_axis(sdf)
