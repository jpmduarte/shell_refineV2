# Training data: each case's x4 image and its SDF target, padded to a 96^3 cube.
#
# Each case keeps its own isotropic grid (its shape varies with the scan and, after size
# normalisation, with the head), so cases are padded to the cube, never resized or
# cropped; size_norm.py guarantees they fit.
#
# Training cases go through the nnU-Net's augmentation, adapted to an SDF target
# (../nnunet_augmentation.py).

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import load as load_image  # noqa: E402
from geometry import CUBE, FACTOR  # noqa: E402
from make_sdf import load as load_sdf, load_mm as load_sdf_mm  # noqa: E402
from nnunet_augmentation import augment_nnunet, pad_to_cube  # noqa: E402


# (image, SDF) pairs from cache_dir/x4 and cache_dir/sdf/x4 (or sdf/x4mm when augmenting),
# each of shape (1, 96, 96, 96). `factor` and `cube` pick another grid (the resolution
# study: x8 in 48^3, x2 in 192^3).
class SDFDataset(Dataset):

    def __init__(self, cache_dir: str, case_ids: list[str], augment: bool = False,
                 seed: int = None, factor: int = FACTOR, cube: int = CUBE,
                 scale_p: float = 0.2, scale_range: tuple = (0.7, 1.4)):
        self.cache_dir = cache_dir
        self.case_ids  = case_ids
        self.augment   = augment
        self.factor    = factor
        self.cube      = cube
        # The augmentation's scaling: probability and range (nnU-Net's by default).
        self.scale_p, self.scale_range = scale_p, tuple(scale_range)
        mm_dir = os.path.join(cache_dir, "sdf", f"x{factor}mm")
        if augment and not os.path.isdir(mm_dir):
            raise ValueError(f"augmentation needs the untruncated field {mm_dir} "
                             f"(make_sdf.py --untruncated, or size_norm.py)")
        # The random generator for augmentation. With several loader workers, train.py
        # (seed_worker) gives each worker its own.
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        img, _ = load_image(self.cache_dir, cid, self.factor)
        img = img.astype(np.float32)

        if self.augment:
            sdf_mm, _ = load_sdf_mm(self.cache_dir, cid, self.factor)
            img, sdf = augment_nnunet(img, sdf_mm.astype(np.float32), self._rng, self.cube,
                                      scale_p=self.scale_p, scale_range=self.scale_range)
        else:
            sdf, _ = load_sdf(self.cache_dir, cid, self.factor)
            # Pad the image with 0 and the SDF with +1 (outside; 0 would put a fake surface
            # along the padding). 0 is the z-scored volume's mean, not its background: about
            # half of each x4 volume is background, at -0.6 to -0.8, so the padding (like
            # the rotation fill in nnunet_augmentation) makes a step edge. Training and
            # inference (infer.py) both pad with 0, so they agree; changing it would mean
            # retraining.
            img = pad_to_cube(img, self.cube, cval=0.0)
            sdf = pad_to_cube(sdf.astype(np.float32), self.cube, cval=1.0)

        # Add the channel axis: (1, D, H, W).
        x = img[np.newaxis].astype(np.float32)
        y = sdf[np.newaxis].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
