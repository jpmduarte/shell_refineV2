# The baseline's training data (size_normalization_approach/dataset.py, unchanged) plus
# the polar targets of v2: the head's centre and its radii from there, computed on the
# augmented SDF so a rotated or scaled head has the right ones.
#
# Items: (x, y, centre (3,), radii (D,) in voxels, weight).

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "size_normalization_approach")]
from dataset import SDFDataset  # noqa: E402
from make_sdf import TRUNC_MM  # noqa: E402

import polar  # noqa: E402


class PolarDataset(SDFDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirs = polar.directions()

    def __getitem__(self, idx: int):
        x, y = super().__getitem__(idx)
        # The grid's voxel size (augmentation scales the content, never the grid).
        with np.load(os.path.join(self.cache_dir, f"x{self.factor}", f"{self.case_ids[idx]}.npz")) as d:
            spacing = float(np.abs(d["affine"][0, 0]))
        c, r, w = polar.centre_targets(y[0].numpy(), spacing, TRUNC_MM, self._rng, self.dirs)
        return x, y, torch.from_numpy(c), torch.from_numpy(r), torch.tensor(w, dtype=torch.float32)
