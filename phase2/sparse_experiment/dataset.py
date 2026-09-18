"""Phase 2 dataset: band voxels only, native resolution, no crop/tiling.

For each case: take the SDF prior (phase 1's prediction once available; the true
native SDF for now, so the sparse mechanism can be validated independent of phase 1's
training state), extract {|sdf| < band_mm} as the active set, and pair each active
voxel with (image, sdf_prior) features and the real label as target.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from cache import load as load_image, load_label  # noqa: E402
from make_sdf import load as load_sdf  # noqa: E402

BAND_MM  = 5.0
TRUNC_MM = 10.0


def load_case_band(cache_dir: str, cid: str, band_mm: float = BAND_MM,
                   trunc_mm: float = TRUNC_MM, sdf_prior: np.ndarray = None):
    """Returns (coords[N,3] int, features[N,2] float, target[N] float) for one case's
    band. sdf_prior overrides the true SDF (pass phase 1's prediction once available)."""
    img, _ = load_image(cache_dir, cid, 1)
    sdf, _ = load_sdf(cache_dir, cid, 1)
    sdf = sdf[0] if sdf.ndim == 4 else sdf
    if sdf_prior is None:
        sdf_prior = sdf

    label, _ = load_label(cache_dir, cid)
    label = label[0] if label.ndim == 4 else label

    band = np.abs(sdf_prior) < (band_mm / trunc_mm)
    coords = np.argwhere(band).astype(np.int64)

    features = np.stack([img[band], sdf_prior[band]], axis=1).astype(np.float32)
    target = label[band].astype(np.float32)

    return coords, features, target


class Phase2Dataset(Dataset):
    """One case per item — coords/features/target vary in size, so batching concats
    them with a batch column rather than torch.stack (see collate below)."""

    def __init__(self, cache_dir: str, case_ids: list[str], band_mm: float = BAND_MM):
        self.cache_dir = cache_dir
        self.case_ids  = case_ids
        self.band_mm   = band_mm

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        coords, features, target = load_case_band(self.cache_dir, cid, self.band_mm)
        return torch.from_numpy(coords), torch.from_numpy(features), torch.from_numpy(target)


def collate(batch):
    """Concatenate variable-sized per-case sparse tensors into one batch, prefixing
    coords with a batch-index column — the convention build_neighbor_index expects."""
    all_coords, all_features, all_targets = [], [], []
    for b, (coords, features, target) in enumerate(batch):
        batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int64)
        all_coords.append(torch.cat([batch_col, coords], dim=1))
        all_features.append(features)
        all_targets.append(target)

    return (torch.cat(all_coords, dim=0),
            torch.cat(all_features, dim=0),
            torch.cat(all_targets, dim=0))
