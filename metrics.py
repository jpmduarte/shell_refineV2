# Evaluation metrics.
#
# On the SDF itself (used during training):  mae, dice_from_sdf, boundary_dice_from_sdf
# On binary masks at native resolution:      dice, hd95, asd (surface distances in mm)

import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance


# Mean absolute error between two fields.
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


# Dice between the regions where each SDF is negative (inside the head).
def dice_from_sdf(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    return dice((pred < 0).float(), (target < 0).float())


# Dice counted only within band_mm of the true surface. Whole-volume Dice is diluted by
# the large, easy interior and exterior; this follows the boundary alone. Training logs it.
def boundary_dice_from_sdf(pred: torch.Tensor, target: torch.Tensor,
                           band_mm: float = 5.0, trunc_mm: float = 10.0) -> float | None:
    band = (target.abs() < (band_mm / trunc_mm))
    if band.sum() == 0:
        return None
    return dice((pred < 0).float()[band], (target < 0).float()[band])


# Dice between two masks (values >= threshold count as inside). None when both are empty:
# 0/0 is undefined, not a score of 0.
def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float | None:
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    total        = pred_bin.sum().item() + target_bin.sum().item()

    if total == 0:
        return None
    return (2.0 * intersection) / total


# MONAI's metrics expect a (batch, channel, ...) tensor.
def _as_batch(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32))[None, None]


# 95th-percentile Hausdorff distance in mm. None if either mask is empty (MONAI would
# return inf or nan, which would silently corrupt an average).
def hd95(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    if not pred.any() or not target.any():
        return None
    d = compute_hausdorff_distance(_as_batch(pred), _as_batch(target),
                                    include_background=True, percentile=95, spacing=spacing)
    return float(d.item())


# Average symmetric surface distance in mm. None if either mask is empty.
def asd(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    if not pred.any() or not target.any():
        return None
    d = compute_average_surface_distance(_as_batch(pred), _as_batch(target),
                                          include_background=True, symmetric=True, spacing=spacing)
    return float(d.item())
