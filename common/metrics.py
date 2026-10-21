"""Metrics on the SDF (training: mae, dice_from_sdf, boundary_dice_from_sdf) and on masks (dice, hd95, asd)."""
import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance

from common.geometry import add_batch_axes
from common.sdf_targets import TRUNC_MM

BOUNDARY_BAND_MM = 5.0


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def dice_from_sdf(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    """Dice between the regions where each SDF is negative."""
    return dice((pred < 0).float(), (target < 0).float())


def boundary_dice_from_sdf(pred: torch.Tensor, target: torch.Tensor,
                           band_mm: float = BOUNDARY_BAND_MM, trunc_mm: float = TRUNC_MM) -> float | None:
    """Dice within band_mm of the true surface, which whole-volume Dice dilutes with the easy interior."""
    band = (target.abs() < (band_mm / trunc_mm))
    if band.sum() == 0:
        return None
    return dice((pred < 0).float()[band], (target < 0).float()[band])


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float | None:
    """Dice of two soft masks thresholded; None when both are empty (0/0 is undefined, not 0)."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    total = pred_bin.sum().item() + target_bin.sum().item()

    if total == 0:
        return None
    return (2.0 * intersection) / total


def mask_dice(a, b, if_both_empty: float = float("nan")) -> float:
    """Dice of two boolean masks, numpy or torch."""
    total = a.sum() + b.sum()
    if total == 0:
        return if_both_empty
    return float(2 * (a & b).sum() / total)


def _as_batch(mask: np.ndarray) -> torch.Tensor:
    return add_batch_axes(torch.from_numpy(mask.astype(np.float32)))


def hd95(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    """95th-percentile Hausdorff distance in mm; None if either mask is empty (MONAI would give inf or nan)."""
    if not pred.any() or not target.any():
        return None
    distance = compute_hausdorff_distance(_as_batch(pred), _as_batch(target),
                                          include_background=True, percentile=95, spacing=spacing)
    return float(distance.item())


def asd(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    """Average symmetric surface distance in mm; None if either mask is empty."""
    if not pred.any() or not target.any():
        return None
    distance = compute_average_surface_distance(_as_batch(pred), _as_batch(target),
                                                include_background=True, symmetric=True, spacing=spacing)
    return float(distance.item())
