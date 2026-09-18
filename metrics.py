"""Phase 1 metrics: MAE on the normalised SDF, Dice on its zero level set, plus
surface-distance metrics (native resolution, binary masks, mm units)."""

import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def dice_from_sdf(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    # Signed distance: the interior is the negative side, not values above a threshold.
    return dice((pred < 0).float(), (target < 0).float())


def boundary_dice_from_sdf(pred: torch.Tensor, target: torch.Tensor,
                           band_mm: float = 5.0, trunc_mm: float = 10.0) -> float | None:
    """Dice restricted to voxels within band_mm of the TRUE boundary (|target| < band_mm
    normalised). Whole-volume dice_from_sdf is diluted by the mostly-correct far field —
    a model can raise it without moving the boundary at all. This measures only the part
    that actually determines the crop margin and phase 2's input.
    """
    band = (target.abs() < (band_mm / trunc_mm))
    if band.sum() == 0:
        return None
    return dice((pred < 0).float()[band], (target < 0).float()[band])


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float | None:
    """None when both masks are empty — union == 0 is an undefined ratio, not a 0."""
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    union        = pred_bin.sum().item() + target_bin.sum().item()

    if union == 0:
        return None
    return (2.0 * intersection) / union


def _as_batch(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32))[None, None]


def hd95(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    """95th-percentile Hausdorff distance in mm. None if either mask is empty —
    MONAI returns nan/inf there, which would silently poison a mean() downstream."""
    if not pred.any() or not target.any():
        return None
    d = compute_hausdorff_distance(_as_batch(pred), _as_batch(target),
                                    include_background=True, percentile=95, spacing=spacing)
    return float(d.item())


def asd(pred: np.ndarray, target: np.ndarray, spacing) -> float | None:
    """Average symmetric surface distance in mm. None if either mask is empty."""
    if not pred.any() or not target.any():
        return None
    d = compute_average_surface_distance(_as_batch(pred), _as_batch(target),
                                          include_background=True, symmetric=True, spacing=spacing)
    return float(d.item())
