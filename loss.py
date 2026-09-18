"""Phase 1 losses: SDF regression in [-1, 1]."""

import torch
import torch.nn.functional as F


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def gradient_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum over the 3 spatial axes of the mean L1 distance between first-order finite
    differences of pred and target. Penalises mismatched local slope, largest exactly
    where the SDF changes fastest (near the surface). Matching gradients everywhere
    while the whole field is offset by a constant scores zero here, so this only makes
    sense added on top of a value-matching loss like l1, never used alone.
    """
    total = pred.new_zeros(())
    for axis in (2, 3, 4):  # (N, C, D, H, W)
        total = total + torch.abs(pred.diff(dim=axis) - target.diff(dim=axis)).mean()
    return total


LOSSES = {
    "l1":  l1,
    "mse": mse,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES)}")
    return LOSSES[name]
