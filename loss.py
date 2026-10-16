# The loss: plain L1 on the normalised SDF, plus 0.5 x the surface-weighted L1 below
# (train.py adds the two).

import torch
import torch.nn.functional as F


# L1 weighted by exp(-|true distance| / sigma), so errors near the surface count most.
#
# Only where the field crosses zero matters for the segmentation, but most voxels sit far
# from the surface, where the target saturates at +-1; with plain L1 they dominate the
# gradient while saying nothing about where the surface is. This weighting was the largest
# single gain: +0.0030 Dice over plain L1 on all five folds (coefficient 0.5, sigma 1.5 mm).
#
# sigma is in mm: a voxel on the surface weighs 1.0, one 1.5 mm away 0.37, one 5 mm away
# 0.04. The target is stored normalised to [-1, 1] over trunc_mm, hence the conversion.
def surface_weighted_l1(pred: torch.Tensor, target: torch.Tensor,
                        sigma_mm: float = 1.5, trunc_mm: float = 10.0) -> torch.Tensor:
    distance_mm = target.abs() * trunc_mm
    weight = torch.exp(-distance_mm / sigma_mm)
    # Weighted mean of the absolute error (clamped so an empty weight cannot divide by 0).
    return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(1e-6)


# Dice + BCE on soft masks of the SDF: sigmoid(-d / tau), d in mm. The mask goes from
# inside (1) to outside (0) over about +-2 tau around the surface, so the loss acts on the
# sign of the field exactly where Dice is decided, and trains on the metric we are judged
# by (nnU-Net's own Dice + cross-entropy). The target is the same soft mask of the true SDF,
# so it keeps sub-voxel precision instead of a blocky mask on the coarse grid.
def soft_mask_dice_bce(pred: torch.Tensor, target: torch.Tensor, tau_mm: float = 0.5,
                       trunc_mm: float = 10.0, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(-pred * trunc_mm / tau_mm)
    t = torch.sigmoid(-target * trunc_mm / tau_mm)
    dims = tuple(range(1, p.ndim))                      # soft Dice per sample, then averaged
    inter = (p * t).sum(dims)
    dice = (2 * inter + eps) / (p.sum(dims) + t.sum(dims) + eps)
    return (1 - dice).mean() + F.binary_cross_entropy(p, t)


# The Gaussian contour map of a normalised SDF: exp(-d^2 / 2 sigma^2), with d in mm.
# 1 on the surface, 0.61 at sigma, ~0 a few sigma away (Torres et al., EMBC 2023, define
# their contour target as such a Gaussian-like curve around the object's contour). For an
# SDF it is an exact, differentiable function, so it gives both the contour target (from
# the true SDF) and the contour implied by the predicted SDF (for the consistency loss).
def gaussian_contour(sdf: torch.Tensor, sigma_mm: float = 1.5, trunc_mm: float = 10.0) -> torch.Tensor:
    d_mm = sdf * trunc_mm
    return torch.exp(-d_mm ** 2 / (2 * sigma_mm ** 2))


# Dice + BCE on a mask output, nnU-Net's loss for a binary problem (its DC_and_BCE): the
# network's logits against the head mask, which is the sign of the SDF target (inside where
# it is negative). Soft Dice per sample on sigmoid(logits), averaged, plus BCE on the
# logits. Used by the resolution study's mask arm, against which the SDF regression was
# compared (train.py no longer has that option).
def mask_dice_bce(logits: torch.Tensor, sdf_target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    target = (sdf_target < 0).float()
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    dice = (2 * inter + eps) / (p.sum(dims) + target.sum(dims) + eps)
    return (1 - dice).mean() + F.binary_cross_entropy_with_logits(logits, target)
