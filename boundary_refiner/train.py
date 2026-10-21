"""Phase 2's loss and epoch loops. train_slab.py is the entry point that drives them.

They live apart from it because they assume nothing about patch shape — they only ever
see (x, y, band) tensors — so the loss and the patch geometry stay independent of each
other. The band travels with every patch: the network sees the whole patch as input,
because it needs the surrounding context to decide, but is scored only where refinement
is allowed to act.
"""

import os

import torch


from common.metrics import dice  # noqa: E402


def dice_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
             eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice + BCE on the sigmoid output (v1's best phase 2 loss).

    With `weight` (the band), every voxel outside it contributes exactly zero. A 64^3
    cube around a ~25-voxel-thick shell is mostly deep interior and exterior, which the
    prior already gets right — scoring them dilutes the gradient with voxels the
    refiner is not allowed to change anyway. v1 tested the weak form of this
    (down-weighting non-band voxels to a floor) and measured no benefit; masking them
    out entirely is the untested strong form.
    """
    bce_map = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")

    if weight is None:
        inter = (pred * target).sum()
        soft_dice = 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)
        return soft_dice + bce_map.mean()

    # The Dice term keeps the binary band even when `weight` is graded. Dice is an overlap
    # ratio, and a continuous weight in its denominator changes what fraction of what is
    # being measured — the term would no longer be the band Dice that checkpoint selection
    # and evaluate.py report. The per-voxel BCE has no such constraint, so that is where
    # grading buys resolution.
    hard = (weight > 0).to(weight.dtype)
    inter = (pred * target * hard).sum()
    denom = (pred * hard).sum() + (target * hard).sum()
    soft_dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    bce = (bce_map * weight).sum() / weight.sum().clamp_min(1e-6)
    return soft_dice + bce


def surface_weight(x: torch.Tensor, band: torch.Tensor, sigma_mm: float,
                   trunc_mm: float = 10.0) -> torch.Tensor:
    """The band mask graded by distance to the predicted boundary.

    The binary band treats a voxel on the zero crossing exactly like one 5 mm out, but the
    two are not equally decidable: near the crossing the image is genuinely ambiguous and
    the refiner's answer is what moves the surface, while at 5 mm the prior is almost
    always already right. Grading the weight by exp(-|sdf|/sigma) spends the gradient
    where the decision is.

    This is the phase 2 analogue of surface_weighted_l1, which is the largest single gain
    phase 1 has (+0.0030 over plain L1 on all five folds). The progression it continues is
    already measured: scoring the whole patch uniformly was +0.0010, restricting to the
    binary band was +0.0021.

    Read off channel 1 of the input, which is the prior the crops were built from — no
    extra tensor has to be plumbed through the loader. Multiplied by the band rather than
    replacing it, so voxels the refiner may not rewrite still contribute exactly zero.
    """
    prior = x[:, 1:2]
    return band * torch.exp(-prior.abs() * trunc_mm / sigma_mm)


def train_epoch(model, loader, optimizer, device, band_masked: bool,
                sw_sigma: float = 0.0) -> float:
    model.train()
    total, seen = 0.0, 0
    for x, y, b in loader:
        x, y, b = x.to(device), y.to(device), b.to(device)
        pred = model(x)
        if band_masked and sw_sigma > 0:
            w = surface_weight(x, b, sw_sigma)
        else:
            w = b if band_masked else None
        loss = dice_bce(pred, y, weight=w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        seen  += x.size(0)
    return total / seen


@torch.no_grad()
def val_epoch(model, loader, device, band_masked: bool) -> tuple[float, float]:
    model.eval()
    total, seen = 0.0, 0
    dice_scores = []
    for x, y, b in loader:
        x, y, b = x.to(device), y.to(device), b.to(device)
        pred = model(x)
        total += dice_bce(pred, y, weight=b if band_masked else None).item() * x.size(0)
        seen  += x.size(0)
        # Dice restricted to the band, matching what the refiner is scored on and what
        # evaluate.py reports as "band" — a whole-patch Dice would be dominated by the
        # easy interior the prior already had right.
        for p, t, bb in zip(pred, y, b):
            sel = bb > 0.5
            if sel.any():
                score = dice(p[sel], t[sel])
                if score is not None:
                    dice_scores.append(score)
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return total / seen, mean_dice
