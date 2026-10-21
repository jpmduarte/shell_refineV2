"""The refiner's loss and epoch loops, driven by train_cube.py; the network sees the whole cube but is scored in the band."""
import torch
import torch.nn.functional as F

from common.metrics import dice
from common.sdf_targets import TRUNC_MM

EPS = 1e-6
PRIOR_CHANNEL = slice(1, 2)   # x[:, 1:2], kept as a channel axis


def dice_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Soft Dice inside the band plus BCE weighted by `weight`, which is zero outside the band.

    The Dice term uses the binary band even when `weight` is graded, so it stays the band Dice
    that evaluate.py reports; grading only acts on the per-voxel BCE.
    """
    bce_map = F.binary_cross_entropy(pred, target, reduction="none")
    in_band = (weight > 0).to(weight.dtype)
    intersection = (pred * target * in_band).sum()
    denominator = (pred * in_band).sum() + (target * in_band).sum()
    soft_dice = 1.0 - (2.0 * intersection + EPS) / (denominator + EPS)
    bce = (bce_map * weight).sum() / weight.sum().clamp_min(EPS)
    return soft_dice + bce


def surface_weight(x: torch.Tensor, band: torch.Tensor, sigma_mm: float, trunc_mm: float = TRUNC_MM) -> torch.Tensor:
    """The band graded by exp(-|prior| / sigma): the gradient goes where the surface is decided."""
    prior = x[:, PRIOR_CHANNEL]
    return band * torch.exp(-prior.abs() * trunc_mm / sigma_mm)


def train_epoch(model, loader, optimizer, device, sw_sigma: float = 0.0) -> float:
    """Mean loss per cube; sw_sigma 0 weights the band uniformly."""
    model.train()
    total, seen = 0.0, 0
    for x, y, band in loader:
        x, y, band = x.to(device), y.to(device), band.to(device)
        pred = model(x)
        weight = surface_weight(x, band, sw_sigma) if sw_sigma > 0 else band
        loss = dice_bce(pred, y, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        seen += x.size(0)
    return total / seen


@torch.no_grad()
def val_epoch(model, loader, device) -> tuple[float, float]:
    """(mean loss per cube, mean Dice within the band), what evaluate.py reports as "band"."""
    model.eval()
    total, seen = 0.0, 0
    dice_scores = []
    for x, y, band in loader:
        x, y, band = x.to(device), y.to(device), band.to(device)
        pred = model(x)
        total += dice_bce(pred, y, band).item() * x.size(0)
        seen += x.size(0)
        for cube_pred, cube_target, cube_band in zip(pred, y, band):
            in_band = cube_band > 0.5
            if in_band.any():
                score = dice(cube_pred[in_band], cube_target[in_band])
                if score is not None:
                    dice_scores.append(score)
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return total / seen, mean_dice
