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


def log_cosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """log(cosh(error)): quadratic near zero, linear far from it.

    L1 and L2 each get one half of what this problem wants. L2 gives a gradient that
    shrinks as the error does, which is the wrong signal for sub-voxel placement; L1
    keeps a constant gradient but is not differentiable at zero. Log-cosh is smooth at
    the origin like L2 and bounded in slope like L1, so outliers cannot dominate.
    That matters here because the boundary has real ambiguity of one to two voxels, and
    a loss that chases those voxels distorts the field everywhere else to reach them.

    softplus rather than log(cosh(x)) directly: cosh overflows in float32 around |x|=89,
    and log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2) is exact and stable everywhere.
    """
    e = (pred - target).abs()
    return (e + torch.log1p(torch.exp(-2.0 * e)) - 0.6931471805599453).mean()


def gradient_cosine_loss(pred: torch.Tensor, target: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """1 - cosine similarity between the two fields' spatial gradients.

    gradient_difference_loss compares gradient *magnitudes*; this compares their
    *directions*. For a distance field the direction is the surface normal and carries
    the geometry, while the magnitude should be 1 everywhere by definition — so
    penalising magnitude mismatch spends the loss on something the representation
    already fixes. That may be why the magnitude version measured +0.0010 and was
    inconsistent across folds.

    Gradients are taken with matched slices rather than padding, so all three axes are
    treated alike and no half-voxel shift is introduced. Where the true gradient is
    near zero — the saturated interior and exterior, most of the volume — its direction
    is noise, so the term is weighted by |grad target| and contributes nothing there.
    """
    def grads(v):
        return (v[:, :, 1:, :-1, :-1] - v[:, :, :-1, :-1, :-1],
                v[:, :, :-1, 1:, :-1] - v[:, :, :-1, :-1, :-1],
                v[:, :, :-1, :-1, 1:] - v[:, :, :-1, :-1, :-1])

    px, py, pz = grads(pred)
    gx, gy, gz = grads(target)

    dot = px * gx + py * gy + pz * gz
    norm_p = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + eps)
    norm_g = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + eps)

    misalignment = 1.0 - dot / (norm_p * norm_g + eps)
    # Only where the target actually has a direction to match.
    return (misalignment * norm_g).sum() / norm_g.sum().clamp_min(eps)


def surface_weighted_l1(pred: torch.Tensor, target: torch.Tensor,
                        sigma_mm: float = 1.5, trunc_mm: float = 10.0) -> torch.Tensor:
    """L1 weighted by exp(-|true sdf| / sigma), so error near the surface counts most.

    Plain L1 treats a 9mm error and a 1mm error as differing by a factor of nine, but
    everything downstream depends only on where the field crosses zero. Most voxels sit
    in the saturated far field at +-1, so they dominate the gradient while carrying no
    information about the boundary's position.

    This is a softer relative of the gradient-difference term, which emphasises the
    surface indirectly by penalising slope mismatch where the field changes fastest.
    That measured +0.0010 and was inconsistent across folds; weighting by distance to
    the surface says the same thing directly and with one interpretable parameter.

    sigma is in mm: at 1.5mm a voxel on the surface weighs 1.0, one 1.5mm away weighs
    0.37, and one 5mm away weighs 0.04. The field is normalised to [-1, 1] over
    trunc_mm, so it is converted back to mm to make sigma mean what it says.
    """
    weight = torch.exp(-target.abs() * trunc_mm / sigma_mm)
    return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(1e-6)


LOSSES = {
    "l1":       l1,
    "mse":      mse,
    "log_cosh": log_cosh,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES)}")
    return LOSSES[name]
