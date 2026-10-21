"""The method at inference, native image in and native SDF out, all on the GPU; every script that runs the networks uses it.

    z-score    the native volume, as common/image_cache.py
    pass 1     native -> x4 grid, the plain x4 network, the head volume from its output
    normalise  native -> the case's size-normalised grid
    pass 2     the normalised network(s), optionally with 8-flip TTA, their SDFs averaged
    to native  the field back on the native grid, optionally blended with pass 1's

Two resamplers give the same voxels at a different cost: fast (geometry.affine_resample) and the
caches' own (Spacingd going in, resample_grid coming out).
"""
import contextlib
import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor

from common.geometry import (CUBE, FACTOR, MM3_PER_ML, add_batch_axes, affine_resample, apply_resample,
                             crop_to_shape, downsample, drop_batch_axes, isotropic_spacing, resample_grid,
                             spacing_grid)
from common.model import UNet3D

FLIPS = [[axis for axis, flip in enumerate(flips) if flip]   # every subset of the 3 axes
         for flips in itertools.product((0, 1), repeat=3)]
POOLING_MULTIPLE = 8   # three poolings
K_RANGE = (0.5, 2.0)   # the accepted scale (V / V0)^(1/3): a head of about 16-1040 mL
OUTSIDE_SDF = 1.0      # beyond the scan the SDF reads "far outside"
MS_PER_S = 1e3


def load_net(path: str, device) -> UNet3D:
    """A trained x4 SDF network; its width is read from the checkpoint's first convolution."""
    state = torch.load(path, map_location=device)
    model = UNet3D(base_channels=state["enc1.block.0.weight"].shape[0]).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    _check_trained_on_x4_sdf(path)
    return model


def _check_trained_on_x4_sdf(checkpoint_path):
    """Runs of the resolution study (x2, x8, mask outputs) are not supported anymore."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
    if config.get("factor", FACTOR) != FACTOR or config.get("loss", "sdf") != "sdf":
        raise ValueError(f"{checkpoint_path}: trained with factor={config.get('factor')}, "
                         f"loss={config.get('loss')}; only x4 SDF networks are supported")


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad(x, cube):
    """Pad each axis at its end to the cube, or to the next multiple of 8 if larger; never crops."""
    target = [max(cube, round_up(n, POOLING_MULTIPLE)) for n in x.shape]
    widths = []
    for n, t in zip(reversed(x.shape), reversed(target)):   # F.pad lists the last axis first
        widths += [0, t - n]
    return F.pad(x, widths, value=0.0)


@torch.no_grad()
def forward(model, x, tta):
    """The network on one (D, H, W) volume; with tta, the mean of the 8 mirrored predictions."""
    outputs = []
    for axes in (FLIPS if tta else [[]]):
        flipped = torch.flip(x, axes) if axes else x
        prediction = crop_to_shape(drop_batch_axes(model(add_batch_axes(pad(flipped, CUBE)))), x.shape)
        outputs.append(torch.flip(prediction, axes) if axes else prediction)
    return torch.stack(outputs).mean(0)


class Resampler(NamedTuple):
    to_grid: Callable     # (native volume, native affine, spacing) -> (volume, its affine)
    to_native: Callable   # (field, its affine, native affine, native shape) -> native field


def to_grid_fast(x1, aff1, spacing):
    aff, shape = spacing_grid(aff1, x1.shape, spacing)
    return affine_resample(x1, aff1, aff, shape), aff


def to_native_fast(field, aff, aff1, shape1):
    return affine_resample(field, aff, aff1, shape1, cval=OUTSIDE_SDF)


def to_grid_as_cache(x1, aff1, spacing):
    """MONAI's Spacingd on the CPU, as the caches were built."""
    with_channel = MetaTensor(x1.cpu()[None], affine=torch.from_numpy(aff1))
    resampled = downsample({"x": with_channel}, spacing / isotropic_spacing(aff1), key="x")["x"]
    without_channel = resampled[0]
    return without_channel.as_tensor().to(x1.device).float(), np.asarray(resampled.affine.cpu())


def to_native_as_cache(field, aff, aff1, shape1):
    """A per-voxel float64 grid, as the old predict.py did."""
    grid = resample_grid(aff, aff1, shape1, device=field.device).float()
    return drop_batch_axes(apply_resample(add_batch_axes(field), grid, field.shape, cval=OUTSIDE_SDF))


def resampler(fast: bool) -> Resampler:
    return Resampler(to_grid_fast, to_native_fast) if fast else Resampler(to_grid_as_cache, to_native_as_cache)


class Clock:
    """Milliseconds per stage name, waiting for the GPU at both ends; disabled, it times nothing and never syncs."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t = {}

    def span(self, name):
        return self._timed(name) if self.enabled else contextlib.nullcontext()

    @contextlib.contextmanager
    def _timed(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - start) * MS_PER_S


def zscore(img1, device):
    """Zero mean, unit std over the volume; a constant volume stays zeros instead of NaN."""
    x1 = torch.from_numpy(np.asarray(img1, dtype=np.float32)).to(device)
    std = x1.std()
    return (x1 - x1.mean()) / (std if std > 0 else 1.0)


def volume_ml(field, spacing: float) -> float:
    """Head volume of an SDF on an isotropic grid."""
    return float((field < 0).sum()) * spacing ** 3 / MM3_PER_ML


@dataclass
class Pass1Estimate:
    volume_ml: float
    k: float          # the scale (V / V0)^(1/3)
    fallback: bool    # k outside K_RANGE: pass 1's own segmentation was returned


@torch.no_grad()
def segment(img1, aff1, pass1, models2, v0_ml, tta=False, device="cuda", clock=None, fast=True,
            blend=0.0, k_range=K_RANGE, volume=None) -> tuple[torch.Tensor, Pass1Estimate]:
    """The size-normalised method on one case: (native field on the GPU, pass 1's estimate); the mask is field < 0.

    v0_ml is the volume the pass-2 networks were trained at (their cache's reference_v0). blend mixes
    pass 1's field in: blend x pass 1 + (1 - blend) x pass 2. volume, when given, replaces pass 1's
    estimate (the oracle). A k outside k_range, including no head found (k = 0), falls back to pass 1.
    """
    clock = clock or Clock(enabled=False)
    to_grid, to_native = resampler(fast)
    shape1 = img1.shape
    native_spacing = isotropic_spacing(aff1)

    # Stages named "net: ..." are the networks alone; the rest is the pipeline around them.
    with clock.span("z-score"):
        x1 = zscore(img1, device)

    with clock.span("to x4 grid"):
        x4, aff4 = to_grid(x1, aff1, FACTOR * native_spacing)

    with clock.span("net: pass 1"):
        pass1_single = forward(pass1, x4, tta=False)
        pass1_field = forward(pass1, x4, tta=True) if blend > 0 and tta else pass1_single

    with clock.span("head volume"):
        estimated_ml = volume_ml(pass1_single, FACTOR * native_spacing) if volume is None else float(volume)
        k = (estimated_ml / v0_ml) ** (1 / 3)
    estimate = Pass1Estimate(estimated_ml, k, fallback=not k_range[0] <= k <= k_range[1])

    if estimate.fallback:
        with clock.span("SDF to native grid"):
            return to_native(pass1_field, aff4, aff1, shape1), estimate

    with clock.span("to normalised grid"):
        # No cube cap, unlike build_cache.py: a larger grid is padded; coarsening to fit cost 0.028 Dice.
        xn, affn = to_grid(x1, aff1, FACTOR * native_spacing * k)

    with clock.span("net: pass 2"):
        normalised_field = torch.stack([forward(m, xn, tta) for m in models2]).mean(0)

    with clock.span("SDF to native grid"):
        field = to_native(normalised_field, affn, aff1, shape1)
        if blend > 0:
            # Two views (real size, canonical size) whose errors partly cancel: blend 0.25 gave +0.0010 Dice.
            field = blend * to_native(pass1_field, aff4, aff1, shape1) + (1 - blend) * field

    return field, estimate


@torch.no_grad()
def segment_plain(img1, aff1, models, tta=False, device="cuda", clock=None, fast=True):
    """Without size normalisation: the network(s) on the x4 grid, straight back to native."""
    clock = clock or Clock(enabled=False)
    to_grid, to_native = resampler(fast)

    with clock.span("z-score"):
        x1 = zscore(img1, device)

    with clock.span("to x4 grid"):
        x4, aff4 = to_grid(x1, aff1, FACTOR * isotropic_spacing(aff1))

    with clock.span("net: x4"):
        field4 = torch.stack([forward(m, x4, tta) for m in models]).mean(0)

    with clock.span("SDF to native grid"):
        return to_native(field4, aff4, aff1, img1.shape)
