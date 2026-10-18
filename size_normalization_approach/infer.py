# The method at inference: native image in, native SDF out, everything on the GPU.
#
# Used by every script that runs the networks: evaluation/predict.py, deploy/segment.py (a new
# NIfTI), scripts/bench_pipeline.py (timing) and the size tests, so what is scored is
# exactly what is deployed.
#
# Per case:
#     z-score     normalise the native volume, as cache.py does
#     pass 1      native -> x4 grid, the plain x4 network, head volume from its output
#     normalise   native -> the case's size-normalised grid (size_norm.py's spacing)
#     pass 2      the normalised network: once, with mirroring TTA (8 flips), or several
#                 seeds (an ensemble), their SDFs averaged
#     to native   the field back to the native grid; with blend, mixed with pass 1's first
#
# Two interchangeable resamplers: "fast" (geometry.affine_resample, the default) and
# "geometry", the path the training caches were built with (Spacingd on the CPU going in,
# resample_grid coming out). The same voxels at a different cost.

import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monai.data import MetaTensor  # noqa: E402

from geometry import (CUBE, FACTOR, affine_resample, apply_resample, downsample,  # noqa: E402
                      resample_grid, spacing_grid)
from model import UNet3D  # noqa: E402

# The 8 mirrorings for TTA: every subset of the 3 axes, from none to all three.
FLIPS = [[axis for axis, flip in enumerate(flips) if flip]
         for flips in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]]


# --- Networks ---------------------------------------------------------------------------

# A trained network and the grid it was trained on. train.py writes config.json in the run
# directory, two levels above the checkpoint; runs from before it are x4 SDF networks.
# Returns (model, config) with config {"factor", "cube", "loss"}.
def load_net(path: str, dev):
    state = torch.load(path, map_location=dev)
    # Width read from the checkpoint itself (first convolution). A checkpoint trained with
    # a contour or polar head carries its weights; the head is built so it loads, but unused.
    model = UNet3D(base_channels=state["enc1.block.0.weight"].shape[0],
                   contour_head="contour_head.weight" in state,
                   polar_dirs=state["polar_head.weight"].shape[0] if "polar_head.weight" in state else 0).to(dev)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)

    cfg = {"factor": FACTOR, "cube": CUBE, "loss": "sdf"}
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "config.json")
    if os.path.exists(cfg_path):
        cfg.update({k: v for k, v in json.load(open(cfg_path)).items() if k in cfg})
    return model, cfg


# Pad each axis to the training cube, or to the next multiple of 8 when the scan is
# larger (three poolings need divisibility by 8, and the network is fully convolutional).
# Never crops: F.pad with a negative width would do that silently.
def pad(x, cube):
    target = [max(cube, -(-n // 8) * 8) for n in x.shape]   # -(-n // 8): n/8 rounded up
    widths = []
    for n, t in zip(reversed(x.shape), reversed(target)):   # F.pad lists the last axis first
        widths += [0, t - n]                                # nothing before, the rest after
    return F.pad(x, widths, value=0.0)


# Run a network on one volume, optionally averaging the 8 mirrored predictions (TTA).
# `cube` is the smallest input the network was trained at (96 at x4, 192 at x2); a mask
# network (mask=True) gives the field 1 - 2 sigmoid(logits), negative inside like an SDF:
# interpolating 1 - 2p and cutting at 0 is interpolating p and cutting at 0.5.
@torch.no_grad()
def forward(model, x, tta, cube=CUBE, mask=False):
    shape, outputs = x.shape, []
    for axes in (FLIPS if tta else [[]]):
        xi = torch.flip(x, axes) if axes else x
        xin = pad(xi, cube)[None, None]
        y = model(xin) if not mask else 1 - 2 * torch.sigmoid(model.logits(xin))
        y = y[0, 0, :shape[0], :shape[1], :shape[2]]
        # Mirror the prediction back before averaging.
        outputs.append(torch.flip(y, axes) if axes else y)
    return torch.stack(outputs).mean(0)


# --- Resampling, two interchangeable implementations ------------------------------------

# Native volume -> a grid of the given isotropic spacing. Returns (volume, its affine).
def to_grid_fast(x1, aff1, spacing):
    aff, shape = spacing_grid(aff1, x1.shape, spacing)
    return affine_resample(x1, aff1, aff, shape), aff


# Coarse SDF -> native grid. cval=1: beyond the scan, the SDF is "far outside".
def to_native_fast(f, aff, aff1, shape1):
    return affine_resample(f, aff, aff1, shape1, cval=1.0)


# The cache builders' way: MONAI's Spacingd on the CPU.
def to_grid_geometry(x1, aff1, spacing):
    native_spacing = float(np.abs(aff1[0, 0]))
    item = {"x": MetaTensor(x1.cpu()[None], affine=torch.from_numpy(aff1))}
    out = downsample(item, spacing / native_spacing, key="x")["x"]
    return out[0].as_tensor().to(x1.device).float(), np.asarray(out.affine.cpu())


# The old predict.py's way: a per-voxel float64 grid.
def to_native_geometry(f, aff, aff1, shape1):
    grid = resample_grid(aff, aff1, shape1, device=f.device).float()
    return apply_resample(f[None, None], grid, f.shape, cval=1.0)[0, 0]


def resamplers(fast: bool):
    return (to_grid_fast, to_native_fast) if fast else (to_grid_geometry, to_native_geometry)


# --- Timing -----------------------------------------------------------------------------

# Accumulates milliseconds per stage name. GPU work is asynchronous, so each stage waits
# for it at both ends. Clock(enabled=False), the default everywhere, times nothing and
# never synchronises.
class Clock:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t = {}

    def span(self, name):
        if not self.enabled:
            return contextlib.nullcontext()
        clock = self

        class Span:
            def __enter__(self):
                torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                torch.cuda.synchronize()
                clock.t[name] = clock.t.get(name, 0.0) + (time.perf_counter() - self.t0) * 1e3

        return Span()


# --- The pipeline -----------------------------------------------------------------------

# Zero mean, unit std over the whole volume, as cache.py. A constant volume (std 0) stays
# all zeros instead of turning into NaN.
def zscore(img1, dev):
    x1 = torch.from_numpy(np.asarray(img1, dtype=np.float32)).to(dev)
    std = x1.std()
    return (x1 - x1.mean()) / (std if std > 0 else 1.0)


# Head volume in mL of an SDF on an isotropic grid: the voxels inside, times their volume.
def volume_ml(field, spacing: float) -> float:
    return float((field < 0).sum()) * spacing ** 3 / 1000


# The size-normalised method on one case. Returns (native field on the GPU, info), where
# info = {"volume_ml": pass 1's estimate, "k": the scale, "fallback": True when the scale
# was rejected}; the mask is field < 0.
#
# v0_ml is the canonical head volume the second-pass networks were trained at (read from
# their training cache, size_norm.reference_v0). models2 are the second-pass networks, all
# trained on the same grid (factor2, cube2, mask2: the resolution study's x2 or mask
# networks); with several, their SDFs are averaged. blend > 0 mixes pass 1's SDF into the
# final one: blend x pass 1 + (1 - blend) x pass 2. volume_ml, when given, replaces pass 1's
# estimate (the oracle: the label's volume).
#
# k_range guards the normalisation: pass 1's scale k = (V / V0)^(1/3) is used only within
# it (0.5-2: a head of about 16-1040 mL). Outside it, including a pass 1 that finds no head
# (k = 0, an infinite spacing) or a tiny estimate (a grid too large for the GPU), the case
# returns pass 1's own segmentation instead (size_shift_test.py).
@torch.no_grad()
def segment(img1, aff1, pass1, models2, v0_ml, tta=False, dev="cuda", clock=None, fast=True,
            blend=0.0, factor2=4, cube2=CUBE, mask2=False, k_range=(0.5, 2.0), volume=None):
    clock = clock or Clock(enabled=False)
    to_grid, to_native = resamplers(fast)
    shape1 = img1.shape
    native_spacing = float(np.abs(aff1[0, 0]))

    with clock.span("z-score"):
        x1 = zscore(img1, dev)

    # Stage names starting with "net:" are the networks alone; everything else is the
    # pipeline around them (resampling, normalising, the volume, thresholding, copies).
    with clock.span("to x4 grid"):
        x4, aff4 = to_grid(x1, aff1, 4 * native_spacing)

    with clock.span("net: pass 1"):
        # The plain x4 network.
        f4 = forward(pass1, x4, tta=False)
        # Blending reuses this field. With TTA on, pass 1 gets its 8 flips too (the volume
        # below stays the single-pass one).
        if blend > 0 and tta:
            f4_blend = forward(pass1, x4, tta=True)

    with clock.span("head volume"):
        est_ml = volume_ml(f4, 4 * native_spacing) if volume is None else float(volume)
        k = (est_ml / v0_ml) ** (1 / 3)
        if blend > 0 and tta:
            f4 = f4_blend
    info = {"volume_ml": est_ml, "k": k, "fallback": not k_range[0] <= k <= k_range[1]}

    # No usable scale: fall back to pass 1's segmentation.
    if info["fallback"]:
        with clock.span("SDF to native grid"):
            return to_native(f4, aff4, aff1, shape1), info

    with clock.span("to normalised grid"):
        # The spacing that shows this head at the canonical size. Unlike size_norm.py
        # there is no cube cap: a larger grid is padded instead (pad()), keeping the
        # canonical size; coarsening to fit cost 0.028 Dice on a scan 17% too large.
        xn, affn = to_grid(x1, aff1, factor2 * native_spacing * k)

    with clock.span("net: pass 2"):
        fn = torch.stack([forward(m, xn, tta, cube2, mask2) for m in models2]).mean(0)

    with clock.span("SDF to native grid"):
        field = to_native(fn, affn, aff1, shape1)
        # Pass 1 sees the head at its real size on the plain x4 grid, pass 2 at the
        # canonical size: two different views, whose errors partly cancel. Both fields are
        # brought to the native grid first, since their own grids differ. On 90 cases,
        # blend 0.25 gave +0.0010 Dice without TTA (for ~5 ms) and +0.0004 with it.
        if blend > 0:
            field = blend * to_native(f4, aff4, aff1, shape1) + (1 - blend) * field

    return field, info


# The plain method, without size normalisation: one pass of the network(s) on their own
# fixed grid (x4 unless trained otherwise), each averaged over its 8 flips with TTA,
# straight back to the native grid. Returns the native field on the GPU.
@torch.no_grad()
def segment_plain(img1, aff1, models, tta=False, dev="cuda", clock=None, fast=True,
                  factor=4, cube=CUBE, mask=False):
    clock = clock or Clock(enabled=False)
    to_grid, to_native = resamplers(fast)
    native_spacing = float(np.abs(aff1[0, 0]))

    with clock.span("z-score"):
        x1 = zscore(img1, dev)

    with clock.span("to x4 grid"):
        xg, affg = to_grid(x1, aff1, factor * native_spacing)

    with clock.span("net: x4"):
        fg = torch.stack([forward(m, xg, tta, cube, mask) for m in models]).mean(0)

    with clock.span("SDF to native grid"):
        return to_native(fg, affg, aff1, img1.shape)
