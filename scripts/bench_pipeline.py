# Measures the method's inference cost end to end, and is also the deployment pipeline
# written out in full.
#
#     python scripts/bench_pipeline.py --fold 0 --out bench_ours.json
#     python scripts/bench_pipeline.py --fold 0 --resampler geometry   # the slower resampling
#     python scripts/bench_pipeline.py --fold 0 --blend 0.25           # mix pass 1 into the result
#
# Per case, from the native volume in memory to the native mask, all on the GPU:
#     z-score     normalise the native volume, as cache.py does
#     pass 1      native -> x4 grid, the plain x4 network, head volume from its output
#     normalise   native -> the case's size-normalised grid (size_norm.py's spacing)
#     pass 2      the normalised network: once, with mirroring TTA (8 flips), or the
#                 3-seed ensemble with TTA (24 passes)
#     to native   the field back to the native grid, thresholded at 0; with --blend,
#                 mixed with pass 1's field first
#
# Reported for each configuration: time per case (median and mean, and per stage), peak
# GPU memory (allocated and reserved by PyTorch), peak RAM of the process, CPU time per
# case, parameters, and Dice (computed outside the timing, as a check). Run it with the
# GPU otherwise idle.
#
# --resampler fast (default) uses geometry.affine_resample; --resampler geometry uses the
# path the caches and predict.py use (Spacingd on the CPU going in, resample_grid coming
# out). The same voxels at a different cost.

import argparse
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "size_normalization_approach"))

from monai.data import MetaTensor  # noqa: E402

from cache import case_ids, load as load_image, load_label  # noqa: E402
from geometry import (affine_resample, apply_resample, downsample,  # noqa: E402
                      resample_grid, spacing_grid)
from model import UNet3D  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402

CUBE = 96       # the training input size; larger inputs are padded to a multiple of 8

# The 8 mirrorings for TTA: every subset of the 3 axes, from none to all three.
FLIPS = [[axis for axis, flip in enumerate(flips) if flip]
         for flips in [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]]


# --- Resampling, two interchangeable implementations ------------------------------------

# Native volume -> a grid of the given isotropic spacing. Returns (volume, its affine).
def to_grid_fast(x1, aff1, spacing):
    aff, shape = spacing_grid(aff1, x1.shape, spacing)
    return affine_resample(x1, aff1, aff, shape), aff


# Coarse SDF -> native grid.
def to_native_fast(f, aff, aff1, shape1):
    return affine_resample(f, aff, aff1, shape1, cval=1.0)


# The cache builders' way: MONAI's Spacingd on the CPU.
def to_grid_geometry(x1, aff1, spacing):
    native_spacing = float(np.abs(aff1[0, 0]))
    item = {"x": MetaTensor(x1.cpu()[None], affine=torch.from_numpy(aff1))}
    out = downsample(item, spacing / native_spacing, key="x")["x"]
    return out[0].as_tensor().to(x1.device).float(), np.asarray(out.affine.cpu())


# predict.py's way: a per-voxel float64 grid.
def to_native_geometry(f, aff, aff1, shape1):
    grid = resample_grid(aff, aff1, shape1, device=f.device).float()
    return apply_resample(f[None, None], grid, f.shape, cval=1.0)[0, 0]


# --- The network ----------------------------------------------------------------------

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
@torch.no_grad()
def forward(model, x, tta):
    shape, outputs = x.shape, []
    for axes in (FLIPS if tta else [[]]):
        xi = torch.flip(x, axes) if axes else x
        y = model(pad(xi, CUBE)[None, None])[0, 0, :shape[0], :shape[1], :shape[2]]
        # Mirror the prediction back before averaging.
        outputs.append(torch.flip(y, axes) if axes else y)
    return torch.stack(outputs).mean(0)


# --- Timing -----------------------------------------------------------------------------

# Accumulates milliseconds per stage name. GPU work is asynchronous, so each stage waits
# for it at both ends.
class Clock:
    def __init__(self):
        self.t = {}

    def span(self, name):
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

# One case, native image in, native mask out.
# v0_ml is the canonical head volume the second-pass networks were trained at (read from
# their training cache). blend > 0 mixes pass 1's SDF into the final one:
# blend x pass 1 + (1 - blend) x pass 2.
@torch.no_grad()
def run_case(img1, aff1, pass1, models2, tta, dev, clock, v0_ml, fast=True, blend=0.0):
    to_grid, to_native = ((to_grid_fast, to_native_fast) if fast
                          else (to_grid_geometry, to_native_geometry))
    shape1 = img1.shape
    native_spacing = float(np.abs(aff1[0, 0]))

    with clock.span("z-score"):
        x1 = torch.from_numpy(img1).to(dev).float()
        x1 = (x1 - x1.mean()) / x1.std()

    with clock.span("pass 1 (x4)"):
        # The plain x4 network; the head volume is the number of voxels where its SDF is
        # negative, times the voxel volume (mm^3 -> mL).
        x4, aff4 = to_grid(x1, aff1, 4 * native_spacing)
        f4 = forward(pass1, x4, tta=False)
        volume_ml = float((f4 < 0).sum()) * (4 * native_spacing) ** 3 / 1000

        # Blending reuses this field. With TTA on, pass 1 gets its 8 flips too (the volume
        # above stays the single-pass one).
        if blend > 0 and tta:
            f4 = forward(pass1, x4, tta=True)

    with clock.span("normalise"):
        # The spacing that shows this head at the canonical size. Unlike size_norm.py
        # there is no cube cap: a larger grid is padded instead (pad()), keeping the
        # canonical size; coarsening to fit cost 0.028 Dice on a scan 17% too large.
        k = (volume_ml / v0_ml) ** (1 / 3)
        xn, affn = to_grid(x1, aff1, 4 * native_spacing * k)

    with clock.span("pass 2"):
        # The size-normalised network(s); with several seeds, their SDFs are averaged.
        fn = torch.stack([forward(m, xn, tta) for m in models2]).mean(0)

    with clock.span("to native"):
        field = to_native(fn, affn, aff1, shape1)
        # Pass 1 sees the head at its real size on the plain x4 grid, pass 2 at the
        # canonical size: two different views, whose errors partly cancel. Both fields are
        # brought to the native grid first, since their own grids differ. On 90 cases,
        # blend 0.25 gave +0.0010 Dice without TTA (for ~5 ms) and +0.0004 with it.
        if blend > 0:
            field = blend * to_native(f4, aff4, aff1, shape1) + (1 - blend) * field
        mask = field < 0

    return mask


def main():
    p = argparse.ArgumentParser(description="Time the method end to end, per stage.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="only the first N cases")
    p.add_argument("--warmup", type=int, default=2, help="untimed cases run first")
    p.add_argument("--pass1", default="size_normalization_approach/runs/nnaug_x4e500_s{s}/fold{f}/checkpoints/phase1_last.pth")
    p.add_argument("--pass2", default="size_normalization_approach/runs/nnaug_x4norm_s{s}/fold{f}/checkpoints/phase1_last.pth")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"),
                   help="the second pass's training cache: its norm.json gives V0")
    p.add_argument("--resampler", choices=("fast", "geometry"), default="fast")
    p.add_argument("--blend", type=float, default=0.0,
                   help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()
    fast = a.resampler == "fast"

    dev = torch.device("cuda")
    proc = psutil.Process()

    def load_net(path):
        m = UNet3D().to(dev).eval().requires_grad_(False)
        m.load_state_dict(torch.load(os.path.join(ROOT, path), map_location=dev))
        return m

    # The fold's networks: one first pass, and three seeds of the second.
    pass1 = load_net(a.pass1.format(s=42, f=a.fold))
    seeds = [load_net(a.pass2.format(s=s, f=a.fold)) for s in (42, 43, 44)]
    params = sum(q.numel() for q in pass1.parameters())
    v0_ml = reference_v0(a.reference)

    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42,
                      splits_json=os.path.join(ROOT, "splits.json"))
    cases = fold_split(folds, a.fold)[1][:a.limit or None]

    # name -> (second-pass networks, TTA on/off)
    configs = {"one model": ([seeds[0]], False),
               "one model + TTA": ([seeds[0]], True),
               "ensemble of 3 + TTA": (seeds, True)}
    print(f"GPU {torch.cuda.get_device_name(0)}  CPU {psutil.cpu_count(logical=False)} cores  "
          f"{len(cases)} cases  network params {params:,} each (pass 1 + 1 or 3 pass 2)  "
          f"V0 {v0_ml:.1f} mL")

    results = {}
    for name, (models2, tta) in configs.items():
        # Warm-up: the first calls pay for CUDA initialisation and kernel selection.
        for cid in cases[:a.warmup]:
            img1, aff1 = load_image(a.cache_dir, cid, 1)
            run_case(img1.astype(np.float32), aff1, pass1, models2, tta, dev, Clock(), v0_ml,
                     fast, a.blend)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rss_peak, rows, dices = proc.memory_info().rss, [], []
        cpu_start = sum(proc.cpu_times()[:2])          # user + system CPU seconds

        for cid in cases:
            img1, aff1 = load_image(a.cache_dir, cid, 1)
            clock = Clock()
            mask = run_case(img1.astype(np.float32), aff1, pass1, models2, tta, dev, clock, v0_ml,
                            fast, a.blend)
            rows.append(clock.t)
            rss_peak = max(rss_peak, proc.memory_info().rss)

            # Dice against the label, as a check. Its CPU time is not the method's, so it
            # is added to the starting count, which removes it from the total.
            cpu_before_scoring = sum(proc.cpu_times()[:2])
            lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
            lab = torch.from_numpy(lab).to(dev)
            dices.append(float(2 * (mask & lab).sum() / (mask.sum() + lab.sum())))
            cpu_start += sum(proc.cpu_times()[:2]) - cpu_before_scoring

        cpu_s = sum(proc.cpu_times()[:2]) - cpu_start
        total_ms = np.array([sum(r.values()) for r in rows])
        results[name] = {
            "ms_median": float(np.median(total_ms)), "ms_mean": float(total_ms.mean()),
            "stages_ms_median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]},
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "ram_peak_mb": rss_peak / 2**20,
            "cpu_s_per_case": cpu_s / len(cases),
            # pass 1: once for the volume, plus 8 flips when it is blended with TTA on
            "networks_run": 1 + (8 if tta and a.blend > 0 else 0) + len(models2) * (8 if tta else 1),
            "parameters": params * (1 + len(models2)),
            "dice_mean": float(np.mean(dices)),
        }

        r = results[name]
        print(f"\n{name}: {r['ms_median']:.0f} ms/case median ({r['ms_mean']:.0f} mean), "
              f"GPU peak {r['gpu_peak_allocated_mb']:.0f} MB allocated / {r['gpu_peak_reserved_mb']:.0f} MB reserved, "
              f"RAM peak {r['ram_peak_mb']:.0f} MB, CPU {r['cpu_s_per_case']:.2f} s/case, "
              f"{r['networks_run']} network passes, {r['parameters']:,} params, Dice {r['dice_mean']:.4f}")
        print("   stages (median ms): " + ", ".join(f"{k} {v:.0f}" for k, v in r["stages_ms_median"].items()))

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
