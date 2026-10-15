# Measures the method's inference cost end to end, and is also the deployment pipeline
# written out in full.
#
#     python scripts/bench_pipeline.py --folds 0 1 2 3 4 --repeats 3 --out bench_ours.json
#     python scripts/bench_pipeline.py --folds 0 --resampler geometry   # the slower resampling
#     python scripts/bench_pipeline.py --folds 0 --blend 0.25           # mix pass 1 into the result
#
# Per case, from the native volume in memory to the native mask, all on the GPU:
#     z-score     normalise the native volume, as cache.py does
#     pass 1      native -> x4 grid, the plain x4 network, head volume from its output
#     normalise   native -> the case's size-normalised grid (size_norm.py's spacing)
#     pass 2      the normalised network: once, with mirroring TTA (8 flips), or the
#                 3-seed ensemble with TTA (24 passes)
#     to native   the field back to the native grid, thresholded at 0, the mask copied to
#                 host memory; with --blend, mixed with pass 1's field first
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
# `cube` is the smallest input the network was trained at (96 at x4, 192 at x2); a mask
# network (mask=True) gives the field 1 - 2 sigmoid(logits), negative inside like an SDF.
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
# factor2, cube2 and mask2 describe the second-pass networks (the resolution study: x2 in
# 192^3, mask outputs); the first pass is always the plain x4 SDF network.
#
# k_range guards the normalisation: pass 1's scale k = (V / V0)^(1/3) is used only within
# it (0.5-2: a head of about 16-1040 mL). Outside it, including a pass 1 that finds no head
# (k = 0, an infinite spacing) or a tiny estimate (a grid too large for the GPU), the case
# returns pass 1's own segmentation instead (size_normalization_approach/size_shift_test.py).
@torch.no_grad()
def run_case(img1, aff1, pass1, models2, tta, dev, clock, v0_ml, fast=True, blend=0.0,
             factor2=4, cube2=CUBE, mask2=False, k_range=(0.5, 2.0)):
    to_grid, to_native = ((to_grid_fast, to_native_fast) if fast
                          else (to_grid_geometry, to_native_geometry))
    shape1 = img1.shape
    native_spacing = float(np.abs(aff1[0, 0]))

    with clock.span("z-score"):
        x1 = torch.from_numpy(img1).to(dev).float()
        x1 = (x1 - x1.mean()) / x1.std()

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
        # The number of voxels where pass 1's SDF is negative, times the voxel volume
        # (mm^3 -> mL).
        volume_ml = float((f4 < 0).sum()) * (4 * native_spacing) ** 3 / 1000
        k = (volume_ml / v0_ml) ** (1 / 3)
        if blend > 0 and tta:
            f4 = f4_blend

    # No usable scale: fall back to pass 1's segmentation.
    if not k_range[0] <= k <= k_range[1]:
        with clock.span("SDF to native grid"):
            field = to_native(f4, aff4, aff1, shape1)
        with clock.span("threshold + to host"):
            return (field < 0).cpu()

    with clock.span("to normalised grid"):
        # The spacing that shows this head at the canonical size. Unlike size_norm.py
        # there is no cube cap: a larger grid is padded instead (pad()), keeping the
        # canonical size; coarsening to fit cost 0.028 Dice on a scan 17% too large.
        xn, affn = to_grid(x1, aff1, factor2 * native_spacing * k)

    with clock.span("net: pass 2"):
        # The size-normalised network(s); with several seeds, their SDFs are averaged.
        fn = torch.stack([forward(m, xn, tta, cube2, mask2) for m in models2]).mean(0)

    with clock.span("SDF to native grid"):
        field = to_native(fn, affn, aff1, shape1)
        # Pass 1 sees the head at its real size on the plain x4 grid, pass 2 at the
        # canonical size: two different views, whose errors partly cancel. Both fields are
        # brought to the native grid first, since their own grids differ. On 90 cases,
        # blend 0.25 gave +0.0010 Dice without TTA (for ~5 ms) and +0.0004 with it.
        if blend > 0:
            field = blend * to_native(f4, aff4, aff1, shape1) + (1 - blend) * field

    with clock.span("threshold + to host"):
        # The mask is handed back in host memory, as nnU-Net's predictor returns its
        # segmentation, so both timings end at the same place.
        mask = (field < 0).cpu()

    return mask


# The plain x4 method, without size normalisation: one pass of the x4 network(s), each
# averaged over its 8 flips with TTA, straight back to the native grid.
@torch.no_grad()
def run_case_plain(img1, aff1, models, tta, dev, clock, fast=True):
    to_grid, to_native = ((to_grid_fast, to_native_fast) if fast
                          else (to_grid_geometry, to_native_geometry))
    native_spacing = float(np.abs(aff1[0, 0]))

    with clock.span("z-score"):
        x1 = torch.from_numpy(img1).to(dev).float()
        x1 = (x1 - x1.mean()) / x1.std()

    with clock.span("to x4 grid"):
        x4, aff4 = to_grid(x1, aff1, 4 * native_spacing)

    with clock.span("net: x4"):
        f4 = torch.stack([forward(m, x4, tta) for m in models]).mean(0)

    with clock.span("SDF to native grid"):
        field = to_native(f4, aff4, aff1, img1.shape)

    with clock.span("threshold + to host"):
        mask = (field < 0).cpu()

    return mask


def main():
    p = argparse.ArgumentParser(description="Time the method end to end, per stage.")
    p.add_argument("--folds", type=int, nargs="+", default=[0],
                   help="each fold's validation cases with that fold's networks (0 1 2 3 4: all 90)")
    p.add_argument("--repeats", type=int, default=1,
                   help="timed passes over the cases; the spread between them shows the run-to-run noise")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases of each fold")
    p.add_argument("--warmup", type=int, default=2, help="untimed cases run first")
    p.add_argument("--pass1", default="size_normalization_approach/runs/nnaug_x4e500_s{s}/fold{f}/checkpoints/phase1_last.pth")
    p.add_argument("--pass2", default="size_normalization_approach/runs/nnaug_x4norm_s{s}/fold{f}/checkpoints/phase1_last.pth")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"),
                   help="the second pass's training cache: its norm.json gives V0")
    p.add_argument("--resampler", choices=("fast", "geometry"), default="fast")
    p.add_argument("--blend", type=float, default=0.0,
                   help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="second-pass seeds; the ensemble configurations need more than one")
    p.add_argument("--pass2-factor", type=int, default=4,
                   help="the second pass's grid (2 for the x2 networks); its output type comes "
                        "from the run's config.json")
    p.add_argument("--no-plain", action="store_true",
                   help="skip the plain x4 configurations (no size normalisation)")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()
    cube2 = CUBE * 4 // a.pass2_factor
    fast = a.resampler == "fast"

    dev = torch.device("cuda")
    proc = psutil.Process()

    def load_net(path):
        m = UNet3D().to(dev).eval().requires_grad_(False)
        m.load_state_dict(torch.load(os.path.join(ROOT, path), map_location=dev))
        return m

    v0_ml = reference_v0(a.reference)
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42,
                      splits_json=os.path.join(ROOT, "splits.json"))

    # Each fold's networks: one first pass, and three seeds of the second.
    nets = {}
    for f in a.folds:
        nets[f] = (load_net(a.pass1.format(s=42, f=f)),
                   [load_net(a.pass2.format(s=s, f=f)) for s in a.seeds],
                   fold_split(folds, f)[1][:a.limit or None],
                   [] if a.no_plain else [load_net(a.pass1.format(s=s, f=f)) for s in a.seeds])

    # Whether the second pass outputs a mask: train.py's config.json beside the checkpoints.
    # (train.py writes it in the run directory, two levels above the checkpoint.)
    cfg_path = os.path.join(ROOT, os.path.dirname(os.path.dirname(
        a.pass2.format(s=a.seeds[0], f=a.folds[0]))), "config.json")
    mask2 = os.path.exists(cfg_path) and json.load(open(cfg_path)).get("loss") == "mask"
    params = sum(q.numel() for q in nets[a.folds[0]][0].parameters())
    n_cases = sum(len(n[2]) for n in nets.values())

    # name -> (size normalisation on/off, how many seeds, TTA on/off). Without
    # normalisation the x4 networks (--pass1) are the method; with it, the normalised
    # networks (--pass2) after one x4 pass for the volume.
    n = len(a.seeds)
    configs = {"x4: one model": (False, 1, False),
               "x4: one model + TTA": (False, 1, True),
               f"x4: ensemble of {n}": (False, n, False),
               f"x4: ensemble of {n} + TTA": (False, n, True),
               "one model": (True, 1, False),
               "one model + TTA": (True, 1, True),
               f"ensemble of {n}": (True, n, False),
               f"ensemble of {n} + TTA": (True, n, True)}
    configs = {k: v for k, v in configs.items()
               if (n > 1 or "ensemble" not in k) and not (a.no_plain and not v[0])}
    print(f"second pass: x{a.pass2_factor} in {cube2}^3, output {'mask' if mask2 else 'SDF'}")
    print(f"GPU {torch.cuda.get_device_name(0)}  CPU {psutil.cpu_count(logical=False)} cores  "
          f"folds {a.folds}, {n_cases} cases x {a.repeats} repeats  network params {params:,} "
          f"each (pass 1 + 1 or 3 pass 2)  V0 {v0_ml:.1f} mL")

    results = {}
    # One case through the chosen configuration; returns the native mask.
    def run(f, img1, aff1, norm, n_models, tta, clock):
        pass1, seeds, _, plain = nets[f]
        if norm:
            return run_case(img1, aff1, pass1, seeds[:n_models], tta, dev, clock, v0_ml,
                            fast, a.blend, a.pass2_factor, cube2, mask2)
        return run_case_plain(img1, aff1, plain[:n_models], tta, dev, clock, fast)

    for name, (norm, n_models, tta) in configs.items():
        # Warm-up: the first calls pay for CUDA initialisation and kernel selection.
        f0 = a.folds[0]
        for cid in nets[f0][2][:a.warmup]:
            img1, aff1 = load_image(a.cache_dir, cid, 1)
            run(f0, img1.astype(np.float32), aff1, norm, n_models, tta, Clock())

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rss_peak, rows, dices, repeat_medians = proc.memory_info().rss, [], [], []
        cpu_start = sum(proc.cpu_times()[:2])          # user + system CPU seconds

        for rep in range(a.repeats):
            rep_rows = []
            for f in a.folds:
                cases = nets[f][2]
                for cid in cases:
                    cpu_before_reading = sum(proc.cpu_times()[:2])
                    img1, aff1 = load_image(a.cache_dir, cid, 1)
                    img1 = img1.astype(np.float32)
                    cpu_start += sum(proc.cpu_times()[:2]) - cpu_before_reading
                    clock = Clock()
                    mask = run(f, img1, aff1, norm, n_models, tta, clock)
                    rep_rows.append(clock.t)
                    rss_peak = max(rss_peak, proc.memory_info().rss)

                    # Dice against the label, as a check (first repeat only). Its CPU time
                    # is not the method's, so it is added to the starting count, which
                    # removes it from the total; so is reading the case from disk.
                    cpu_before_scoring = sum(proc.cpu_times()[:2])
                    if rep == 0:
                        lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
                        m = mask.numpy()
                        dices.append(float(2 * (m & lab).sum() / (m.sum() + lab.sum())))
                    cpu_start += sum(proc.cpu_times()[:2]) - cpu_before_scoring
            repeat_medians.append(float(np.median([sum(r.values()) for r in rep_rows])))
            rows += rep_rows

        cpu_s = sum(proc.cpu_times()[:2]) - cpu_start
        total_ms = np.array([sum(r.values()) for r in rows])
        # Per case: the networks alone, and the pipeline around them (total minus networks).
        net_ms = np.array([sum(v for k, v in r.items() if k.startswith("net:")) for r in rows])
        results[name] = {
            "ms_median": float(np.median(total_ms)), "ms_mean": float(total_ms.mean()),
            "network_ms_median": float(np.median(net_ms)),
            "pipeline_ms_median": float(np.median(total_ms - net_ms)),
            "ms_p5": float(np.percentile(total_ms, 5)), "ms_p95": float(np.percentile(total_ms, 95)),
            "ms_median_per_repeat": repeat_medians,
            "n_timed": int(len(total_ms)),
            "stages_ms_median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]},
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "ram_peak_mb": rss_peak / 2**20,
            "cpu_s_per_case": cpu_s / len(total_ms),
            # pass 1: once for the volume, plus 8 flips when it is blended with TTA on
            "networks_run": (1 + (8 if tta and a.blend > 0 else 0) if norm else 0)
                            + n_models * (8 if tta else 1),
            "parameters": params * (n_models + (1 if norm else 0)),
            "dice_mean": float(np.mean(dices)),
        }

        r = results[name]
        print(f"\n{name}: {r['ms_median']:.0f} ms/case median ({r['ms_mean']:.0f} mean, "
              f"p5-p95 {r['ms_p5']:.0f}-{r['ms_p95']:.0f}, medians per repeat "
              f"{', '.join(f'{m:.0f}' for m in repeat_medians)}; {r['n_timed']} timed), "
              f"GPU peak {r['gpu_peak_allocated_mb']:.0f} MB allocated / {r['gpu_peak_reserved_mb']:.0f} MB reserved, "
              f"RAM peak {r['ram_peak_mb']:.0f} MB, CPU {r['cpu_s_per_case']:.2f} s/case, "
              f"{r['networks_run']} network passes, {r['parameters']:,} params, Dice {r['dice_mean']:.4f}")
        print(f"   networks only {r['network_ms_median']:.1f} ms, pipeline around them "
              f"{r['pipeline_ms_median']:.1f} ms (medians per case)")
        print("   stages (median ms): " + ", ".join(f"{k} {v:.1f}" for k, v in r["stages_ms_median"].items()))

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
