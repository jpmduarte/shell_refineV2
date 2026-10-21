# Measures the method's inference cost end to end. The pipeline itself is
# size_normalization/infer.py; this only times it.
#
#     python scripts/benchmark_inference.py --folds 0 1 2 3 4 --repeats 3 --out bench_ours.json
#     python scripts/benchmark_inference.py --folds 0 --resampler geometry   # the slower resampling
#     python scripts/benchmark_inference.py --folds 0 --blend 0.25           # mix pass 1 into the result
#
# Per case, from the native volume in memory to the native mask in host memory, all on
# the GPU (the stages are infer.py's), for the plain x4 method and the size-normalised one,
# each with one model or the 3-seed ensemble, with and without TTA.
#
# Reported for each configuration: time per case (median and mean, and per stage), peak
# GPU memory (allocated and reserved by PyTorch), peak RAM of the process, CPU time per
# case, parameters, and Dice (computed outside the timing, as a check). Run it with the
# GPU otherwise idle.

import argparse
import json
import os
import sys

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import ROOT, SPLITS_JSON, NATIVE_CACHE, NORM_CACHE  # noqa: E402

from common.image_cache import case_ids, load as load_image, load_label  # noqa: E402
from size_normalization.infer import Clock, load_net, segment, segment_plain  # noqa: E402
from size_normalization.build_cache import reference_v0  # noqa: E402
from common.folds import fold_split, get_folds  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Time the method end to end, per stage.")
    p.add_argument("--folds", type=int, nargs="+", default=[0],
                   help="each fold's validation cases with that fold's networks (0 1 2 3 4: all 90)")
    p.add_argument("--repeats", type=int, default=1,
                   help="timed passes over the cases; the spread between them shows the run-to-run noise")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases of each fold")
    p.add_argument("--warmup", type=int, default=2, help="untimed cases run first")
    p.add_argument("--pass1", default="size_normalization/runs/nnaug_x4e500_s{s}/fold{f}/checkpoints/phase1_last.pth")
    p.add_argument("--pass2", default="size_normalization/runs/nnaug_x4norm_s{s}/fold{f}/checkpoints/phase1_last.pth",
                   help="the second-pass networks; their grid and output type come from the "
                        "run's config.json")
    p.add_argument("--cache-dir", default=NATIVE_CACHE)
    p.add_argument("--reference", default=NORM_CACHE,
                   help="the second pass's training cache: its norm.json gives V0")
    p.add_argument("--resampler", choices=("fast", "geometry"), default="fast")
    p.add_argument("--blend", type=float, default=0.0,
                   help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="second-pass seeds; the ensemble configurations need more than one")
    p.add_argument("--no-plain", action="store_true",
                   help="skip the plain x4 configurations (no size normalisation)")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()
    fast = a.resampler == "fast"

    dev = torch.device("cuda")
    proc = psutil.Process()

    def net(path):
        return load_net(os.path.join(ROOT, path), dev)[0]

    v0_ml = reference_v0(a.reference)
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42,
                      splits_json=SPLITS_JSON)

    # Each fold's networks: one first pass, and three seeds of the second.
    nets = {}
    for f in a.folds:
        nets[f] = (net(a.pass1.format(s=42, f=f)),
                   [net(a.pass2.format(s=s, f=f)) for s in a.seeds],
                   fold_split(folds, f)[1][:a.limit or None],
                   [] if a.no_plain else [net(a.pass1.format(s=s, f=f)) for s in a.seeds])

    # The second pass's grid and output (the resolution study: x2 in 192^3, mask outputs).
    cfg = load_net(os.path.join(ROOT, a.pass2.format(s=a.seeds[0], f=a.folds[0])), dev)[1]
    grid = dict(factor2=cfg["factor"], cube2=cfg["cube"], mask2=cfg["loss"] == "mask")
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
    print(f"second pass: x{cfg['factor']} in {cfg['cube']}^3, output {cfg['loss']}")
    print(f"GPU {torch.cuda.get_device_name(0)}  CPU {psutil.cpu_count(logical=False)} cores  "
          f"folds {a.folds}, {n_cases} cases x {a.repeats} repeats  network params {params:,} "
          f"each (pass 1 + 1 or 3 pass 2)  V0 {v0_ml:.1f} mL")

    results = {}
    # One case through the chosen configuration; returns the native mask in host memory,
    # as nnU-Net's predictor returns its segmentation, so both timings end at the same place.
    def run(f, img1, aff1, norm, n_models, tta, clock):
        pass1, seeds, _, plain = nets[f]
        if norm:
            field, _ = segment(img1, aff1, pass1, seeds[:n_models], v0_ml, tta, dev, clock,
                               fast, a.blend, **grid)
        else:
            field = segment_plain(img1, aff1, plain[:n_models], tta, dev, clock, fast)
        with clock.span("threshold + to host"):
            return (field < 0).cpu()

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
