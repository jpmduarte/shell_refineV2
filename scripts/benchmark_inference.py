"""Time the method end to end, per stage, from the native volume in memory to the native mask in host memory.

The pipeline is size_normalization/infer.py; this only times it, for the plain x4 and the
size-normalised method, each with one model or the seed ensemble, with and without TTA. Reported per
configuration: time per case (median, mean, per stage), peak GPU memory, peak RAM, CPU time per case,
parameters and Dice (outside the timing, as a check). Run it with the GPU otherwise idle.
"""
import argparse
import json
import os
from typing import NamedTuple

import numpy as np
import psutil
import torch

from common.folds import N_FOLDS, SEED as FOLDS_SEED, fold_split, get_folds
from common.image_cache import case_ids, load as load_image, load_label
from common.metrics import mask_dice
from common.paths import LAST_CHECKPOINT, NATIVE_CACHE, NORM_CACHE, PASS1_RUN, PASS2_RUN, ROOT, SPLITS_JSON
from size_normalization.build_cache import reference_v0
from size_normalization.infer import FLIPS, Clock, load_net, segment, segment_plain

MB = 2 ** 20
RUN_PATTERN = "size_normalization/runs/{run}_s{{s}}/fold{{f}}/checkpoints/" + LAST_CHECKPOINT
SEED_OF_PASS1 = 42

USAGE = """\
examples:
  python scripts/benchmark_inference.py --folds 0 1 2 3 4 --repeats 3 --out bench_ours.json
  python scripts/benchmark_inference.py --folds 0 --resampler geometry   # the caches' slower resampling
  python scripts/benchmark_inference.py --folds 0 --blend 0.25           # mix pass 1 into the result"""


class Configuration(NamedTuple):
    normalised: bool   # the size-normalised method; otherwise the plain x4 networks are the method
    n_models: int
    tta: bool


class FoldNetworks(NamedTuple):
    pass1: torch.nn.Module
    pass2: list
    cases: list
    plain: list


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="each fold's validation cases with that fold's networks (0 1 2 3 4: all 90)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed passes over the cases; the spread between them shows the run-to-run noise")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases of each fold")
    parser.add_argument("--warmup", type=int, default=2, help="untimed cases run first")
    parser.add_argument("--pass1", default=RUN_PATTERN.format(run=PASS1_RUN))
    parser.add_argument("--pass2", default=RUN_PATTERN.format(run=PASS2_RUN), help="the second-pass networks")
    parser.add_argument("--cache-dir", default=NATIVE_CACHE)
    parser.add_argument("--reference", default=NORM_CACHE,
                        help="the second pass's training cache: its norm.json gives V0")
    parser.add_argument("--resampler", choices=("fast", "geometry"), default="fast")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="second-pass seeds; the ensemble configurations need more than one")
    parser.add_argument("--no-plain", action="store_true",
                        help="skip the plain x4 configurations (no size normalisation)")
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    return parser.parse_args()


def configurations(n_seeds, no_plain) -> dict[str, Configuration]:
    configs = {"x4: one model": Configuration(False, 1, False),
               "x4: one model + TTA": Configuration(False, 1, True),
               f"x4: ensemble of {n_seeds}": Configuration(False, n_seeds, False),
               f"x4: ensemble of {n_seeds} + TTA": Configuration(False, n_seeds, True),
               "one model": Configuration(True, 1, False),
               "one model + TTA": Configuration(True, 1, True),
               f"ensemble of {n_seeds}": Configuration(True, n_seeds, False),
               f"ensemble of {n_seeds} + TTA": Configuration(True, n_seeds, True)}
    return {name: c for name, c in configs.items()
            if (n_seeds > 1 or "ensemble" not in name) and not (no_plain and not c.normalised)}


def cpu_seconds(proc) -> float:
    user, system = proc.cpu_times()[:2]
    return user + system


class Benchmark:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.fast = args.resampler == "fast"
        self.proc = psutil.Process()
        self.v0_ml = reference_v0(args.reference)
        folds = get_folds(case_ids(args.cache_dir), n_folds=N_FOLDS, seed=FOLDS_SEED, splits_json=SPLITS_JSON)
        self.networks = {f: self._fold_networks(folds, f) for f in args.folds}
        self.params = sum(q.numel() for q in self.networks[args.folds[0]].pass1.parameters())
        self.n_cases = sum(len(n.cases) for n in self.networks.values())

    def _net(self, pattern, seed, fold):
        return load_net(os.path.join(ROOT, pattern.format(s=seed, f=fold)), self.device)

    def _fold_networks(self, folds, fold) -> FoldNetworks:
        args = self.args
        return FoldNetworks(self._net(args.pass1, SEED_OF_PASS1, fold),
                            [self._net(args.pass2, s, fold) for s in args.seeds],
                            fold_split(folds, fold)[1][:args.limit or None],
                            [] if args.no_plain else [self._net(args.pass1, s, fold) for s in args.seeds])

    def run(self, fold, img1, aff1, config: Configuration, clock):
        """One case; ends at the native mask in host memory, where nnU-Net's predictor ends too."""
        nets = self.networks[fold]
        if config.normalised:
            field, _ = segment(img1, aff1, nets.pass1, nets.pass2[:config.n_models], self.v0_ml, config.tta,
                               self.device, clock, self.fast, self.args.blend)
        else:
            field = segment_plain(img1, aff1, nets.plain[:config.n_models], config.tta, self.device, clock,
                                  self.fast)
        with clock.span("threshold + to host"):
            return (field < 0).cpu()

    def warm_up(self, config):
        """The first calls pay for CUDA initialisation and kernel selection."""
        first_fold = self.args.folds[0]
        for cid in self.networks[first_fold].cases[:self.args.warmup]:
            img1, aff1 = load_image(self.args.cache_dir, cid, 1)
            self.run(first_fold, img1.astype(np.float32), aff1, config, Clock())

    def measure(self, config: Configuration) -> dict:
        args, proc = self.args, self.proc
        self.warm_up(config)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rss_peak, rows, dices, repeat_medians = proc.memory_info().rss, [], [], []
        # Reading from disk and scoring are not the method's: their CPU time is added to the start, cancelling it.
        cpu_start = cpu_seconds(proc)

        for repeat in range(args.repeats):
            repeat_rows = []
            for fold in args.folds:
                for cid in self.networks[fold].cases:
                    before_reading = cpu_seconds(proc)
                    img1, aff1 = load_image(args.cache_dir, cid, 1)
                    img1 = img1.astype(np.float32)
                    cpu_start += cpu_seconds(proc) - before_reading
                    clock = Clock()
                    mask = self.run(fold, img1, aff1, config, clock)
                    repeat_rows.append(clock.t)
                    rss_peak = max(rss_peak, proc.memory_info().rss)

                    before_scoring = cpu_seconds(proc)
                    if repeat == 0:
                        label = np.asarray(load_label(args.cache_dir, cid)[0]).squeeze().astype(bool)
                        dices.append(mask_dice(mask.numpy(), label))
                    cpu_start += cpu_seconds(proc) - before_scoring
            repeat_medians.append(float(np.median([sum(r.values()) for r in repeat_rows])))
            rows += repeat_rows

        return self._result(config, rows, repeat_medians, dices, rss_peak, cpu_seconds(proc) - cpu_start)

    def _result(self, config, rows, repeat_medians, dices, rss_peak, cpu_s) -> dict:
        total_ms = np.array([sum(r.values()) for r in rows])
        net_ms = np.array([sum(v for k, v in r.items() if k.startswith("net:")) for r in rows])
        pass1_runs = 1 + (len(FLIPS) if config.tta and self.args.blend > 0 else 0) if config.normalised else 0
        return {
            "ms_median": float(np.median(total_ms)), "ms_mean": float(total_ms.mean()),
            "network_ms_median": float(np.median(net_ms)),
            "pipeline_ms_median": float(np.median(total_ms - net_ms)),
            "ms_p5": float(np.percentile(total_ms, 5)), "ms_p95": float(np.percentile(total_ms, 95)),
            "ms_median_per_repeat": repeat_medians,
            "n_timed": int(len(total_ms)),
            "stages_ms_median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]},
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / MB,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
            "ram_peak_mb": rss_peak / MB,
            "cpu_s_per_case": cpu_s / len(total_ms),
            "networks_run": pass1_runs + config.n_models * (len(FLIPS) if config.tta else 1),
            "parameters": self.params * (config.n_models + (1 if config.normalised else 0)),
            "dice_mean": float(np.mean(dices)),
        }


def print_result(name, r):
    print(f"\n{name}: {r['ms_median']:.0f} ms/case median ({r['ms_mean']:.0f} mean, "
          f"p5-p95 {r['ms_p5']:.0f}-{r['ms_p95']:.0f}, medians per repeat "
          f"{', '.join(f'{m:.0f}' for m in r['ms_median_per_repeat'])}; {r['n_timed']} timed), "
          f"GPU peak {r['gpu_peak_allocated_mb']:.0f} MB allocated / {r['gpu_peak_reserved_mb']:.0f} MB reserved, "
          f"RAM peak {r['ram_peak_mb']:.0f} MB, CPU {r['cpu_s_per_case']:.2f} s/case, "
          f"{r['networks_run']} network passes, {r['parameters']:,} params, Dice {r['dice_mean']:.4f}")
    print(f"   networks only {r['network_ms_median']:.1f} ms, pipeline around them "
          f"{r['pipeline_ms_median']:.1f} ms (medians per case)")
    print("   stages (median ms): " + ", ".join(f"{k} {v:.1f}" for k, v in r["stages_ms_median"].items()))


def main():
    args = parse_args()
    benchmark = Benchmark(args, torch.device("cuda"))
    configs = configurations(len(args.seeds), args.no_plain)
    print("second pass: x4 in 96^3, output sdf")
    print(f"GPU {torch.cuda.get_device_name(0)}  CPU {psutil.cpu_count(logical=False)} cores  "
          f"folds {args.folds}, {benchmark.n_cases} cases x {args.repeats} repeats  network params "
          f"{benchmark.params:,} each (pass 1 + 1 or 3 pass 2)  V0 {benchmark.v0_ml:.1f} mL")

    results = {}
    for name, config in configs.items():
        results[name] = benchmark.measure(config)
        print_result(name, results[name])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
