# Measures nnU-Net's inference cost on the same cases and machine as bench_pipeline.py.
#
#     conda activate nnunet-bench      # nnunetv2 is installed in its own environment
#     python scripts/bench_nnunet.py --fold 0 --out bench_nnunet.json
#
# The fold's trained 3d_fullres network (checkpoint_final.pth) predicts each validation
# volume, from the native array in memory to the native mask, through nnU-Net's own
# inference path: crop, z-score, sliding window (patch 112x160x128, step 0.5, Gaussian
# weighting), export. Run with and without its mirroring TTA (8 flips per window).
#
# Input: the 0.4 mm cache, which is already nnU-Net's target spacing (so it resamples
# nothing), transposed to the (z, y, x) order it was trained in. Dice is computed outside
# the timing, as a check that the network sees what it was trained on. Run it with the
# GPU otherwise idle.

import argparse
import json
import os
import time

import numpy as np
import psutil
import torch

# nnU-Net refuses to import without these set; inference does not use them.
for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(k, os.environ.get("TEMP", "."))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "nnunet/runs/grouped_250ep/nnUNet_results/Dataset501_FetalHead/"
                           "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres")


def main():
    p = argparse.ArgumentParser(description="Time nnU-Net inference on one fold.")
    p.add_argument("--folds", type=int, nargs="+", default=[0],
                   help="each fold's validation cases with that fold's network (0 1 2 3 4: all 90)")
    p.add_argument("--repeats", type=int, default=1,
                   help="timed passes over the cases; the spread between them shows the run-to-run noise")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases of each fold")
    p.add_argument("--warmup", type=int, default=2, help="untimed cases run first")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()

    # The same validation cases as ours (nnU-Net was trained on the same folds).
    folds = json.load(open(os.path.join(ROOT, "splits.json")))["folds"]
    proc = psutil.Process()
    results = {}

    for name, mirror in (("nnU-Net", False), ("nnU-Net + TTA", True)):
        # One predictor per fold, each with that fold's network.
        predictors = {}
        for f in a.folds:
            pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirror,
                                 perform_everything_on_device=True, device=torch.device("cuda"),
                                 verbose=False, verbose_preprocessing=False, allow_tqdm=False)
            pr.initialize_from_trained_model_folder(a.model, use_folds=(f,),
                                                    checkpoint_name="checkpoint_final.pth")
            predictors[f] = (pr, sorted(folds[f])[:a.limit or None])
        params = sum(q.numel() for q in predictors[a.folds[0]][0].network.parameters())

        # Predict one case; returns (milliseconds, CPU seconds, mask in (z, y, x) order).
        # Reading the case from disk happens before the clock starts.
        def run(predictor, cid):
            img = np.load(os.path.join(ROOT, "caches", "cache", "x1", f"{cid}.npz"))["img"]
            # (x, y, z) -> (z, y, x), plus a channel axis.
            x = np.ascontiguousarray(img.astype(np.float32).transpose(2, 1, 0))[None]
            torch.cuda.synchronize()
            cpu0 = sum(proc.cpu_times()[:2])                # user + system CPU seconds
            t0 = time.perf_counter()
            seg = predictor.predict_single_npy_array(x, {"spacing": [0.4, 0.4, 0.4]}, None, None, False)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e3, sum(proc.cpu_times()[:2]) - cpu0, seg

        # Warm-up: the first calls pay for CUDA initialisation and kernel selection.
        pr0, cases0 = predictors[a.folds[0]]
        for cid in cases0[:a.warmup]:
            run(pr0, cid)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rss_peak, times, dices, repeat_medians = proc.memory_info().rss, [], [], []
        cpu_s = 0.0
        for rep in range(a.repeats):
            rep_times = []
            for f in a.folds:
                predictor, cases = predictors[f]
                for cid in cases:
                    ms, cpu, seg = run(predictor, cid)
                    cpu_s += cpu
                    rep_times.append(ms)
                    rss_peak = max(rss_peak, proc.memory_info().rss)

                    # Dice against the label, in the same (z, y, x) order (first repeat only).
                    if rep == 0:
                        lab = np.load(os.path.join(ROOT, "caches", "cache", "label", f"{cid}.npz"))["label"].squeeze()
                        lab = lab.astype(bool).transpose(2, 1, 0)
                        mask = seg.astype(bool)
                        dices.append(float(2 * (mask & lab).sum() / (mask.sum() + lab.sum())))
                    print(f"  rep {rep} fold {f} {cid[:34]:<34} {ms:6.0f} ms", flush=True)
            repeat_medians.append(float(np.median(rep_times)))
            times += rep_times

        t = np.array(times)
        results[name] = {
            "ms_median": float(np.median(t)), "ms_mean": float(t.mean()),
            "ms_p5": float(np.percentile(t, 5)), "ms_p95": float(np.percentile(t, 95)),
            "ms_median_per_repeat": repeat_medians,
            "n_timed": int(len(t)),
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "ram_peak_mb": rss_peak / 2**20,
            "cpu_s_per_case": cpu_s / len(t),
            "parameters": params,
            "dice_mean": float(np.mean(dices)),
        }
        r = results[name]
        print(f"\n{name}: {r['ms_median']:.0f} ms/case median ({r['ms_mean']:.0f} mean, "
              f"p5-p95 {r['ms_p5']:.0f}-{r['ms_p95']:.0f}, medians per repeat "
              f"{', '.join(f'{m:.0f}' for m in repeat_medians)}; {r['n_timed']} timed), "
              f"GPU peak {r['gpu_peak_allocated_mb']:.0f} MB allocated / {r['gpu_peak_reserved_mb']:.0f} MB reserved, "
              f"RAM peak {r['ram_peak_mb']:.0f} MB, CPU {r['cpu_s_per_case']:.2f} s/case, "
              f"{params:,} params, Dice {r['dice_mean']:.4f}\n", flush=True)

        # Free the GPU before the next configuration.
        del predictors, pr0
        torch.cuda.empty_cache()

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
