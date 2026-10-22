"""Does a better head-volume estimate improve the size normalisation? The volume fed to pass 2, estimated several ways.

Everything else fixed (pass 2: one model, no TTA, each seed its own fold's networks), all 90 cases, 3 seeds:
    single     pass 1 once (the deployed method)
    tta        pass 1 averaged over its 8 mirrorings
    ens_tta    pass 1 of the 3 seeds, each with TTA, averaged (same fold: no leak)
    iter1/2    from single, re-estimate on pass 2's own output and run pass 2 again, once or twice
    ens_iter1  ens_tta, then one such iteration
    oracle     the label's volume: the best any estimate can do
Pass 2 always resamples from the native volume (no double interpolation).
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from common.folds import N_FOLDS, SEED as FOLDS_SEED, fold_split, get_folds
from common.geometry import FACTOR, MM3_PER_ML, isotropic_spacing, voxel_spacing
from common.image_cache import case_ids, load as load_image, load_label
from common.metrics import hd95, mask_dice
from datasets import add_dataset_argument, get as get_dataset
from size_normalization.build_cache import reference_v0
from size_normalization.infer import forward, load_net, to_grid_fast, to_native_fast, volume_ml, zscore

SEEDS = (42, 43, 44)
VARIANTS = ("single", "tta", "ens_tta", "iter1", "iter2", "ens_iter1", "oracle")


@torch.no_grad()
def pass2(x1, aff1, net, vol_ml, v0_ml):
    """(the SDF on the native grid, the volume measured on pass 2's own grid) for a given head volume."""
    k = (vol_ml / v0_ml) ** (1 / 3)
    normalised_spacing = FACTOR * isotropic_spacing(aff1) * k
    xn, affn = to_grid_fast(x1, aff1, normalised_spacing)
    field = forward(net, xn, False)
    return to_native_fast(field, affn, aff1, x1.shape), volume_ml(field, normalised_spacing)


def case_rows(cid, fold, x1, aff1, label, pass1, norm, v0_ml):
    """One row per seed: Dice, HD95 and volume error of every variant."""
    spacing = voxel_spacing(aff1)
    true_ml = float(label.sum()) * float(spacing[0]) ** 3 / MM3_PER_ML
    spacing4 = FACTOR * float(spacing[0])
    x4, _ = to_grid_fast(x1, aff1, spacing4)
    with torch.no_grad():
        single_and_tta = {s: (forward(pass1[s], x4, False), forward(pass1[s], x4, True)) for s in SEEDS}
    ensemble_ml = volume_ml(torch.stack([single_and_tta[s][1] for s in SEEDS]).mean(0), spacing4)

    rows = []
    for s in SEEDS:
        single, tta = single_and_tta[s]
        volumes = {"single": volume_ml(single, spacing4), "tta": volume_ml(tta, spacing4),
                   "ens_tta": ensemble_ml, "oracle": true_ml}
        results = {name: pass2(x1, aff1, norm[s], volumes[name], v0_ml) for name in volumes}
        for name, start in (("iter1", "single"), ("iter2", "iter1"), ("ens_iter1", "ens_tta")):
            volumes[name] = results[start][1]
            results[name] = pass2(x1, aff1, norm[s], volumes[name], v0_ml)

        row = {"case": cid, "fold": fold, "seed": s, "true_ml": true_ml}
        for name in VARIANTS:
            mask = (results[name][0] < 0).cpu().numpy()
            row[name] = {"dice": mask_dice(mask, label), "hd95": hd95(mask, label, spacing),
                         "vol_err": volumes[name] / true_ml - 1}
        rows.append(row)
    return rows


def print_table(rows):
    """Per case the mean over the seeds, then over the cases."""
    cases = sorted({r["case"] for r in rows})
    per_case = {v: np.array([np.mean([r[v]["dice"] for r in rows if r["case"] == c]) for c in cases])
                for v in VARIANTS}
    hd = {v: np.mean([r[v]["hd95"] for r in rows]) for v in VARIANTS}
    print("\n| Volume fed to pass 2 | Dice | vs single | p | HD95 | Volume error, median abs | signed mean | max abs |")
    print("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        error = np.array([r[v]["vol_err"] for r in rows])
        p_value = "—" if v == "single" else f"{wilcoxon(per_case[v], per_case['single']).pvalue:.2g}"
        print(f"| {v} | {per_case[v].mean():.4f} | {per_case[v].mean() - per_case['single'].mean():+.4f} | "
              f"{p_value} | {hd[v]:.2f} | {100 * np.median(abs(error)):.1f}% | {100 * error.mean():+.1f}% | "
              f"{100 * abs(error).max():.1f}% |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_argument(parser)
    parser.add_argument("--cache-dir", default=None, help="default: the dataset's cache")
    parser.add_argument("--reference", default=None, help="default: the dataset's size-normalised cache")
    parser.add_argument("--out", default=None, help="default: <the dataset's runs>/pass1_volume_test.json")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.cache_dir = args.cache_dir or dataset.cache
    args.reference = args.reference or dataset.norm_cache
    args.out = args.out or os.path.join(dataset.runs, "pass1_volume_test.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v0_ml = reference_v0(args.reference)
    folds = get_folds(dataset, case_ids(args.cache_dir), n_folds=N_FOLDS, seed=FOLDS_SEED)

    rows = []
    for fold in range(N_FOLDS):
        pass1 = {s: load_net(dataset.checkpoint(dataset.pass1_run, s, fold), device) for s in SEEDS}
        norm = {s: load_net(dataset.checkpoint(dataset.pass2_run, s, fold), device) for s in SEEDS}
        for cid in fold_split(folds, fold)[1]:
            img1, aff1 = load_image(args.cache_dir, cid, 1)
            label = np.asarray(load_label(args.cache_dir, cid)[0]).squeeze().astype(bool)
            x1 = zscore(np.asarray(img1).squeeze(), device)
            rows += case_rows(cid, fold, x1, aff1, label, pass1, norm, v0_ml)
            torch.cuda.empty_cache()
            latest = rows[-len(SEEDS):]
            print(f"fold {fold} {cid[:34]:<34} single {np.mean([r['single']['dice'] for r in latest]):.4f}  "
                  f"ens_iter1 {np.mean([r['ens_iter1']['dice'] for r in latest]):.4f}  "
                  f"oracle {np.mean([r['oracle']['dice'] for r in latest]):.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print_table(rows)
    print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
