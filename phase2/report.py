"""Per-fold report: what the refiner did, and what it was allowed to reach.

Pairs each fold's segmentation metrics with the surface coverage of the band it was
restricted to, because the two only mean something together. A fold that gains little
with 99.9% coverage is a refiner problem; one that gains little with 94% coverage was
never allowed to fix the rest.

    python phase2/report.py --runs phase2/runs/band5_f{fold} --crops-dir phase2/crops/band5
    python phase2/report.py --runs phase2/runs/band{w}_f{fold} --widths 3 5 7.5 10
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from splits import fold_split, get_folds  # noqa: E402

TRUNC_MM = 10.0


def surface_coverage(crops_dir: str, case_ids: list[str],
                     trunc_mm: float = TRUNC_MM) -> list[float]:
    """Fraction of each case's true surface lying inside its band.

    Surface voxels, not volume: a band that misses a thin strip of boundary loses that
    strip whatever it contains, and volume is dominated by an interior every method
    already gets right. Measuring volume put this at 0.065%; measuring surface put the
    same 5mm band at 1.65%.
    """
    out = []
    for cid in case_ids:
        path = os.path.join(crops_dir, f"{cid}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            prior = d["sdf_prior"].astype(np.float32)
            mask = d["mask"].astype(bool)
            band = d["band"].astype(bool)

        surface = mask & ~binary_erosion(mask)
        if not surface.any():
            continue
        out.append(float(band[surface].mean()))
    return out


def fold_row(run_dir: str, crops_dir: str, case_ids: list[str]) -> dict | None:
    summary = os.path.join(run_dir, "eval_val", "summary.json")
    if not os.path.exists(summary):
        return None
    a = json.load(open(summary))["aggregate"]
    cov = surface_coverage(crops_dir, case_ids) if crops_dir else []

    return {
        "coarse": a["dice_coarse"]["mean"],
        "refined": a["dice_refined"]["mean"],
        "delta": a["dice_refined"]["mean"] - a["dice_coarse"]["mean"],
        "hd95": a["hd95_refined"]["mean"],
        "asd": a["asd_refined"]["mean"],
        "cov_mean": float(np.mean(cov)) if cov else float("nan"),
        "cov_min": float(np.min(cov)) if cov else float("nan"),
    }


def print_table(label: str, rows: dict[int, dict]) -> dict | None:
    if not rows:
        print(f"{label}: no results yet")
        return None

    print(f"\n=== {label} ===")
    print(f"{'fold':<6}{'coarse':>9}{'refined':>10}{'delta':>9}"
          f"{'cov mean':>11}{'cov min':>10}{'HD95':>8}{'ASD':>8}")
    print("-" * 71)
    for f in sorted(rows):
        r = rows[f]
        print(f"{f:<6}{r['coarse']:9.4f}{r['refined']:10.4f}{r['delta']:+9.4f}"
              f"{100*r['cov_mean']:10.2f}%{100*r['cov_min']:9.2f}%"
              f"{r['hd95']:8.2f}{r['asd']:8.3f}")
    print("-" * 71)

    agg = {k: np.array([rows[f][k] for f in sorted(rows)])
           for k in ("coarse", "refined", "delta", "cov_mean", "cov_min", "hd95", "asd")}
    print(f"{'mean':<6}{agg['coarse'].mean():9.4f}{agg['refined'].mean():10.4f}"
          f"{agg['delta'].mean():+9.4f}{100*agg['cov_mean'].mean():10.2f}%"
          f"{100*agg['cov_min'].min():9.2f}%{agg['hd95'].mean():8.2f}"
          f"{agg['asd'].mean():8.3f}")
    print(f"{'std':<6}{agg['coarse'].std():9.4f}{agg['refined'].std():10.4f}"
          f"{agg['delta'].std():9.4f}")
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=str, required=True,
                   help="run directory pattern with {fold} and optionally {w}")
    p.add_argument("--crops-dir", type=str, default=None,
                   help="crop directory pattern; {w} allowed")
    p.add_argument("--widths", type=float, nargs="*", default=None,
                   help="band widths to compare; omit for a single configuration")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--splits-json", type=str,
                   default=os.path.join(REPO_ROOT, "splits.json"))
    a = p.parse_args()

    folds = get_folds(n_folds=a.n_folds, splits_json=a.splits_json)
    summaries = {}

    for w in (a.widths or [None]):
        rows = {}
        for f in range(a.n_folds):
            run = a.runs.format(fold=f, w=w)
            crops = a.crops_dir.format(fold=f, w=w) if a.crops_dir else None
            _, val = fold_split(folds, f)
            r = fold_row(run, crops, val)
            if r:
                rows[f] = r
        label = f"band {w} mm" if w is not None else "results"
        agg = print_table(label, rows)
        if agg is not None:
            summaries[w] = agg

    if len(summaries) > 1:
        print(f"\n\n=== band width against what it buys ===")
        print(f"{'band':<10}{'coverage':>11}{'worst':>9}{'coarse':>9}"
              f"{'refined':>10}{'delta':>9}{'HD95':>8}{'ASD':>8}")
        print("-" * 74)
        for w in sorted(summaries):
            s = summaries[w]
            print(f"{w:<10.1f}{100*s['cov_mean'].mean():10.2f}%"
                  f"{100*s['cov_min'].min():8.2f}%{s['coarse'].mean():9.4f}"
                  f"{s['refined'].mean():10.4f}{s['delta'].mean():+9.4f}"
                  f"{s['hd95'].mean():8.2f}{s['asd'].mean():8.3f}")
        print("-" * 74)
        print("\n  If refined Dice climbs with coverage, the band was the ceiling and")
        print("  the effort belongs in localisation. If it flattens while coverage")
        print("  still rises, the refiner is the limit and the band was never it.")


if __name__ == "__main__":
    main()
