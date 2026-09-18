"""Compare refiners by how much they actually know, not by one operating point.

Flip accuracy at threshold 0.5 conflates two things: how much information a refiner
has, and how willing it is to act on it. A model that rewrites more of the band will
show lower accuracy even if it is strictly better informed. So this sweeps the
threshold for each run, builds its (coverage, accuracy) curve, and reads both curves at
matched coverage. A run whose curve sits above another's everywhere knows more; curves
that cross mean the runs are differently calibrated, not differently informed.

Coverage here is the share of prior-wrong band voxels the refiner touches, so the two
axes are "how much of the error did it attempt" against "how often was it right".

    python phase2/compare.py --runs phase2/runs/cube64 phase2/runs/slab128x128x32
"""

import argparse
import os

import numpy as np

THRESHOLDS = np.concatenate([np.arange(0.05, 0.95, 0.025), [0.95]])


def curve(pred_dir: str):
    """(coverage, accuracy, net) per threshold, pooled over cases."""
    flips = np.zeros(len(THRESHOLDS))
    right = np.zeros(len(THRESHOLDS))
    opportunity = 0

    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(pred_dir, f))
        band = d["band"].astype(bool)
        mask = d["mask"].astype(bool)[band]
        coarse = (d["coarse"].astype(np.float32) >= 0.5)[band]
        prob = d["refined"].astype(np.float32)[band]
        opportunity += int((coarse != mask).sum())

        for i, t in enumerate(THRESHOLDS):
            refined = prob >= t
            flipped = refined != coarse
            flips[i] += int(flipped.sum())
            right[i] += int((flipped & (refined == mask)).sum())

    coverage = flips / max(opportunity, 1)
    accuracy = np.divide(right, flips, out=np.full_like(right, np.nan), where=flips > 0)
    net = (2 * right - flips) / max(opportunity, 1)
    return coverage, accuracy, net, opportunity


def confidence(pred_dir: str) -> dict:
    """How decisive are the probabilities inside the band? A better-informed refiner
    should put less mass near 0.5, where it is effectively abstaining."""
    undecided = total = 0
    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(pred_dir, f))
        p = d["refined"].astype(np.float32)[d["band"].astype(bool)]
        undecided += int(((p > 0.35) & (p < 0.65)).sum())
        total += p.size
    return {"undecided_pct": 100.0 * undecided / max(total, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=str, nargs="+", required=True,
                   help="run directories, each holding preds_val/")
    p.add_argument("--at", type=float, nargs="+",
                   default=[0.25, 0.30, 0.40, 0.50, 0.70],
                   help="coverage levels to read the curves at")
    p.add_argument("--tol", type=float, default=0.03,
                   help="how close a threshold's coverage must be to count at a level")
    a = p.parse_args()

    results = {}
    for run in a.runs:
        pred_dir = os.path.join(run, "preds_val")
        if not os.path.isdir(pred_dir):
            print(f"skipping {run}: no preds_val/")
            continue
        cov, acc, net, opp = curve(pred_dir)
        results[os.path.basename(run.rstrip("/\\"))] = (cov, acc, net, opp,
                                                        confidence(pred_dir))

    if not results:
        return

    print("Accuracy of flips, read at matched coverage")
    print("(coverage = share of prior-wrong band voxels the refiner touched)\n")
    header = f"{'coverage':>10}" + "".join(f"{n[:18]:>20}" for n in results)
    print(header)
    print("-" * len(header))
    for target in a.at:
        row = f"{100*target:>9.0f}%"
        for cov, acc, _net, _opp, _c in results.values():
            # Coverage is not monotone in threshold — it is U-shaped, minimal near 0.45
            # and rising towards both extremes, so two thresholds reach any given
            # coverage with very different accuracy. Interpolating over sorted coverage
            # silently mixes the two branches. Take the upper envelope instead: the best
            # accuracy actually achievable at that coverage.
            near = np.abs(cov - target) <= a.tol
            value = np.nanmax(acc[near]) if near.any() and np.isfinite(acc[near]).any() else np.nan
            row += f"{100*value:>19.1f}%" if np.isfinite(value) else f"{'-':>20}"
        print(row)

    print(f"\n{'':>10}" + "".join(f"{n[:18]:>20}" for n in results))
    print("-" * len(header))
    for label, pick in (("best net fixed", lambda c, ac, n: f"{100*np.nanmax(n):.1f}%"),
                        ("at its best thr", lambda c, ac, n: f"{100*ac[np.nanargmax(n)]:.1f}% acc"),
                        ("coverage there", lambda c, ac, n: f"{100*c[np.nanargmax(n)]:.1f}%")):
        row = f"{label:>10}"
        for cov, acc, net, _opp, _c in results.values():
            row += f"{pick(cov, acc, net):>20}"
        print(row)

    row = f"{'undecided':>10}"
    for *_rest, conf in results.values():
        row += f"{conf['undecided_pct']:>19.1f}%"
    print(row + "   (band voxels with p in 0.35-0.65)")


if __name__ == "__main__":
    main()
