"""Does phase 2 actually change the band, or does it learn to copy the prior?

A +0.002 Dice gain has two very different explanations, and the fix differs for each:
    - the refiner rewrites a lot and gets roughly as much wrong as right
    - the refiner barely rewrites anything, because reproducing its input is the
      safest thing it can do

This counts, inside the band, how many voxels flip between the coarse mask and the
refined one, and whether each flip moved toward the ground truth or away from it.

    python phase2/diagnose.py --pred-dir phase2/runs/fold0_masked/preds_val
"""

import argparse
import os

import numpy as np


def diagnose_case(path: str, threshold: float) -> dict:
    d = np.load(path)
    band    = d["band"].astype(bool)
    mask    = d["mask"].astype(bool)
    coarse  = d["coarse"].astype(np.float32) >= threshold
    refined = d["refined"].astype(np.float32) >= threshold

    in_band = band.sum()
    coarse_wrong = (coarse != mask) & band          # what there was to fix
    flipped      = (coarse != refined) & band       # what the refiner touched

    to_correct = flipped & (refined == mask)        # flip landed on the truth
    to_wrong   = flipped & (refined != mask)        # flip left the truth

    return {
        "case":          os.path.basename(path)[:-4],
        "band":          int(in_band),
        "opportunity":   int(coarse_wrong.sum()),
        "flipped":       int(flipped.sum()),
        "to_correct":    int(to_correct.sum()),
        "to_wrong":      int(to_wrong.sum()),
        "net":           int(to_correct.sum() - to_wrong.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir",  type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    rows = [diagnose_case(os.path.join(a.pred_dir, f), a.threshold)
            for f in sorted(os.listdir(a.pred_dir)) if f.endswith(".npz")]

    print(f"{'Case':<40} {'band':>10} {'wrong':>9} {'flipped':>9} "
          f"{'->right':>9} {'->wrong':>9} {'net':>9}  {'%fixed':>7}")
    print("-" * 108)
    for r in rows:
        pct = 100.0 * r["net"] / r["opportunity"] if r["opportunity"] else 0.0
        print(f"{r['case'][:38]:<40} {r['band']:>10,} {r['opportunity']:>9,} "
              f"{r['flipped']:>9,} {r['to_correct']:>9,} {r['to_wrong']:>9,} "
              f"{r['net']:>+9,}  {pct:>6.1f}%")
    print("-" * 108)

    tot = {k: sum(r[k] for r in rows) for k in ("band", "opportunity", "flipped",
                                                 "to_correct", "to_wrong", "net")}
    print(f"\n  band voxels                 {tot['band']:>12,}")
    print(f"  wrong in coarse (to fix)    {tot['opportunity']:>12,}"
          f"   ({100*tot['opportunity']/tot['band']:.2f}% of band)")
    print(f"  touched by the refiner      {tot['flipped']:>12,}"
          f"   ({100*tot['flipped']/tot['band']:.2f}% of band,"
          f" {100*tot['flipped']/tot['opportunity']:.1f}% of what was wrong)")
    print(f"    of those, -> correct      {tot['to_correct']:>12,}"
          f"   ({100*tot['to_correct']/tot['flipped']:.1f}% of flips)")
    print(f"    of those, -> wrong        {tot['to_wrong']:>12,}"
          f"   ({100*tot['to_wrong']/tot['flipped']:.1f}% of flips)")
    print(f"  net voxels corrected        {tot['net']:>+12,}"
          f"   ({100*tot['net']/tot['opportunity']:.1f}% of what was wrong)")


if __name__ == "__main__":
    main()
