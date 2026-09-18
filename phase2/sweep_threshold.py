"""Is the refiner just being conservative?

It only flips 27% of the voxels the prior got wrong; the rest never cross 0.5. If that
is conservatism rather than ignorance, some other threshold should touch more of the
error without the flips getting less accurate. If accuracy falls as soon as it touches
more, the extra voxels were ones it had no opinion about, and the ceiling is
information, not calibration.

Only band voxels are affected: outside the band `refined` is already binary (phase 1's
mask), so thresholding it changes nothing there.

    python phase2/sweep_threshold.py --pred-dir phase2/runs/fold0_masked/preds_val
"""

import argparse
import os

import numpy as np


def sweep_case(path: str, thresholds) -> dict:
    d = np.load(path)
    band    = d["band"].astype(bool)
    mask    = d["mask"].astype(bool)[band]
    coarse  = (d["coarse"].astype(np.float32) >= 0.5)[band]
    prob    = d["refined"].astype(np.float32)[band]

    # Dice needs the whole volume, not just the band: the interior dominates it and is
    # identical across thresholds, so carry its counts once.
    full_mask   = d["mask"].astype(bool)
    full_coarse = d["coarse"].astype(np.float32) >= 0.5
    out_band_inter = int((full_coarse & full_mask & ~band).sum())
    out_band_pred  = int((full_coarse & ~band).sum())
    gt_total       = int(full_mask.sum())

    per_t = {}
    for t in thresholds:
        refined = prob >= t
        flipped    = refined != coarse
        to_correct = int((flipped & (refined == mask)).sum())
        to_wrong   = int((flipped & (refined != mask)).sum())
        inter = out_band_inter + int((refined & mask).sum())
        pred  = out_band_pred + int(refined.sum())
        per_t[t] = {"flipped": int(flipped.sum()), "to_correct": to_correct,
                    "to_wrong": to_wrong, "inter": inter, "pred": pred}

    return {"per_t": per_t, "gt_total": gt_total,
            "opportunity": int((coarse != mask).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=str, required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
    a = p.parse_args()

    files = sorted(f for f in os.listdir(a.pred_dir) if f.endswith(".npz"))
    cases = [sweep_case(os.path.join(a.pred_dir, f), a.thresholds) for f in files]

    opportunity = sum(c["opportunity"] for c in cases)
    print(f"{len(files)} cases   {opportunity:,} band voxels wrong in the prior\n")
    print(f"{'thr':>6} {'flipped':>12} {'%of wrong':>10} {'acc':>8} {'net':>12} "
          f"{'%fixed':>8} {'Dice':>8}")
    print("-" * 70)

    for t in a.thresholds:
        flipped = sum(c["per_t"][t]["flipped"] for c in cases)
        corr    = sum(c["per_t"][t]["to_correct"] for c in cases)
        wrong   = sum(c["per_t"][t]["to_wrong"] for c in cases)
        # Dice averaged over cases, not pooled — matches how evaluate.py reports it.
        dices = [2.0 * c["per_t"][t]["inter"] / (c["per_t"][t]["pred"] + c["gt_total"])
                 for c in cases]
        acc = corr / flipped if flipped else float("nan")
        net = corr - wrong
        print(f"{t:>6.2f} {flipped:>12,} {100*flipped/opportunity:>9.1f}% "
              f"{100*acc:>7.1f}% {net:>+12,} {100*net/opportunity:>7.1f}% "
              f"{np.mean(dices):>8.4f}")


if __name__ == "__main__":
    main()
