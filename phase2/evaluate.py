"""Phase 2 metrics: reads predict.py's stitched output and scores it against the
ground truth, next to phase 1's coarse mask so the refinement's actual contribution is
visible rather than implied.

Three Dice numbers per case, because they answer different questions:
    coarse   phase 1 alone, upsampled  — the baseline phase 2 has to beat
    refined  after phase 2 rewrites the band — the number that matters
    band     Dice restricted to the band — where refinement was allowed to act

Reporting `band` alone would flatter the model: it is the region it was trained on and
(with --band-masked-loss) scored on. `refined` is the honest number, because a boundary
that truly sits outside the band is counted as the error it is.

    python phase2/evaluate.py --pred-dir phase2/runs/fold0/preds_val \\
        --out-dir phase2/runs/fold0/eval_val
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from metrics import asd, dice, hd95  # noqa: E402


def evaluate(pred_dir: str, threshold: float) -> list[dict]:
    results = []
    for fname in sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz")):
        d = np.load(os.path.join(pred_dir, fname))
        refined = d["refined"].astype(np.float32)
        coarse  = d["coarse"].astype(np.float32)
        band    = d["band"].astype(bool)
        mask    = d["mask"].astype(np.float32)
        spacing = tuple(d["spacing"].tolist())

        refined_t, coarse_t, mask_t = (torch.from_numpy(a) for a in (refined, coarse, mask))
        band_t = torch.from_numpy(band)

        refined_bin = refined >= threshold
        coarse_bin  = coarse.astype(bool)
        mask_bin    = mask.astype(bool)

        results.append({
            "case":         fname[:-4],
            "coarse":       dice(coarse_t,  mask_t, threshold=threshold),
            "refined":      dice(refined_t, mask_t, threshold=threshold),
            "band":         dice(refined_t[band_t], mask_t[band_t], threshold=threshold),
            "hd95_coarse":  hd95(coarse_bin,  mask_bin, spacing),
            "hd95_refined": hd95(refined_bin, mask_bin, spacing),
            "asd_coarse":   asd(coarse_bin,  mask_bin, spacing),
            "asd_refined":  asd(refined_bin, mask_bin, spacing),
            # How much of the true boundary error phase 2 could not have fixed, because
            # the ground truth disagreed with the prior outside the band it may touch.
            "gt_outside_band_pct": round(
                100.0 * float((mask_bin != coarse_bin)[~band].mean()), 4),
        })
    return results


def print_and_write(line: str, f):
    print(line)
    f.write(line + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir",  type=str, required=True)
    p.add_argument("--out-dir",   type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    results = evaluate(a.pred_dir, a.threshold)

    def column(key):
        return [r[key] for r in results if r[key] is not None]

    summary_path = os.path.join(a.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        print_and_write(f"{'Case':<44}  {'coarse':>7}  {'refined':>7}  {'band':>7}"
                        f"  {'hd95_c':>7}  {'hd95_r':>7}  {'asd_c':>6}  {'asd_r':>6}"
                        f"  {'gt_out%':>7}", f)
        print_and_write("-" * 110, f)
        for r in results:
            def fmt(key, spec=".4f"):
                v = r[key]
                return f"{v:{spec}}" if v is not None else "   skip"
            print_and_write(
                f"  {r['case'][:42]:<42}  {fmt('coarse')}  {fmt('refined')}  {fmt('band')}"
                f"  {fmt('hd95_coarse', '7.2f')}  {fmt('hd95_refined', '7.2f')}"
                f"  {fmt('asd_coarse', '6.2f')}  {fmt('asd_refined', '6.2f')}"
                f"  {r['gt_outside_band_pct']:7.3f}", f)
        print_and_write("-" * 110, f)

        coarse, refined, band = column("coarse"), column("refined"), column("band")
        hd95_c, hd95_r = column("hd95_coarse"), column("hd95_refined")
        asd_c,  asd_r  = column("asd_coarse"),  column("asd_refined")

        if refined:
            print_and_write(f"\n  Mean Dice coarse  : {np.mean(coarse):.4f}", f)
            print_and_write(f"  Mean Dice refined : {np.mean(refined):.4f}"
                            f"  (delta {np.mean(refined) - np.mean(coarse):+.4f})", f)
            print_and_write(f"  Mean Dice in band : {np.mean(band):.4f}", f)
            print_and_write(f"  Std  Dice refined : {np.std(refined):.4f}", f)
            print_and_write(f"  Min  Dice refined : {np.min(refined):.4f}", f)
        if hd95_r:
            print_and_write(f"\n  Mean HD95 coarse  : {np.mean(hd95_c):.2f} mm", f)
            print_and_write(f"  Mean HD95 refined : {np.mean(hd95_r):.2f} mm"
                            f"  (delta {np.mean(hd95_r) - np.mean(hd95_c):+.2f})", f)
        if asd_r:
            print_and_write(f"\n  Mean ASD  coarse  : {np.mean(asd_c):.2f} mm", f)
            print_and_write(f"  Mean ASD  refined : {np.mean(asd_r):.2f} mm"
                            f"  (delta {np.mean(asd_r) - np.mean(asd_c):+.2f})", f)

        out_pct = [r["gt_outside_band_pct"] for r in results]
        print_and_write(f"\n  GT-vs-prior disagreement outside the band (unreachable):", f)
        print_and_write(f"    mean {np.mean(out_pct):.3f}%   worst {np.max(out_pct):.3f}%", f)

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None

    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump({
            "pred_dir": a.pred_dir,
            "threshold": a.threshold,
            "cases": results,
            "aggregate": {
                "dice_coarse": agg(coarse), "dice_refined": agg(refined),
                "dice_band": agg(band),
                "hd95_coarse": agg(hd95_c), "hd95_refined": agg(hd95_r),
                "asd_coarse": agg(asd_c),  "asd_refined": agg(asd_r),
                "gt_outside_band_pct": agg(out_pct),
            },
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
