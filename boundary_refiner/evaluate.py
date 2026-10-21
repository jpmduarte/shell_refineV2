"""Score predict.py's refined output against the ground truth, next to the first pass's coarse mask.

Three Dice per case: coarse (the first pass alone, the baseline), refined (after the band is
rewritten: the number that matters, since a boundary outside the band counts as the error it is)
and band (only where the refiner may act, which flatters it).
"""
import argparse
import json
import os

import numpy as np
import torch

from common.metrics import asd, dice, hd95

RULE = "-" * 110
USAGE = """\
example:
  python boundary_refiner/evaluate.py --pred-dir boundary_refiner/runs/fold0/preds_val \\
      --out-dir boundary_refiner/runs/fold0/eval_val"""


def evaluate(pred_dir: str, threshold: float) -> list[dict]:
    results = []
    for fname in sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz")):
        prediction = np.load(os.path.join(pred_dir, fname))
        refined = prediction["refined"].astype(np.float32)
        coarse = prediction["coarse"].astype(np.float32)
        band = prediction["band"].astype(bool)
        mask = prediction["mask"].astype(np.float32)
        spacing = tuple(prediction["spacing"].tolist())

        refined_t, coarse_t, mask_t = (torch.from_numpy(a) for a in (refined, coarse, mask))
        band_t = torch.from_numpy(band)

        refined_bin = refined >= threshold
        coarse_bin = coarse.astype(bool)
        mask_bin = mask.astype(bool)

        results.append({
            "case": os.path.splitext(fname)[0],
            "coarse": dice(coarse_t, mask_t, threshold=threshold),
            "refined": dice(refined_t, mask_t, threshold=threshold),
            "band": dice(refined_t[band_t], mask_t[band_t], threshold=threshold),
            "hd95_coarse": hd95(coarse_bin, mask_bin, spacing),
            "hd95_refined": hd95(refined_bin, mask_bin, spacing),
            "asd_coarse": asd(coarse_bin, mask_bin, spacing),
            "asd_refined": asd(refined_bin, mask_bin, spacing),
            # The error the refiner could not fix: label and prior disagreeing outside the band.
            "gt_outside_band_pct": round(100.0 * float((mask_bin != coarse_bin)[~band].mean()), 4),
        })
    return results


def fmt(value, spec=".4f"):
    return f"{value:{spec}}" if value is not None else "   skip"


def summary_lines(results, columns) -> list[str]:
    lines = [f"{'Case':<44}  {'coarse':>7}  {'refined':>7}  {'band':>7}"
             f"  {'hd95_c':>7}  {'hd95_r':>7}  {'asd_c':>6}  {'asd_r':>6}  {'gt_out%':>7}", RULE]
    for r in results:
        lines.append(f"  {r['case'][:42]:<42}  {fmt(r['coarse'])}  {fmt(r['refined'])}  {fmt(r['band'])}"
                     f"  {fmt(r['hd95_coarse'], '7.2f')}  {fmt(r['hd95_refined'], '7.2f')}"
                     f"  {fmt(r['asd_coarse'], '6.2f')}  {fmt(r['asd_refined'], '6.2f')}"
                     f"  {r['gt_outside_band_pct']:7.3f}")
    lines.append(RULE)

    coarse, refined, band = columns["coarse"], columns["refined"], columns["band"]
    hd95_c, hd95_r = columns["hd95_coarse"], columns["hd95_refined"]
    asd_c, asd_r = columns["asd_coarse"], columns["asd_refined"]
    if refined:
        lines += [f"\n  Mean Dice coarse  : {np.mean(coarse):.4f}",
                  f"  Mean Dice refined : {np.mean(refined):.4f}  (delta {np.mean(refined) - np.mean(coarse):+.4f})",
                  f"  Mean Dice in band : {np.mean(band):.4f}",
                  f"  Std  Dice refined : {np.std(refined):.4f}",
                  f"  Min  Dice refined : {np.min(refined):.4f}"]
    if hd95_r:
        lines += [f"\n  Mean HD95 coarse  : {np.mean(hd95_c):.2f} mm",
                  f"  Mean HD95 refined : {np.mean(hd95_r):.2f} mm  (delta {np.mean(hd95_r) - np.mean(hd95_c):+.2f})"]
    if asd_r:
        lines += [f"\n  Mean ASD  coarse  : {np.mean(asd_c):.2f} mm",
                  f"  Mean ASD  refined : {np.mean(asd_r):.2f} mm  (delta {np.mean(asd_r) - np.mean(asd_c):+.2f})"]

    out_pct = columns["gt_outside_band_pct"]
    lines += ["\n  GT-vs-prior disagreement outside the band (unreachable):",
              f"    mean {np.mean(out_pct):.3f}%   worst {np.max(out_pct):.3f}%"]
    return lines


def aggregate(values):
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = evaluate(args.pred_dir, args.threshold)
    keys = ["coarse", "refined", "band", "hd95_coarse", "hd95_refined", "asd_coarse", "asd_refined"]
    columns = {key: [r[key] for r in results if r[key] is not None] for key in keys}
    columns["gt_outside_band_pct"] = [r["gt_outside_band_pct"] for r in results]

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines(results, columns):
            print(line)
            f.write(line + "\n")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "pred_dir": args.pred_dir,
            "threshold": args.threshold,
            "cases": results,
            "aggregate": {
                "dice_coarse": aggregate(columns["coarse"]), "dice_refined": aggregate(columns["refined"]),
                "dice_band": aggregate(columns["band"]),
                "hd95_coarse": aggregate(columns["hd95_coarse"]), "hd95_refined": aggregate(columns["hd95_refined"]),
                "asd_coarse": aggregate(columns["asd_coarse"]), "asd_refined": aggregate(columns["asd_refined"]),
                "gt_outside_band_pct": aggregate(columns["gt_outside_band_pct"]),
            },
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
