"""Metrics only: reads predictions already saved by predict.py and scores them two
ways, both at native resolution.

    - MAE + Dice-from-sign against the true native SDF (cache/sdf/x1) — how good is
      the upsampled field itself.
    - Dice + HD95 + ASD against the real, never-resampled ground truth label
      (cache/label) — the number that reflects actual end-user accuracy.

Does not run the model or upsample anything itself — that is predict.py's job.

    python phase1/predict.py --checkpoint phase1/runs/fold0/checkpoints/phase1_best.pth \\
        --fold 0 --split val --out-dir phase1/runs/fold0/preds_val
    python phase1/evaluate.py --pred-dir phase1/runs/fold0/preds_val --out-dir phase1/runs/fold0/eval_val
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import load_label  # noqa: E402
from make_sdf import load as load_sdf_target  # noqa: E402
from metrics import asd, dice, dice_from_sdf, hd95, mae  # noqa: E402


def evaluate(pred_dir: str, cache_dir: str) -> list[dict]:
    results = []
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz"))

    for fname in files:
        cid = fname[:-4]
        d = np.load(os.path.join(pred_dir, fname))
        pred_sdf, affine = d["sdf"], d["affine"]

        target_sdf, _ = load_sdf_target(cache_dir, cid, 1)
        pred_t   = torch.from_numpy(pred_sdf).float()
        target_t = torch.from_numpy(target_sdf.squeeze()).float()

        label, _ = load_label(cache_dir, cid)
        label = (label[0] if label.ndim == 4 else label).astype(bool)
        pred_mask = pred_sdf < 0
        spacing = np.abs(np.diag(affine[:3, :3]))

        results.append({
            "case": cid,
            "mae":            mae(pred_t, target_t),
            "dice_sdf":       dice_from_sdf(pred_t, target_t),
            "dice_label":     dice(torch.from_numpy(pred_mask.astype(np.float32)),
                                   torch.from_numpy(label.astype(np.float32))),
            "hd95":           hd95(pred_mask, label, spacing),
            "asd":            asd(pred_mask, label, spacing),
        })

    return results


def print_and_write(line: str, f):
    print(line)
    f.write(line + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir",  type=str, required=True, help="directory of predict.py output")
    p.add_argument("--cache-dir", type=str, default=os.path.join(PARENT, "cache"))
    p.add_argument("--out-dir",   type=str, required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    results = evaluate(a.pred_dir, a.cache_dir)
    valid_sdf   = [r for r in results if r["dice_sdf"]   is not None]
    valid_label = [r for r in results if r["dice_label"] is not None]

    summary_path = os.path.join(a.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        print_and_write(f"{'Case':<44}  {'MAE':>6}  {'DiceSDF':>7}  {'DiceLbl':>7}  {'HD95mm':>7}  {'ASDmm':>6}", f)
        print_and_write("-" * 88, f)
        for r in results:
            def fmt(v, spec=".4f"):
                return f"{v:{spec}}" if v is not None else "  skip"
            print_and_write(f"  {r['case'][:42]:<42}  {fmt(r['mae']):>6}  {fmt(r['dice_sdf']):>7}"
                            f"  {fmt(r['dice_label']):>7}  {fmt(r['hd95'], '.3f'):>7}  {fmt(r['asd'], '.3f'):>6}", f)
        print_and_write("-" * 88, f)

        mae_scores       = [r["mae"] for r in results]
        dice_sdf_scores   = [r["dice_sdf"]   for r in valid_sdf]
        dice_label_scores = [r["dice_label"] for r in valid_label]
        hd95_scores       = [r["hd95"] for r in results if r["hd95"] is not None]
        asd_scores        = [r["asd"]  for r in results if r["asd"]  is not None]

        print_and_write(f"\n  Mean MAE        : {np.mean(mae_scores):.4f}  (std {np.std(mae_scores):.4f})", f)
        if dice_sdf_scores:
            print_and_write(f"  Mean Dice (SDF)  : {np.mean(dice_sdf_scores):.4f}  (std {np.std(dice_sdf_scores):.4f})", f)
        if dice_label_scores:
            print_and_write(f"  Mean Dice (label): {np.mean(dice_label_scores):.4f}  (std {np.std(dice_label_scores):.4f})", f)
        if hd95_scores:
            print_and_write(f"  Mean HD95        : {np.mean(hd95_scores):.3f} mm  (std {np.std(hd95_scores):.3f})", f)
        if asd_scores:
            print_and_write(f"  Mean ASD         : {np.mean(asd_scores):.3f} mm  (std {np.std(asd_scores):.3f})", f)

    summary_json = {
        "pred_dir": a.pred_dir,
        "cases": results,
        "aggregate": {
            "mae_mean":        float(np.mean(mae_scores)),
            "mae_std":         float(np.std(mae_scores)),
            "dice_sdf_mean":   float(np.mean(dice_sdf_scores))   if dice_sdf_scores   else None,
            "dice_sdf_std":    float(np.std(dice_sdf_scores))    if dice_sdf_scores   else None,
            "dice_label_mean": float(np.mean(dice_label_scores)) if dice_label_scores else None,
            "dice_label_std":  float(np.std(dice_label_scores))  if dice_label_scores else None,
            "hd95_mean":       float(np.mean(hd95_scores)) if hd95_scores else None,
            "hd95_std":        float(np.std(hd95_scores))  if hd95_scores else None,
            "asd_mean":        float(np.mean(asd_scores))  if asd_scores  else None,
            "asd_std":         float(np.std(asd_scores))   if asd_scores  else None,
        },
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
