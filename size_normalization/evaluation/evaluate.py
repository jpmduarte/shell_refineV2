"""Score predict.py's saved predictions at native resolution; it runs no model.

The SDF against the true native SDF (MAE, Dice of the negative regions) says how good the field is;
the mask (SDF < 0) against the original, never resampled label (Dice, HD95, ASD) gives the reported
numbers. Writes summary.txt (per case and the means) and summary.json to --out-dir.
"""
import argparse
import json
import os

import numpy as np
import torch

from common.geometry import voxel_spacing
from common.image_cache import load_label
from common.metrics import asd, dice, dice_from_sdf, hd95, mae
from common.sdf_targets import load as load_sdf_target
from datasets import add_dataset_argument, get as get_dataset

METRICS = ["mae", "dice_sdf", "dice_label", "hd95", "asd"]
RULE = "-" * 88


def evaluate(pred_dir: str, cache_dir: str) -> list[dict]:
    """One dictionary of scores per prediction file."""
    results = []
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz"))

    for fname in files:
        cid = os.path.splitext(fname)[0]
        prediction = np.load(os.path.join(pred_dir, fname))
        pred_sdf, affine = prediction["sdf"], prediction["affine"]

        target_sdf, _ = load_sdf_target(cache_dir, cid, 1)
        pred_t = torch.from_numpy(pred_sdf).float()
        target_t = torch.from_numpy(target_sdf.squeeze()).float()

        label, _ = load_label(cache_dir, cid)
        label = label.astype(bool)
        pred_mask = pred_sdf < 0
        spacing = voxel_spacing(affine)

        results.append({
            "case": cid,
            "mae": mae(pred_t, target_t),
            "dice_sdf": dice_from_sdf(pred_t, target_t),
            "dice_label": dice(torch.from_numpy(pred_mask.astype(np.float32)),
                               torch.from_numpy(label.astype(np.float32))),
            "hd95": hd95(pred_mask, label, spacing),
            "asd": asd(pred_mask, label, spacing),
        })

    return results


def defined_scores(results, metric) -> list[float]:
    """The metric over the cases where it is defined (not None)."""
    return [r[metric] for r in results if r[metric] is not None]


def fmt(value, spec=".4f"):
    return f"{value:{spec}}" if value is not None else "  skip"


def summary_lines(results, scores) -> list[str]:
    lines = [f"{'Case':<44}  {'MAE':>6}  {'DiceSDF':>7}  {'DiceLbl':>7}  {'HD95mm':>7}  {'ASDmm':>6}", RULE]
    for r in results:
        lines.append(f"  {r['case'][:42]:<42}  {fmt(r['mae']):>6}  {fmt(r['dice_sdf']):>7}"
                     f"  {fmt(r['dice_label']):>7}  {fmt(r['hd95'], '.3f'):>7}  {fmt(r['asd'], '.3f'):>6}")
    lines.append(RULE)

    mae_scores = scores["mae"]
    lines.append(f"\n  Mean MAE        : {np.mean(mae_scores):.4f}  (std {np.std(mae_scores):.4f})")
    for metric, label, spec, unit in (("dice_sdf", "Dice (SDF)  ", ".4f", ""),
                                      ("dice_label", "Dice (label)", ".4f", ""),
                                      ("hd95", "HD95        ", ".3f", " mm"),
                                      ("asd", "ASD         ", ".3f", " mm")):
        values = scores[metric]
        if values:
            lines.append(f"  Mean {label}: {np.mean(values):{spec}}{unit}  (std {np.std(values):{spec}})")
    return lines


def mean_std(scores):
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=str, required=True, help="predict.py's output folder")
    parser.add_argument("--cache-dir", type=str, default=None, help="default: the dataset's cache")
    parser.add_argument("--out-dir", type=str, required=True)
    add_dataset_argument(parser)
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or get_dataset(args.dataset).cache

    os.makedirs(args.out_dir, exist_ok=True)
    results = evaluate(args.pred_dir, args.cache_dir)
    scores = {"mae": [r["mae"] for r in results]}
    scores.update({metric: defined_scores(results, metric) for metric in METRICS[1:]})

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines(results, scores):
            print(line)
            f.write(line + "\n")

    aggregate = {}
    for metric in METRICS:
        aggregate[f"{metric}_mean"], aggregate[f"{metric}_std"] = mean_std(scores[metric])
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"pred_dir": args.pred_dir, "cases": results, "aggregate": aggregate}, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
