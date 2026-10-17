# Scores nnU-Net's predictions with our metric code against our labels, so both methods
# are measured the same way.
#
#     python scripts/score_nnunet.py --pred-root <dir with fold0..fold4> \
#         --case-map case_map.json --out summary.json
#
# nnU-Net writes <pred-root>/fold<k>/caseNNN.nii.gz in the original NIfTI orientation and
# under its own case numbers. Each prediction is loaded and reoriented to RAS exactly as
# cache.py treats labels, renamed to our case id through case_map.json, checked to lie on
# the same grid as the cached label, and then scored with metrics.py, as
# size_normalization_approach/evaluation/evaluate.py scores ours.

import argparse
import glob
import json
import os
import statistics as st
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd  # noqa: E402

from cache import load_label  # noqa: E402
from metrics import asd, dice, hd95  # noqa: E402

# The label part of cache.PREPROCESS: load, add the channel axis, reorient to RAS.
LOAD = Compose([
    LoadImaged(keys=["pred"]),
    EnsureChannelFirstd(keys=["pred"]),
    Orientationd(keys=["pred"], axcodes="RAS", labels=None),
])


def main():
    p = argparse.ArgumentParser(description="Score nnU-Net predictions with our metrics.")
    p.add_argument("--pred-root", required=True)
    p.add_argument("--case-map", required=True, help="case_map.json: our id -> caseNNN")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # caseNNN -> our case id
    our_id = {v: k for k, v in json.load(open(a.case_map)).items()}

    folds, cases = {}, []
    for fold_dir in sorted(glob.glob(os.path.join(a.pred_root, "fold*"))):
        k = int(os.path.basename(fold_dir)[4:])
        rows = []
        for path in sorted(glob.glob(os.path.join(fold_dir, "case*.nii.gz"))):
            cid = our_id[os.path.basename(path)[:-7]]
            item = LOAD({"pred": path})
            pred = np.asarray(item["pred"][0]) > 0
            aff = np.asarray(item["pred"].affine)

            # After reorienting, the prediction must sit on exactly the label's grid.
            label, label_aff = load_label(a.cache_dir, cid)
            label = (label[0] if label.ndim == 4 else label).astype(bool)
            if pred.shape != label.shape or np.abs(aff - label_aff).max() > 1e-3:
                raise SystemExit(f"{cid}: geometry mismatch after RAS "
                                 f"{pred.shape} vs {label.shape}")

            spacing = np.abs(np.diag(label_aff[:3, :3]))
            r = {"case": cid, "fold": k,
                 "dice": dice(torch.from_numpy(pred.astype(np.float32)),
                              torch.from_numpy(label.astype(np.float32))),
                 "hd95": hd95(pred, label, spacing),
                 "asd": asd(pred, label, spacing)}
            rows.append(r)
            print(f"fold{k} {cid[:36]:<36} dice {r['dice']:.4f} hd95 {r['hd95']:.3f} "
                  f"asd {r['asd']:.3f}", flush=True)

        # Per-fold means.
        # Means over the cases where each metric is defined: hd95/asd are None for an
        # empty mask, which st.mean cannot take.
        folds[k] = {m: st.mean(r[m] for r in rows if r[m] is not None)
                    for m in ("dice", "hd95", "asd")}
        folds[k]["n"] = len(rows)
        cases.extend(rows)

    # Overall: the mean of the fold means, and the spread of Dice between folds.
    overall = {m: st.mean(f[m] for f in folds.values()) for m in ("dice", "hd95", "asd")}
    overall["dice_std_between_folds"] = (st.stdev(f["dice"] for f in folds.values())
                                         if len(folds) > 1 else None)
    print("\nper fold:", {k: round(v["dice"], 4) for k, v in folds.items()})
    print(f"overall  Dice {overall['dice']:.4f}  HD95 {overall['hd95']:.3f}  "
          f"ASD {overall['asd']:.3f}")

    with open(a.out, "w") as fh:
        json.dump({"pred_root": a.pred_root, "folds": folds, "overall": overall,
                   "cases": cases}, fh, indent=2)


if __name__ == "__main__":
    main()
