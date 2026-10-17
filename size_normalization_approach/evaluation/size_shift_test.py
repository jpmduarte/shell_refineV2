# Size-shift test: does size normalisation make the method robust to head size?
#
#     python size_normalization_approach/evaluation/size_shift_test.py
#     python size_normalization_approach/evaluation/size_shift_test.py --scales 0.7 1.0 1.3 --fold 2 --seed 43
#
# Each validation volume of the fold is resampled so the head spans s times more voxels
# at the same 0.4 mm voxel, as if the fetus were bigger (s > 1) or smaller (s < 1); the
# label goes through its native SDF the same way, so it keeps a smooth surface. Both
# methods then run exactly as deployed (infer.py):
#     plain x4    the x4 network alone (runs/nnaug_x4e500_s<seed>)
#     normalised  that network for the head volume, then the size-normalised network
#                 (runs/nnaug_x4norm_s<seed>) on a grid that brings the head to V0
# Dice at native resolution against the resampled label, no TTA, plus pass 1's volume
# error, since the normalised method only knows the head's size through it.
#
# The normalised method only knows the head's size through pass 1. Its scale
# k = (V / V0)^(1/3) is accepted only within --k-range (default 0.5-2, a head of about
# 16-1040 mL); outside it, including a pass 1 that finds no head (k = 0), the method falls
# back to pass 1's own segmentation (the plain x4 mask), and the case is counted. Without
# that rule a tiny but non-zero estimate asks for a normalised grid far too large for the
# GPU. The rule lives in the pipeline itself (infer.py segment, k_range).
#
# Assumes the cache's geometry: RAS, isotropic voxels (0.4 mm), so the affine is diagonal.
#
# Writes one row per (scale, case) to --out as JSON.

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
SNA    = os.path.dirname(HERE)          # size_normalization_approach/: the method, its runs
ROOT   = os.path.dirname(SNA)
sys.path[:0] = [ROOT, SNA]

from cache import case_ids, load as load_image  # noqa: E402
from geometry import affine_resample, spacing_grid  # noqa: E402
from infer import load_net, segment, segment_plain  # noqa: E402
from make_sdf import load as load_sdf  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


# The same physical field of view, sampled at spacing / s and then read as voxels of the
# original spacing: the content appears s times larger (s > 1) or smaller (s < 1).
# Nothing is cropped; the array grows or shrinks with s.
def zoom(arr: np.ndarray, affine: np.ndarray, s: float, cval: float, dev) -> torch.Tensor:
    spacing = float(np.abs(affine[0, 0]))
    dst_affine, dst_shape = spacing_grid(affine, arr.shape, spacing / s)
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(dev)
    return affine_resample(t, affine, dst_affine, dst_shape, cval=cval)


# Both masks empty cannot happen here (every case has a head); it would read as 1.
def dice(a: torch.Tensor, b: torch.Tensor) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def main():
    p = argparse.ArgumentParser(description="Robustness of plain and size-normalised x4 to head size.")
    p.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
                   help="linear head scales; the volume changes by s^3")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"),
                   help="the normalised network's training cache: its norm.json gives V0")
    p.add_argument("--k-range", type=float, nargs=2, default=[0.5, 2.0],
                   help="pass-1 scales accepted by the normalised method; outside: pass 1's mask")
    p.add_argument("--out", default=None, help="default: runs/size_shift_fold<k>_s<seed>.json")
    a = p.parse_args()
    out = a.out or os.path.join(SNA, "runs", f"size_shift_fold{a.fold}_s{a.seed}.json")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def net(run):
        return load_net(os.path.join(SNA, "runs", f"{run}_s{a.seed}", f"fold{a.fold}",
                                     "checkpoints", "phase1_last.pth"), dev)[0]

    plain = net("nnaug_x4e500")       # the plain x4 method, and pass 1 of the other
    norm = net("nnaug_x4norm")        # pass 2 of the size-normalised method
    v0_ml = reference_v0(a.reference)
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42,
                      splits_json=os.path.join(ROOT, "splits.json"))
    cases = fold_split(folds, a.fold)[1]

    rows = []
    for s in a.scales:
        for cid in cases:
            img1, aff1 = load_image(a.cache_dir, cid, 1)
            sdf1, _ = load_sdf(a.cache_dir, cid, 1)
            img = zoom(img1.squeeze(), aff1, s, 0.0, dev).cpu().numpy()
            # The label through its SDF: outside the volume is "far outside" (+1).
            label = (zoom(sdf1.squeeze(), aff1, s, 1.0, dev) < 0).cpu()
            spacing = float(np.abs(aff1[0, 0]))

            mask_plain = (segment_plain(img, aff1, [plain], False, dev) < 0).cpu()
            # A rejected scale returns pass 1's segmentation: the plain mask itself.
            field, info = segment(img, aff1, plain, [norm], v0_ml, False, dev,
                                  k_range=tuple(a.k_range))
            mask_norm = (field < 0).cpu()

            rows.append({"scale": s, "case": cid,
                         "true_ml": float(label.sum()) * spacing ** 3 / 1000,
                         "est_ml": info["volume_ml"], "k": info["k"],
                         "pass1_rejected": info["fallback"],
                         "plain": dice(mask_plain, label), "norm": dice(mask_norm, label)})
            torch.cuda.empty_cache()

        r = [x for x in rows if x["scale"] == s]
        d_plain = np.array([x["plain"] for x in r])
        d_norm = np.array([x["norm"] for x in r])
        verr = np.array([x["est_ml"] / x["true_ml"] - 1 for x in r if x["true_ml"] > 0])
        print(f"s={s:4.2f}  volume x{s ** 3:4.2f} (median {np.median([x['true_ml'] for x in r]):5.0f} mL)  "
              f"plain {d_plain.mean():.4f}  normalised {d_norm.mean():.4f}  "
              f"diff {d_norm.mean() - d_plain.mean():+.4f}  normalised better {int((d_norm > d_plain).sum())}/{len(r)}  "
              f"pass-1 volume error median |{100 * np.median(abs(verr)):.1f}|% (signed mean {100 * verr.mean():+.1f}%)  "
              f"k {min(x['k'] for x in r):.2f}-{max(x['k'] for x in r):.2f}  "
              f"pass 1 rejected: {sum(x['pass1_rejected'] for x in r)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
