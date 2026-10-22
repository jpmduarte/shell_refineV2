"""Size-shift test: is the method robust to head size? Plain x4 against size-normalised, both as deployed.

Each validation volume is resampled so the head spans s times more voxels at the same 0.4 mm voxel
(the label through its native SDF, so it keeps a smooth surface). Scores: Dice at native resolution
against that label, no TTA, and pass 1's volume error. A scale k outside --k-range falls back to
pass 1's mask (infer.segment), and the case is counted. Writes one row per (scale, case) to --out.
"""
import argparse
import json
import os

import numpy as np
import torch

from common.folds import N_FOLDS, SEED as FOLDS_SEED, fold_split, get_folds
from common.geometry import MM3_PER_ML, affine_resample, isotropic_spacing, spacing_grid
from common.image_cache import case_ids, load as load_image
from common.metrics import mask_dice
from common.sdf_targets import load as load_sdf
from datasets import add_dataset_argument, get as get_dataset
from size_normalization.build_cache import reference_v0
from size_normalization.infer import OUTSIDE_SDF, load_net, segment, segment_plain

USAGE = """\
examples:
  python size_normalization/evaluation/size_shift_test.py
  python size_normalization/evaluation/size_shift_test.py --scales 0.7 1.0 1.3 --fold 2 --seed 43"""


def zoom(arr: np.ndarray, affine: np.ndarray, s: float, cval: float, device) -> torch.Tensor:
    """The same field of view sampled at spacing / s and read at the original spacing: content s times larger."""
    dst_affine, dst_shape = spacing_grid(affine, arr.shape, isotropic_spacing(affine) / s)
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)
    return affine_resample(t, affine, dst_affine, dst_shape, cval=cval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
                        help="linear head scales; the volume changes by s^3")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    add_dataset_argument(parser)
    parser.add_argument("--cache-dir", default=None, help="default: the dataset's cache")
    parser.add_argument("--reference", default=None,
                        help="the normalised network's training cache, whose norm.json gives V0; default: the dataset's")
    parser.add_argument("--k-range", type=float, nargs=2, default=None,
                        help="pass-1 scales accepted by the normalised method; outside: pass 1's mask. "
                             "Default: the dataset's")
    parser.add_argument("--plain-run", default=None,
                        help="the run family scored as the plain method (e.g. nnaug_x4strongscale, "
                             "trained with strong scaling); pass 1 of the normalised method stays the "
                             "dataset's pass-1 run. Default: that run")
    parser.add_argument("--out", default=None,
                        help="default: <the dataset's runs>/size_shift[_<plain-run>]_fold<k>_s<seed>.json")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.cache_dir = args.cache_dir or dataset.cache
    args.reference = args.reference or dataset.norm_cache
    args.k_range = tuple(args.k_range) if args.k_range else dataset.k_range
    args.plain_run = args.plain_run or dataset.pass1_run
    return args


def print_scale_summary(s, rows):
    plain = np.array([x["plain"] for x in rows])
    norm = np.array([x["norm"] for x in rows])
    volume_error = np.array([x["est_ml"] / x["true_ml"] - 1 for x in rows if x["true_ml"] > 0])
    print(f"s={s:4.2f}  volume x{s ** 3:4.2f} (median {np.median([x['true_ml'] for x in rows]):5.0f} mL)  "
          f"plain {plain.mean():.4f}  normalised {norm.mean():.4f}  "
          f"diff {norm.mean() - plain.mean():+.4f}  normalised better {int((norm > plain).sum())}/{len(rows)}  "
          f"pass-1 volume error median |{100 * np.median(abs(volume_error)):.1f}|% "
          f"(signed mean {100 * volume_error.mean():+.1f}%)  "
          f"k {min(x['k'] for x in rows):.2f}-{max(x['k'] for x in rows):.2f}  "
          f"pass 1 rejected: {sum(x['pass1_rejected'] for x in rows)}", flush=True)


def main():
    args = parse_args()
    dataset = get_dataset(args.dataset)
    tag = "" if args.plain_run == dataset.pass1_run else f"_{args.plain_run}"
    out = args.out or os.path.join(dataset.runs, f"size_shift{tag}_fold{args.fold}_s{args.seed}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def net(run):
        return load_net(dataset.checkpoint(run, args.seed, args.fold), device)

    pass1 = net(dataset.pass1_run)
    norm = net(dataset.pass2_run)
    plain = net(args.plain_run) if args.plain_run != dataset.pass1_run else pass1
    v0_ml = reference_v0(args.reference)
    folds = get_folds(dataset, case_ids(args.cache_dir), n_folds=N_FOLDS, seed=FOLDS_SEED)
    _, cases = fold_split(folds, args.fold)

    rows = []
    for s in args.scales:
        for cid in cases:
            img1, aff1 = load_image(args.cache_dir, cid, 1)
            sdf1, _ = load_sdf(args.cache_dir, cid, 1)
            img = zoom(img1.squeeze(), aff1, s, 0.0, device).cpu().numpy()
            label = (zoom(sdf1.squeeze(), aff1, s, OUTSIDE_SDF, device) < 0).cpu()

            mask_plain = (segment_plain(img, aff1, [plain], False, device) < 0).cpu()
            field, estimate = segment(img, aff1, pass1, [norm], v0_ml, False, device, k_range=args.k_range)
            mask_norm = (field < 0).cpu()

            rows.append({"scale": s, "case": cid,
                         "true_ml": float(label.sum()) * isotropic_spacing(aff1) ** 3 / MM3_PER_ML,
                         "est_ml": estimate.volume_ml, "k": estimate.k,
                         "pass1_rejected": estimate.fallback,
                         "plain": mask_dice(mask_plain, label, if_both_empty=1.0),
                         "norm": mask_dice(mask_norm, label, if_both_empty=1.0)})
            torch.cuda.empty_cache()

        print_scale_summary(s, [x for x in rows if x["scale"] == s])

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
