"""Run the method on cached cases and save each predicted SDF at native resolution, through infer.py as deployed.

Output: <out-dir>/<case>.npz with sdf (normalised, native grid), affine (the native one) and, for the
size-normalised method, what pass 1 measured (volume_ml, k, fallback). evaluate.py scores these files.
"""
import argparse
import dataclasses
import os

import numpy as np
import torch

from common.folds import SEED as FOLDS_SEED, fold_split, get_folds
from common.geometry import CUBE, FACTOR, MM3_PER_ML, isotropic_spacing
from common.image_cache import case_ids, load as load_image, load_label
from datasets import add_dataset_argument, get as get_dataset
from size_normalization.build_cache import reference_v0
from size_normalization.infer import load_net, segment, segment_plain

USAGE = """\
examples:
  the size-normalised method (pass 1 for the volume, then pass 2), as deployed:
    python size_normalization/evaluation/predict.py --fold 0 --out-dir <run>/fold0/preds_val \\
        --pass1 runs/nnaug_x4e500_s42/fold0/checkpoints/phase1_last.pth \\
        --checkpoint runs/nnaug_x4norm_s42/fold0/checkpoints/phase1_last.pth
  the plain x4 method (also pass 1 on its own): no --pass1
    python size_normalization/evaluation/predict.py --fold 0 --out-dir ... --checkpoint <x4 run>/..."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="the network(s); several are averaged. The size-normalised (pass 2) "
                             "networks with --pass1, the plain method's without")
    parser.add_argument("--pass1", type=str, default=None,
                        help="the plain x4 network that estimates the head volume; makes this the "
                             "size-normalised method")
    parser.add_argument("--tta", action="store_true", help="average the 8 mirrorings")
    parser.add_argument("--oracle", action="store_true",
                        help="normalise with the label's head volume instead of pass 1's estimate")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    parser.add_argument("--reference", type=str, default=None,
                        help="the pass-2 networks' training cache, whose norm.json gives V0; default: the dataset's")
    parser.add_argument("--case", type=str, default=None, help="a single case id")
    parser.add_argument("--all", action="store_true", help="every cached case")
    parser.add_argument("--fold", type=int, default=None, help="one fold's cases")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--splits-json", type=str, default=None, help="default: the dataset's splits.json")
    parser.add_argument("--cache-dir", type=str, default=None, help="the native cache (x1, label); default: the dataset's")
    parser.add_argument("--out-dir", type=str, required=True)
    add_dataset_argument(parser)
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.reference = args.reference or dataset.norm_cache
    args.splits_json = args.splits_json or dataset.splits_json
    args.cache_dir = args.cache_dir or dataset.cache

    if sum(x is not None for x in (args.case, args.fold)) + int(args.all) != 1:
        parser.error("pass exactly one of --case <cid>, --all, or --fold <k>")
    if (args.oracle or args.blend) and not args.pass1:
        parser.error("--oracle and --blend need --pass1 (the size-normalised method)")
    return args


def select_cases(args) -> list[str]:
    if args.fold is not None:
        folds = get_folds(get_dataset(args.dataset), case_ids(args.cache_dir), n_folds=args.n_folds, seed=FOLDS_SEED,
                          splits_json=args.splits_json)
        train_ids, val_ids = fold_split(folds, args.fold)
        return val_ids if args.split == "val" else train_ids
    if args.all:
        return case_ids(args.cache_dir)
    return [args.case]


def oracle_volume_ml(cache_dir, cid, aff1) -> float:
    label, _ = load_label(cache_dir, cid)
    return float(label.astype(bool).sum()) * isotropic_spacing(aff1) ** 3 / MM3_PER_ML


def main():
    args = parse_args()
    dataset = get_dataset(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_net(c, device) for c in args.checkpoint]
    if args.pass1:
        pass1 = load_net(args.pass1, device)
        v0_ml = reference_v0(args.reference)
    print(f"{'size-normalised' if args.pass1 else 'plain'} method, {len(models)} network(s) on x{FACTOR} "
          f"in {CUBE}^3, output sdf" + (", TTA" if args.tta else "")
          + (", oracle volume" if args.oracle else ""))

    os.makedirs(args.out_dir, exist_ok=True)
    cids = select_cases(args)

    for i, cid in enumerate(cids, start=1):
        img1, aff1 = load_image(args.cache_dir, cid, 1)
        img1 = np.asarray(img1, dtype=np.float32).squeeze()
        pass1_record = {}
        if args.pass1:
            volume = oracle_volume_ml(args.cache_dir, cid, aff1) if args.oracle else None
            field, estimate = segment(img1, aff1, pass1, models, v0_ml, args.tta, device,
                                      blend=args.blend, k_range=dataset.k_range, volume=volume)
            pass1_record = dataclasses.asdict(estimate)
        else:
            field = segment_plain(img1, aff1, models, args.tta, device)
        sdf = field.cpu().numpy().astype(np.float32)
        np.savez(os.path.join(args.out_dir, f"{cid}.npz"), sdf=sdf, affine=aff1, **pass1_record)

        line = f"[{i}/{len(cids)}] {cid[:44]:<44} native={sdf.shape}"
        if pass1_record:
            line += f"  {estimate.volume_ml:.1f} mL  k {estimate.k:.3f}"
            line += "  FALLBACK to pass 1" if estimate.fallback else ""
        print(line, flush=True)
        torch.cuda.empty_cache()

    print(f"\n{len(cids)} case(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
