"""The refiner's priors: a trained first-pass network's out-of-fold prediction for every case.

Each case is predicted by the networks of the fold that held it out, so the refiner learns the errors
the first pass makes on unseen scans. <out-dir>/<case>.npz holds the native SDF and affine, the format
make_crops.py and size_normalization/evaluation/evaluate.py read. Several --seeds are averaged.
"""
import argparse
import os
import zipfile

import numpy as np
import torch
from numpy.lib import format as npy_format

from common.folds import N_FOLDS, SEED as FOLDS_SEED, fold_split, get_folds
from common.geometry import FACTOR
from common.image_cache import case_ids, load as load_image
from common.paths import REFINER_DIR
from datasets import add_dataset_argument, get as get_dataset
from size_normalization.infer import forward, load_net, to_native_as_cache


USAGE = """\
examples:
  python boundary_refiner/make_priors.py --seeds 42                 -> priors/x4_s42/
  python boundary_refiner/make_priors.py --seeds 42 43 44 --tta     -> priors/x4_ens_s42-43-44_tta/
  the size-normalised networks, each seed on its own estimated cache:
    --checkpoint "size_normalization/runs/nnaug_x4norm_s{s}/fold{f}/checkpoints/phase1_last.pth"
    --cache-dir "caches/cache_norm4_est_s{s}" --out-dir boundary_refiner/priors/x4norm_ens_tta"""


def native_geometry(cache_dir: str, cid: str):
    """The native shape and affine, the shape read from the .npy header without loading the voxels."""
    path = os.path.join(cache_dir, "x1", f"{cid}.npz")
    with np.load(path) as cached:
        affine = cached["affine"]
    with zipfile.ZipFile(path) as archive, archive.open("img.npy") as f:
        major, _ = npy_format.read_magic(f)
        read_header = npy_format.read_array_header_1_0 if major == 1 else npy_format.read_array_header_2_0
        shape, _, _ = read_header(f)
    return tuple(shape), affine


@torch.no_grad()
def predict_one(model, cache_dir: str, cid: str, device, tta: bool = False):
    """(the SDF on the native grid, on the GPU; the native affine)."""
    img4, affine4 = load_image(cache_dir, cid, FACTOR)
    native_shape, affine1 = native_geometry(cache_dir, cid)
    field4 = forward(model, torch.from_numpy(img4.astype(np.float32)).to(device), tta)
    return to_native_as_cache(field4, affine4, affine1, native_shape), affine1


def output_dir(args) -> str:
    if args.out_dir:
        return args.out_dir
    seeds = f"s{args.seeds[0]}" if len(args.seeds) == 1 else "ens_s" + "-".join(map(str, args.seeds))
    return os.path.join(REFINER_DIR, "priors", f"x4_{seeds}" + ("_tta" if args.tta else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="one seed, or several: their SDFs are averaged (an ensemble)")
    parser.add_argument("--tta", action="store_true", help="average the 8 mirrored predictions")
    add_dataset_argument(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path pattern, with {s} for the seed and {f} for the fold; "
                             "default: the dataset's pass-1 run")
    parser.add_argument("--cache-dir", default=None,
                        help="the grid the networks read; may contain {s} for a cache per seed; default: the dataset's")
    parser.add_argument("--splits-json", default=None, help="default: the dataset's splits.json")
    parser.add_argument("--out-dir", default=None,
                        help="default: priors/x4_s<seed> (or x4_ens_s42-43-44, with _tta) here")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.checkpoint = args.checkpoint or dataset.checkpoint(dataset.pass1_run, "{s}", "{f}")
    args.cache_dir = args.cache_dir or dataset.cache

    out_dir = output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = get_folds(dataset, case_ids(args.cache_dir.format(s=args.seeds[0])), n_folds=N_FOLDS, seed=FOLDS_SEED,
                      splits_json=args.splits_json)

    for k in range(len(folds)):
        models = [load_net(args.checkpoint.format(s=seed, f=k), device) for seed in args.seeds]
        _, val_ids = fold_split(folds, k)

        for cid in val_ids:
            # Each seed on its own cache; the SDFs are averaged on the native grid, which they share.
            predictions = [predict_one(m, args.cache_dir.format(s=seed), cid, device, args.tta)
                           for m, seed in zip(models, args.seeds)]
            fields = [field for field, _ in predictions]
            sdf = fields[0] if len(fields) == 1 else torch.stack(fields).mean(0)
            affine1 = predictions[0][1]
            np.savez(os.path.join(out_dir, f"{cid}.npz"), sdf=sdf.cpu().numpy().astype(np.float32), affine=affine1)
        print(f"fold {k}: {len(val_ids)} priors", flush=True)

    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
