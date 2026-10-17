# Runs the method on cached cases and saves each predicted SDF at native resolution.
#
#     # the size-normalised method (pass 1 for the volume, then pass 2), as deployed:
#     python size_normalization_approach/evaluation/predict.py --fold 0 --out-dir <run>/fold0/preds_val \
#         --pass1 runs/nnaug_x4e500_s42/fold0/checkpoints/phase1_last.pth \
#         --checkpoint runs/nnaug_x4norm_s42/fold0/checkpoints/phase1_last.pth
#     # the plain x4 method (also pass 1 on its own): no --pass1
#     python size_normalization_approach/evaluation/predict.py --fold 0 --out-dir ... --checkpoint <x4 run>/...
#
# Several --checkpoint are averaged as SDFs (the ensemble); --tta adds the 8 mirrorings;
# --oracle normalises with the label's volume instead of pass 1's estimate (an upper bound).
# Everything goes through infer.py, the same code segment.py and the benchmark run, from the
# native image in caches/cache/x1.
#
# Output: <out-dir>/<case>.npz with
#     sdf      the predicted SDF, normalised to [-1, 1], at native resolution
#     affine   the native affine (the same as caches/cache/x1/<case>.npz)
#     volume_ml, k, fallback   what pass 1 measured (size-normalised method only)
# size_normalization_approach/evaluation/evaluate.py scores these files.

import argparse
import os
import sys

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
SNA    = os.path.dirname(HERE)          # size_normalization_approach/: the method, its runs
ROOT   = os.path.dirname(SNA)
sys.path[:0] = [ROOT, SNA]

from cache import case_ids, load as load_image, load_label  # noqa: E402
from infer import load_net, segment, segment_plain  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Predict native-resolution SDFs with trained networks.")
    p.add_argument("--checkpoint",  type=str, nargs="+", required=True,
                   help="the network(s); several are averaged. The size-normalised (pass 2) "
                        "networks with --pass1, the plain method's without")
    p.add_argument("--pass1",       type=str, default=None,
                   help="the plain x4 network that estimates the head volume; makes this the "
                        "size-normalised method")
    p.add_argument("--tta",         action="store_true", help="average the 8 mirrorings")
    p.add_argument("--oracle",      action="store_true",
                   help="normalise with the label's head volume instead of pass 1's estimate")
    p.add_argument("--blend",       type=float, default=0.0,
                   help="weight of pass 1's SDF in the final one (0 = off; 0.25 tested)")
    p.add_argument("--reference",   type=str, default=os.path.join(ROOT, "caches", "cache_norm4"),
                   help="the pass-2 networks' training cache: its norm.json gives V0")
    p.add_argument("--case",        type=str, default=None, help="a single case id")
    p.add_argument("--all",         action="store_true", help="every cached case")
    p.add_argument("--fold",        type=int, default=None, help="one fold's cases")
    p.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--splits-json", type=str, default=os.path.join(ROOT, "splits.json"))
    p.add_argument("--cache-dir",   type=str, default=os.path.join(ROOT, "caches", "cache"),
                   help="the native cache (x1, label)")
    p.add_argument("--out-dir",     type=str, required=True)
    a = p.parse_args()

    if sum(x is not None for x in (a.case, a.fold)) + int(a.all) != 1:
        p.error("pass exactly one of --case <cid>, --all, or --fold <k>")
    if (a.oracle or a.blend) and not a.pass1:
        p.error("--oracle and --blend need --pass1 (the size-normalised method)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = [load_net(c, dev) for c in a.checkpoint]
    models, cfg = [m for m, _ in nets], nets[0][1]
    if any(c != cfg for _, c in nets):
        p.error("the --checkpoint networks were trained on different grids")
    grid = dict(factor2=cfg["factor"], cube2=cfg["cube"], mask2=cfg["loss"] == "mask")
    if a.pass1:
        pass1, _ = load_net(a.pass1, dev)
        v0_ml = reference_v0(a.reference)
    print(f"{'size-normalised' if a.pass1 else 'plain'} method, {len(models)} network(s) on x{cfg['factor']} "
          f"in {cfg['cube']}^3, output {cfg['loss']}" + (", TTA" if a.tta else "")
          + (", oracle volume" if a.oracle else ""))

    os.makedirs(a.out_dir, exist_ok=True)

    # The cases to predict.
    if a.fold is not None:
        folds = get_folds(case_ids(a.cache_dir), n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
        train_ids, val_ids = fold_split(folds, a.fold)
        cids = val_ids if a.split == "val" else train_ids
    elif a.all:
        cids = case_ids(a.cache_dir)
    else:
        cids = [a.case]

    for i, cid in enumerate(cids, start=1):
        img1, aff1 = load_image(a.cache_dir, cid, 1)
        img1 = np.asarray(img1, dtype=np.float32).squeeze()
        extra = {}
        if a.pass1:
            volume = None
            if a.oracle:
                lab, _ = load_label(a.cache_dir, cid)
                volume = float(lab.astype(bool).sum()) * float(np.abs(aff1[0, 0])) ** 3 / 1000
            field, info = segment(img1, aff1, pass1, models, v0_ml, a.tta, dev, blend=a.blend,
                                  volume=volume, **grid)
            extra = info
        else:
            field = segment_plain(img1, aff1, models, a.tta, dev, factor=grid["factor2"],
                                  cube=grid["cube2"], mask=grid["mask2"])
        sdf = field.cpu().numpy().astype(np.float32)
        np.savez(os.path.join(a.out_dir, f"{cid}.npz"), sdf=sdf, affine=aff1, **extra)
        print(f"[{i}/{len(cids)}] {cid[:44]:<44} native={sdf.shape}"
              + (f"  {extra['volume_ml']:.1f} mL  k {extra['k']:.3f}"
                 + ("  FALLBACK to pass 1" if extra["fallback"] else "") if extra else ""), flush=True)
        torch.cuda.empty_cache()

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")


if __name__ == "__main__":
    main()
