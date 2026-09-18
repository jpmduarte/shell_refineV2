"""Run phase 1 on one or all cases, upsample the predicted SDF back to native
resolution, and save it.

    python phase1/predict.py --checkpoint phase1/runs/fold0/checkpoints/phase1_best.pth --case <cid> --out-dir preds
    python phase1/predict.py --checkpoint phase1/runs/fold0/checkpoints/phase1_best.pth --all --out-dir preds
    python phase1/predict.py --checkpoint phase1/runs/fold0/checkpoints/phase1_best.pth --fold 0 --split val --out-dir phase1/runs/fold0/preds_val

Output: <out_dir>/<case>.npz with keys
    sdf      predicted SDF, normalised [-1, 1], at native (x1) resolution
    affine   the native affine (same as cache/x1/<case>.npz)
"""

import argparse
import os
import sys

import numpy as np
import torch
from monai.data import MetaTensor

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import case_ids, load as load_image  # noqa: E402
from dataset import pad_to_cube, CUBE  # noqa: E402
from geometry import upsample  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


@torch.no_grad()
def predict_one(model, cache_dir: str, cid: str, device: torch.device, cube: int = CUBE):
    img8, _        = load_image(cache_dir, cid, 8)
    img1, affine1  = load_image(cache_dir, cid, 1)

    img_padded = pad_to_cube(img8.astype(np.float32), cube, cval=0.0)
    x = torch.from_numpy(img_padded[np.newaxis, np.newaxis]).float().to(device)

    pred = model(x)[0]  # (1, cube, cube, cube), normalised [-1, 1] — drop batch dim,
                        # SpatialResample only supports channel-first (no batch)
    pred = pred[:, :img8.shape[0], :img8.shape[1], :img8.shape[2]]  # undo the pad

    _, affine8 = load_image(cache_dir, cid, 8)
    meta = MetaTensor(pred.double(), affine=torch.from_numpy(affine8))
    native = upsample(meta, torch.from_numpy(affine1), img1.shape, cval=1.0)

    return native[0].cpu().numpy().astype(np.float32), affine1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--case",        type=str, default=None, help="single case id")
    p.add_argument("--all",         action="store_true", help="run every cached case")
    p.add_argument("--fold",        type=int, default=None, help="run one fold's split")
    p.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--splits-json", type=str, default=os.path.join(PARENT, "splits.json"))
    p.add_argument("--base-ch",     type=int, default=16)
    p.add_argument("--cache-dir",   type=str, default=os.path.join(PARENT, "cache"))
    p.add_argument("--out-dir",     type=str, required=True)
    a = p.parse_args()

    if sum(x is not None for x in (a.case, a.fold)) + int(a.all) != 1:
        p.error("pass exactly one of --case <cid>, --all, or --fold <k>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, base_channels=a.base_ch, out_activation="tanh").to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    os.makedirs(a.out_dir, exist_ok=True)

    if a.fold is not None:
        ids = case_ids(a.cache_dir)
        folds = get_folds(ids, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
        train_ids, val_ids = fold_split(folds, a.fold)
        cids = val_ids if a.split == "val" else train_ids
    elif a.all:
        cids = case_ids(a.cache_dir)
    else:
        cids = [a.case]

    print(f"Model: {model_summary(model)}")

    prof = Profiler("phase1_predict")
    with prof:
        for i, cid in enumerate(cids, start=1):
            with prof.span("predict_case") as s:
                sdf, affine = predict_one(model, a.cache_dir, cid, device)
                s["items"] = 1
            np.savez(os.path.join(a.out_dir, f"{cid}.npz"), sdf=sdf, affine=affine)
            print(f"[{i}/{len(cids)}] {cid[:44]:<44} native={sdf.shape}", flush=True)

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")


if __name__ == "__main__":
    main()
