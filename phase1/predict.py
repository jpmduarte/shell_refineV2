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
import zipfile

import numpy as np
import torch
from numpy.lib import format as npy_format

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import case_ids, load as load_image  # noqa: E402
from dataset import pad_to_cube, CUBE  # noqa: E402
from geometry import apply_resample, resample_grid  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def native_geometry(cache_dir: str, cid: str):
    """Native shape and affine without decompressing the voxels.

    An npz is a zip of .npy members, and a .npy header already carries the shape. Asking
    numpy for d["img"].shape decompresses all 78 MB to answer a question the header
    holds: 50ms against 0.5ms.
    """
    path = os.path.join(cache_dir, "x1", f"{cid}.npz")
    with np.load(path) as d:
        affine = d["affine"]
    with zipfile.ZipFile(path) as z, z.open("img.npy") as f:
        major, _ = npy_format.read_magic(f)
        reader = (npy_format.read_array_header_1_0 if major == 1
                  else npy_format.read_array_header_2_0)
        shape, _, _ = reader(f)
    return tuple(shape), affine

@torch.no_grad()
def predict_one(model, cache_dir: str, cid: str, device: torch.device, cube: int = CUBE):
    """Predict the field and carry it up to native resolution.

    The upsample dominated this: SpatialResample builds its grid from the affines in
    float64 on the CPU, which measured 563ms per case against the network's 16ms. The
    grid depends only on the affines, which are constants, so it can be built once and
    the sampling done on the GPU in float32 — 149x faster, and agreeing with
    SpatialResample to 1e-5, which is float precision rather than different geometry.
    """
    img8, affine8 = load_image(cache_dir, cid, 8)
    # Only the native shape and affine are needed, not 78 MB of voxels: reading the
    # header alone saves 65ms per case.
    native_shape, affine1 = native_geometry(cache_dir, cid)

    img_padded = pad_to_cube(img8.astype(np.float32), cube, cval=0.0)
    x = torch.from_numpy(img_padded[np.newaxis, np.newaxis]).float().to(device)

    pred = model(x)
    pred = pred[:, :, :img8.shape[0], :img8.shape[1], :img8.shape[2]]  # undo the pad

    grid = resample_grid(affine8, affine1, native_shape, device=device).float()
    # cval=1: outside the volume an SDF is far outside. Padding with 0 would put a
    # surface along the border.
    native = apply_resample(pred, grid, img8.shape, cval=1.0)

    return native[0, 0].cpu().numpy().astype(np.float32), affine1


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
