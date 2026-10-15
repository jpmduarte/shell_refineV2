# Runs a trained network on cached cases and saves each predicted SDF at native
# resolution.
#
#     python size_normalization_approach/predict.py --checkpoint <run>/fold0/checkpoints/phase1_last.pth \
#         --fold 0 --split val --cache-dir caches/cache_norm4_est_s42 --out-dir <run>/fold0/preds_val
#     python size_normalization_approach/predict.py --checkpoint ... --case <case id> --out-dir preds
#     python size_normalization_approach/predict.py --checkpoint ... --all --out-dir preds
#
# --cache-dir is the grid the network works on: caches/cache/ for the plain x4 pass, a
# size-normalised cache built from estimated volumes (cache_norm4_est_s<seed>) for the
# second pass.
#
# Output: <out-dir>/<case>.npz with
#     sdf      the predicted SDF, normalised to [-1, 1], at native resolution
#     affine   the native affine (the same as caches/cache/x1/<case>.npz)
# size_normalization_approach/evaluate.py scores these files.

import argparse
import json
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
from dataset import CUBE, FACTOR  # noqa: E402
from geometry import apply_resample, resample_grid  # noqa: E402
from model import UNet3D  # noqa: E402
from fine_input import FINE, load_fine, pack  # noqa: E402
from nnunet_augmentation import pad_to_cube  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


# The native shape and affine of a case, without loading its voxels.
# An .npz is a zip of .npy files, and an .npy header already records the array's shape;
# reading only the header takes 0.5 ms instead of 50 ms to decompress 78 MB.
def native_geometry(cache_dir: str, cid: str):
    path = os.path.join(cache_dir, "x1", f"{cid}.npz")
    with np.load(path) as d:
        affine = d["affine"]
    with zipfile.ZipFile(path) as z, z.open("img.npy") as f:
        major, _ = npy_format.read_magic(f)
        read_header = (npy_format.read_array_header_1_0 if major == 1
                       else npy_format.read_array_header_2_0)
        shape, _, _ = read_header(f)
    return tuple(shape), affine


# Predict one case on its coarse grid and carry the field to the native grid.
# Returns (native SDF as float32, native affine).
@torch.no_grad()
# A network trained with --fine-input (8 input channels) gets the 2x finer image packed
# into channels (fine_input.py) instead of the x4 image. `factor` and `cube` are the grid
# the network was trained on; a mask network (loss "mask") outputs logits, whose
# probability p is carried as the field 1 - 2p (negative inside, like an SDF), so the
# same resampling and threshold at 0 apply: interpolating 1 - 2p and cutting at 0 is
# interpolating p and cutting at 0.5.
def predict_one(model, cache_dir: str, cid: str, device: torch.device,
                factor: int = FACTOR, cube: int = CUBE, loss: str = "sdf"):
    img4, affine4 = load_image(cache_dir, cid, factor)
    native_shape, affine1 = native_geometry(cache_dir, cid)

    # Pad to the network's cube, predict, and cut the padding off again.
    if model.enc1.block[0].in_channels == FINE ** 3:
        fine = pad_to_cube(load_fine(cache_dir, cid), FINE * CUBE, cval=0.0)
        x = torch.from_numpy(pack(fine)[np.newaxis]).float().to(device)
    else:
        img_padded = pad_to_cube(img4.astype(np.float32), cube, cval=0.0)
        x = torch.from_numpy(img_padded[np.newaxis, np.newaxis]).float().to(device)
    pred = model(x) if loss == "sdf" else 1 - 2 * torch.sigmoid(model.logits(x))
    pred = pred[:, :, :img4.shape[0], :img4.shape[1], :img4.shape[2]]

    # Back to native resolution through the two affines, on the GPU (MONAI's own
    # resampler builds the same grid on the CPU in float64: 563 ms per case against the
    # network's 16 ms). cval=1: beyond the scan, the SDF is "far outside".
    grid = resample_grid(affine4, affine1, native_shape, device=device).float()
    native = apply_resample(pred, grid, img4.shape, cval=1.0)

    return native[0, 0].cpu().numpy().astype(np.float32), affine1


def main():
    p = argparse.ArgumentParser(description="Predict native-resolution SDFs with a trained network.")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--case",        type=str, default=None, help="a single case id")
    p.add_argument("--all",         action="store_true", help="every cached case")
    p.add_argument("--fold",        type=int, default=None, help="one fold's cases")
    p.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--splits-json", type=str, default=os.path.join(PARENT, "splits.json"))
    p.add_argument("--cache-dir",   type=str, default=os.path.join(PARENT, "caches", "cache"))
    p.add_argument("--out-dir",     type=str, required=True)
    a = p.parse_args()

    if sum(x is not None for x in (a.case, a.fold)) + int(a.all) != 1:
        p.error("pass exactly one of --case <cid>, --all, or --fold <k>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A checkpoint trained with a contour head carries its weights; the head is built so
    # the checkpoint loads, but only the SDF is predicted.
    state = torch.load(a.checkpoint, map_location=device)
    # Input channels and width read from the checkpoint itself (first convolution).
    model = UNet3D(in_channels=state["enc1.block.0.weight"].shape[1],
                   base_channels=state["enc1.block.0.weight"].shape[0],
                   contour_head="contour_head.weight" in state).to(device)
    model.load_state_dict(state)
    model.eval()

    # The grid and output the network was trained with (train.py writes config.json in the
    # run directory, two levels above the checkpoint); runs from before it are x4 SDF.
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint))),
                            "config.json")
    cfg = {"factor": FACTOR, "cube": CUBE, "loss": "sdf"}
    if os.path.exists(cfg_path):
        cfg.update(json.load(open(cfg_path)))
    print(f"Grid x{cfg['factor']} in {cfg['cube']}^3, output: {cfg['loss']}")

    os.makedirs(a.out_dir, exist_ok=True)

    # The cases to predict.
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

    prof = Profiler("predict")
    with prof:
        for i, cid in enumerate(cids, start=1):
            with prof.span("predict_case") as s:
                sdf, affine = predict_one(model, a.cache_dir, cid, device,
                                          cfg["factor"], cfg["cube"], cfg["loss"])
                s["items"] = 1
            np.savez(os.path.join(a.out_dir, f"{cid}.npz"), sdf=sdf, affine=affine)
            print(f"[{i}/{len(cids)}] {cid[:44]:<44} native={sdf.shape}", flush=True)

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")


if __name__ == "__main__":
    main()
