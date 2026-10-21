# Phase 2 inference: refines the band of each crop with the trained refiner, tiling it
# with cubes and blending the overlapping predictions.
#
#     python boundary_refiner/predict.py \
#         --checkpoint boundary_refiner/runs/x4_s42_cube/fold0/checkpoints/phase2_last.pth \
#         --crops-dir boundary_refiner/crops/x4_s42 --fold 0 \
#         --out-dir boundary_refiner/runs/x4_s42_cube/fold0/preds_val
#
# Tiling: the crop is covered by a regular grid of cubes, neighbours overlapping by
# --overlap of their side; cubes that contain no band voxel are skipped. Every voxel of
# the band is predicted by several cubes.
#
# Blending: those predictions are averaged with a Gaussian weight that peaks at each
# cube's centre (--blend-sigma, as a fraction of the cube's side). A voxel near a cube's
# edge was decided with its context cut off on one side; weighting by distance from the
# centre lets each voxel be decided mostly by the cube that saw the most around it, and
# hides the seams where one cube ends and the next begins. --blend-sigma 0 averages
# uniformly.
#
# Outside the band, and anywhere no cube reached, the prior's own mask is kept.
#
# TTA (--tta): each cube is also predicted mirrored along every combination of the three
# axes (8 versions), each prediction mirrored back, and the 8 averaged, as phase 1 does.
#
# The network is rebuilt from config.json beside the checkpoint's run (written by
# train_cube.py), so the architecture arms need no extra flags here.
#
# Output: <out-dir>/<case>.npz with refined (probability), coarse (the prior's mask),
# band, mask (ground truth) and spacing, the format evaluate.py reads.

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import SPLITS_JSON  # noqa: E402

from boundary_refiner.dataset import CUBE, pad_to_shape  # noqa: E402
from common.profiler import Profiler, model_summary  # noqa: E402
from boundary_refiner.refiner_model import RefinerUNet  # noqa: E402
from common.folds import fold_split, get_folds  # noqa: E402


# Gaussian weight over a cube, 1 at the centre. sigma is a fraction of the side, so at
# 1/8 a corner weighs about 2% of the centre; floored above zero so a voxel reached only
# by cube corners is still decided.
def gaussian_weight(side: int, sigma_frac: float) -> np.ndarray:
    x = (np.arange(side, dtype=np.float32) - (side - 1) / 2.0) / side
    g = np.exp(-0.5 * (x / sigma_frac) ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return np.maximum(w, 1e-3).astype(np.float32)


# Start positions along one axis: every `stride`, plus a last one flush with the end.
def axis_starts(length: int, side: int, stride: int) -> list[int]:
    last = max(length - side, 0)
    starts = list(range(0, last + 1, max(1, stride)))
    if starts[-1] != last:
        starts.append(last)
    return starts


# The 8 mirrorings for TTA, as tensor dims of a (batch, channel, z, y, x) block: every
# subset of the three spatial axes, from none to all three.
FLIPS = [[d for d, f in zip((2, 3, 4), fl) if f] for fl in itertools.product((0, 1), repeat=3)]


# The network's probabilities for a batch of cubes; with TTA, the mean over the 8
# mirrored versions, each mirrored back first.
def run_model(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    if not tta:
        return model(x)
    total = torch.zeros_like(x[:, :1])
    for dims in FLIPS:
        xi = torch.flip(x, dims) if dims else x
        y = model(xi)
        total += torch.flip(y, dims) if dims else y
    return total / len(FLIPS)


# Refine one crop. Returns (refined probability, prior mask, band, true mask, spacing,
# number of cubes, fraction of the band covered).
@torch.no_grad()
def predict_case(model, crop_path: str, device, cube: int, overlap: float,
                 sigma_frac: float, batch: int, tta: bool = False):
    with np.load(crop_path) as z:
        arrays = {k: z[k].astype(np.float32) for k in ("img", "sdf_prior", "band", "mask")}
        coarse = z["coarse_mask"].astype(np.float32)
        spacing = z["spacing"]
    crop_shape = arrays["img"].shape
    padded = pad_to_shape(arrays, cube)
    img, prior, band = padded["img"], padded["sdf_prior"], padded["band"] > 0.5

    # The prior's mask, padded the same way: the answer wherever the refiner does not act.
    refined = np.zeros(img.shape, dtype=np.float32)
    refined[tuple(slice(0, s) for s in crop_shape)] = coarse

    # Every cube of the grid that touches the band.
    stride = max(1, int(cube * (1.0 - overlap)))
    corners = [o for o in itertools.product(*(axis_starts(n, cube, stride) for n in img.shape))
               if band[tuple(slice(oi, oi + cube) for oi in o)].any()]

    weight = gaussian_weight(cube, sigma_frac) if sigma_frac > 0 else np.ones((cube,) * 3, np.float32)
    accum = np.zeros(img.shape, dtype=np.float32)    # sum of weight x prediction
    total = np.zeros(img.shape, dtype=np.float32)    # sum of weights

    for start in range(0, len(corners), batch):
        chunk = corners[start:start + batch]
        slices = [tuple(slice(oi, oi + cube) for oi in o) for o in chunk]
        x = np.stack([np.stack([img[sl], prior[sl]]) for sl in slices])
        probs = run_model(model, torch.from_numpy(x).to(device), tta)[:, 0].cpu().numpy()
        for sl, p in zip(slices, probs):
            accum[sl] += weight * p
            total[sl] += weight

    # Inside the band, where some cube reached: the blended prediction.
    reached = band & (total > 0)
    refined[reached] = accum[reached] / total[reached]

    coverage = float(reached.sum()) / float(band.sum()) if band.any() else 1.0
    unpad = tuple(slice(0, s) for s in crop_shape)
    return (refined[unpad], coarse, band[unpad], arrays["mask"], spacing, len(corners), coverage)


def main():
    p = argparse.ArgumentParser(description="Refine the band of each crop, with blending.")
    p.add_argument("--checkpoint",  type=str,   required=True)
    p.add_argument("--crops-dir",   type=str,   required=True)
    p.add_argument("--out-dir",     type=str,   required=True)
    p.add_argument("--fold",        type=int,   default=None, help="one fold's cases; default all")
    p.add_argument("--split",       type=str,   default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int,   default=5)
    p.add_argument("--cube",        type=int,   default=None,
                   help="default: from the run's config.json (else 64)")
    p.add_argument("--tta",         action="store_true", help="average the 8 mirrored predictions")
    p.add_argument("--overlap",     type=float, default=0.5,
                   help="fraction of the cube's side shared by neighbouring cubes")
    p.add_argument("--blend-sigma", type=float, default=0.125,
                   help="Gaussian blending width as a fraction of the cube's side; 0 = uniform")
    p.add_argument("--tile-batch",  type=int,   default=8, help="cubes per forward pass")
    p.add_argument("--splits-json", type=str,   default=SPLITS_JSON)
    a = p.parse_args()

    # The network as it was trained: <run>/config.json, the run being two levels above
    # <run>/checkpoints/phase2_last.pth. Older runs without one used the defaults.
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint))),
                               "config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    cube = a.cube or config.get("cube", CUBE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefinerUNet(base_channels=config.get("base_ch", 16), depth=config.get("depth", 3),
                        kernel=config.get("kernel", 3)).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    # The cases: one fold's validation (or training) cases, or every crop.
    available = sorted(f[:-4] for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    if a.fold is not None:
        folds = get_folds(available, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
        train_ids, val_ids = fold_split(folds, a.fold)
        wanted = val_ids if a.split == "val" else train_ids
        cids = [c for c in wanted if c in set(available)]
    else:
        cids = available

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"Model: {model_summary(model)} {config}  cubes {cube}^3, overlap {a.overlap}, "
          f"blend sigma {a.blend_sigma}, TTA {a.tta}")

    coverages = []
    prof = Profiler("refiner_predict")
    with prof:
        for i, cid in enumerate(cids, start=1):
            with prof.span("predict_case") as s:
                refined, coarse, band, mask, spacing, n_cubes, coverage = predict_case(
                    model, os.path.join(a.crops_dir, f"{cid}.npz"), device,
                    cube, a.overlap, a.blend_sigma, a.tile_batch, a.tta)
                s["items"] = 1
            np.savez(os.path.join(a.out_dir, f"{cid}.npz"), refined=refined, coarse=coarse,
                     band=band.astype(np.uint8), mask=mask, spacing=spacing)
            coverages.append(coverage)
            print(f"[{i}/{len(cids)}] {cid[:40]:<40} cubes {n_cubes:4d}  band covered "
                  f"{coverage:.1%}", flush=True)

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")
    print(f"band coverage: mean {np.mean(coverages):.1%}  worst {np.min(coverages):.1%}")


if __name__ == "__main__":
    main()
