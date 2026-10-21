"""Refine the band of each crop: tile it with overlapping cubes and blend their predictions.

Cubes overlap by --overlap of their side, and only cubes touching the band run. The predictions are
averaged with a Gaussian weight peaking at each cube's centre (--blend-sigma, a fraction of the side;
0 = uniform), so each voxel is decided mostly by the cube that saw the most around it. Outside the
band, and where no cube reached, the prior's mask is kept. The network is rebuilt from the run's
config.json. Output: <out-dir>/<case>.npz with refined (probability), coarse, band, mask and spacing.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from boundary_refiner.dataset import CUBE, cube_slices, pad_to_shape
from boundary_refiner.refiner_model import RefinerUNet
from common.folds import SEED as FOLDS_SEED, fold_split, get_folds
from common.paths import SPLITS_JSON
from common.profiler import Profiler, model_summary

SPATIAL_DIMS = (2, 3, 4)   # of a (batch, channel, z, y, x) block
FLIPS = [[dim for dim, flip in zip(SPATIAL_DIMS, flips) if flip] for flips in itertools.product((0, 1), repeat=3)]
MIN_BLEND_WEIGHT = 1e-3    # so a voxel reached only by cube corners is still decided
PROBABILITY_CHANNEL = 0

USAGE = """\
example:
  python boundary_refiner/predict.py \\
      --checkpoint boundary_refiner/runs/x4_s42_cube/fold0/checkpoints/phase2_last.pth \\
      --crops-dir boundary_refiner/crops/x4_s42 --fold 0 --out-dir boundary_refiner/runs/x4_s42_cube/fold0/preds_val"""


def gaussian_weight(side: int, sigma_frac: float) -> np.ndarray:
    """1 at the cube's centre; at sigma 1/8 of the side a corner weighs about 2% of it."""
    x = (np.arange(side, dtype=np.float32) - (side - 1) / 2.0) / side
    g = np.exp(-0.5 * (x / sigma_frac) ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return np.maximum(w, MIN_BLEND_WEIGHT).astype(np.float32)


def axis_starts(length: int, side: int, stride: int) -> list[int]:
    """Every `stride`, plus a last start flush with the end."""
    last = max(length - side, 0)
    starts = list(range(0, last + 1, max(1, stride)))
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_model(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """Probabilities for a batch of cubes; with TTA, the mean over the 8 mirrorings."""
    if not tta:
        return model(x)
    total = torch.zeros_like(x[:, :1])
    for dims in FLIPS:
        flipped = torch.flip(x, dims) if dims else x
        y = model(flipped)
        total += torch.flip(y, dims) if dims else y
    return total / len(FLIPS)


@torch.no_grad()
def predict_case(model, crop_path: str, device, cube: int, overlap: float,
                 sigma_frac: float, batch: int, tta: bool = False):
    """(refined probability, prior mask, band, true mask, spacing, number of cubes, share of the band covered)."""
    with np.load(crop_path) as crop:
        arrays = {k: crop[k].astype(np.float32) for k in ("img", "sdf_prior", "band", "mask")}
        coarse = crop["coarse_mask"].astype(np.float32)
        spacing = crop["spacing"]
    unpadded = tuple(slice(0, s) for s in arrays["img"].shape)
    padded = pad_to_shape(arrays, cube)
    img, prior, band = padded["img"], padded["sdf_prior"], padded["band"] > 0.5

    refined = np.zeros(img.shape, dtype=np.float32)
    refined[unpadded] = coarse

    stride = max(1, int(cube * (1.0 - overlap)))
    corners = [o for o in itertools.product(*(axis_starts(n, cube, stride) for n in img.shape))
               if band[cube_slices(o, cube)].any()]

    weight = gaussian_weight(cube, sigma_frac) if sigma_frac > 0 else np.ones((cube,) * 3, np.float32)
    weighted_sum = np.zeros(img.shape, dtype=np.float32)
    weight_sum = np.zeros(img.shape, dtype=np.float32)

    for start in range(0, len(corners), batch):
        boxes = [cube_slices(o, cube) for o in corners[start:start + batch]]
        x = np.stack([np.stack([img[box], prior[box]]) for box in boxes])
        probabilities = run_model(model, torch.from_numpy(x).to(device), tta)[:, PROBABILITY_CHANNEL].cpu().numpy()
        for box, p in zip(boxes, probabilities):
            weighted_sum[box] += weight * p
            weight_sum[box] += weight

    reached = band & (weight_sum > 0)
    refined[reached] = weighted_sum[reached] / weight_sum[reached]

    coverage = float(reached.sum()) / float(band.sum()) if band.any() else 1.0
    return refined[unpadded], coarse, band[unpadded], arrays["mask"], spacing, len(corners), coverage


def load_refiner(checkpoint, device):
    """(the network as trained, from <run>/config.json two levels above the checkpoint; that config)."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint))), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    model = RefinerUNet(base_channels=config.get("base_ch", 16), depth=config.get("depth", 3),
                        kernel=config.get("kernel", 3)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval(), config


def select_cases(args) -> list[str]:
    available = sorted(os.path.splitext(f)[0] for f in os.listdir(args.crops_dir) if f.endswith(".npz"))
    if args.fold is None:
        return available
    folds = get_folds(available, n_folds=args.n_folds, seed=FOLDS_SEED, splits_json=args.splits_json)
    train_ids, val_ids = fold_split(folds, args.fold)
    wanted = val_ids if args.split == "val" else train_ids
    return [c for c in wanted if c in set(available)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--crops-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None, help="one fold's cases; default all")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cube", type=int, default=None, help="default: from the run's config.json (else 64)")
    parser.add_argument("--tta", action="store_true", help="average the 8 mirrored predictions")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="fraction of the cube's side shared by neighbouring cubes")
    parser.add_argument("--blend-sigma", type=float, default=0.125,
                        help="Gaussian blending width as a fraction of the cube's side; 0 = uniform")
    parser.add_argument("--tile-batch", type=int, default=8, help="cubes per forward pass")
    parser.add_argument("--splits-json", type=str, default=SPLITS_JSON)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_refiner(args.checkpoint, device)
    cube = args.cube or config.get("cube", CUBE)
    cids = select_cases(args)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Model: {model_summary(model)} {config}  cubes {cube}^3, overlap {args.overlap}, "
          f"blend sigma {args.blend_sigma}, TTA {args.tta}")

    coverages = []
    profiler = Profiler("refiner_predict")
    with profiler:
        for i, cid in enumerate(cids, start=1):
            with profiler.span("predict_case") as span:
                refined, coarse, band, mask, spacing, n_cubes, coverage = predict_case(
                    model, os.path.join(args.crops_dir, f"{cid}.npz"), device,
                    cube, args.overlap, args.blend_sigma, args.tile_batch, args.tta)
                span["items"] = 1
            np.savez(os.path.join(args.out_dir, f"{cid}.npz"), refined=refined, coarse=coarse,
                     band=band.astype(np.uint8), mask=mask, spacing=spacing)
            coverages.append(coverage)
            print(f"[{i}/{len(cids)}] {cid[:40]:<40} cubes {n_cubes:4d}  band covered "
                  f"{coverage:.1%}", flush=True)

    print(f"\n{len(cids)} case(s) -> {args.out_dir}")
    print(f"band coverage: mean {np.mean(coverages):.1%}  worst {np.min(coverages):.1%}")


if __name__ == "__main__":
    main()
