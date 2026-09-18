"""Phase 2 inference: tile the crop, run the refiner on tiles that touch the band,
average overlapping predictions, and stitch the result over phase 1's coarse mask.

Tiles are a deterministic grid at --stride (default 48 = 3/4 of a 64 patch). v1 used
stride 32 (50% overlap), which costs ~2.5x more tiles for the same coverage; 48 still
gives every interior voxel 2-3 overlapping predictions to average over.

Everything outside the band keeps phase 1's answer, so no compute is spent deep inside
or far outside the object.

    python phase2/predict.py --checkpoint phase2/runs/fold0/checkpoints/phase2_best.pth \\
        --crops-dir phase2/crops/all --fold 0 --split val --out-dir phase2/runs/fold0/preds_val

Output: <out_dir>/<case>.npz
    refined   per-voxel probability after refinement, crop-shaped
    coarse    phase 1's mask alone, crop-shaped (the baseline to beat)
    band      where refinement was allowed to act
    mask      ground truth
    spacing   native voxel spacing in mm
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from dataset import pad_to_patch, PATCH_SIZE  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402

DEFAULT_STRIDE = 48


def axis_origins(dim: int, patch: int, stride: int) -> list[int]:
    last = max(dim - patch, 0)
    stops = list(range(0, last + 1, stride))
    if stops[-1] != last:
        stops.append(last)
    return stops


def tile_origins(shape, patch: int, stride: int):
    return itertools.product(*(axis_origins(d, patch, stride) for d in shape))


@torch.no_grad()
def predict_case(model, crop_path: str, device, patch: int, stride: int, tile_batch: int):
    data = np.load(crop_path)
    mask    = data["mask"].astype(np.float32)
    spacing = tuple(data["spacing"].tolist())
    crop_shape = data["img"].shape

    arrays = pad_to_patch({
        "img":       data["img"].astype(np.float32),
        "sdf_prior": data["sdf_prior"].astype(np.float32),
        "band":        data["band"].astype(np.float32),
        "coarse_mask": data["coarse_mask"].astype(np.float32),
    }, patch)
    img_p, prior_p, coarse_p = arrays["img"], arrays["sdf_prior"], arrays["coarse_mask"]
    band = arrays["band"] > 0.5

    accum = np.zeros(img_p.shape, dtype=np.float32)
    count = np.zeros(img_p.shape, dtype=np.float32)

    origins = [o for o in tile_origins(img_p.shape, patch, stride)
              if band[tuple(slice(s, s + patch) for s in o)].any()]

    for start in range(0, len(origins), tile_batch):
        chunk  = origins[start:start + tile_batch]
        slices = [tuple(slice(s, s + patch) for s in o) for o in chunk]
        batch  = np.stack([np.stack([img_p[sl], prior_p[sl]]) for sl in slices])
        preds  = model(torch.from_numpy(batch).to(device)).squeeze(1).cpu().numpy()
        for sl, pred in zip(slices, preds):
            accum[sl] += pred
            count[sl] += 1.0

    refined = coarse_p.copy()
    touched = band & (count > 0)
    refined[touched] = accum[touched] / count[touched]

    unpad = tuple(slice(0, s) for s in crop_shape)
    return (refined[unpad], coarse_p[unpad], band[unpad], mask, spacing, len(origins))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--crops-dir",   type=str, required=True)
    p.add_argument("--out-dir",     type=str, required=True)
    p.add_argument("--fold",        type=int, default=None, help="restrict to one fold's split")
    p.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--patch-size",  type=int, default=PATCH_SIZE, help="must match training")
    p.add_argument("--stride",      type=int, default=DEFAULT_STRIDE)
    p.add_argument("--tile-batch",  type=int, default=8)
    p.add_argument("--base-ch",     type=int, default=16)
    p.add_argument("--splits-json", type=str, default=os.path.join(REPO_ROOT, "splits.json"))
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=2, base_channels=a.base_ch, out_activation="sigmoid").to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    available = sorted(f[:-4] for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    if a.fold is not None:
        folds = get_folds(available, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
        train_ids, val_ids = fold_split(folds, a.fold)
        wanted = val_ids if a.split == "val" else train_ids
        cids = [c for c in wanted if c in set(available)]
    else:
        cids = available

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"Checkpoint: {a.checkpoint}")
    print(f"Cases: {len(cids)}   tiles {a.patch_size}^3 stride={a.stride}")
    print(f"Model: {model_summary(model)}")

    prof = Profiler("phase2_predict")
    with prof:
        for i, cid in enumerate(cids, start=1):
            with prof.span("predict_case") as s:
                refined, coarse, band, mask, spacing, n_tiles = predict_case(
                    model, os.path.join(a.crops_dir, f"{cid}.npz"),
                    device, a.patch_size, a.stride, a.tile_batch)
                s["items"] = n_tiles
            np.savez_compressed(os.path.join(a.out_dir, f"{cid}.npz"),
                                refined=refined, coarse=coarse, band=band.astype(np.uint8),
                                mask=mask, spacing=np.array(spacing, dtype=np.float32))
            print(f"[{i}/{len(cids)}] {cid[:44]:<44} tiles={n_tiles}", flush=True)

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")


if __name__ == "__main__":
    main()
