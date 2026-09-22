"""Phase 2 inference with normal-aligned slabs.

Cubes tile with one grid; slabs cannot, because the orientation that matches the band
changes across the surface. So this lays one grid per orientation and keeps, from each,
only the tiles whose thin axis matches the local normal at their centre — the same rule
the training sampler uses, so inference sees patches shaped like the ones trained on.

Voxels no kept tile covers keep phase 1's mask, which is the safe default. Coverage is
reported per case: if it is not near-complete the tiling needs revisiting, not quiet
acceptance.

    python phase2/predict_slab.py --checkpoint phase2/runs/slab_fold0/checkpoints/phase2_best.pth \\
        --crops-dir phase2/crops/all --fold 0 --split val --out-dir phase2/runs/slab_fold0/preds_val
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

from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from slab import local_normal_axis, LONG, pad_to_shape, THIN  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def axis_origins(dim: int, extent: int, stride: int) -> list[int]:
    last = max(dim - extent, 0)
    stops = list(range(0, last + 1, max(1, stride)))
    if stops[-1] != last:
        stops.append(last)
    return stops


@torch.no_grad()
def predict_case(model, crop_path: str, device, long: int, thin: int,
                 overlap: float, tile_batch: int):
    # Only what inference reads: mask and sdf_true are for evaluation, and skipping
    # them saves about 50ms a case.
    with np.load(crop_path) as _z:
        data = {k: _z[k] for k in ("img", "sdf_prior", "band", "coarse_mask",
                                   "spacing", "bbox", "native_shape", "trunc_mm")
                if k in _z.files}
        data["mask"] = _z["mask"]
    mask    = data["mask"].astype(np.float32)
    spacing = tuple(data["spacing"].tolist())
    crop_shape = data["img"].shape

    arrays = pad_to_shape({
        "img":         data["img"].astype(np.float32),
        "sdf_prior":   data["sdf_prior"].astype(np.float32),
        "band":        data["band"].astype(np.float32),
        "coarse_mask": data["coarse_mask"].astype(np.float32),
    }, long)
    img_p, prior_p = arrays["img"], arrays["sdf_prior"]
    coarse_p, band = arrays["coarse_mask"], arrays["band"] > 0.5

    accum = np.zeros(img_p.shape, dtype=np.float32)
    count = np.zeros(img_p.shape, dtype=np.float32)

    jobs = []
    for axis in (0, 1, 2):
        shape = [long, long, long]
        shape[axis] = thin
        strides = [max(1, int(e * (1.0 - overlap))) for e in shape]
        for origin in itertools.product(*(axis_origins(d, e, s)
                                          for d, e, s in zip(img_p.shape, shape, strides))):
            sl = tuple(slice(o, o + e) for o, e in zip(origin, shape))
            if not band[sl].any():
                continue
            centre = tuple(o + e // 2 for o, e in zip(origin, shape))
            # Keep this tile only where its orientation is the one training would have
            # picked here; another orientation's grid covers the rest.
            if local_normal_axis(prior_p, centre) != axis:
                continue
            jobs.append((sl, axis))

    for start in range(0, len(jobs), tile_batch):
        chunk = jobs[start:start + tile_batch]
        batch = np.stack([
            np.stack([np.moveaxis(img_p[sl], ax, -1), np.moveaxis(prior_p[sl], ax, -1)])
            for sl, ax in chunk
        ])
        preds = model(torch.from_numpy(batch).to(device)).squeeze(1).cpu().numpy()
        for (sl, ax), pred in zip(chunk, preds):
            accum[sl] += np.moveaxis(pred, -1, ax)
            count[sl] += 1.0

    refined = coarse_p.copy()
    touched = band & (count > 0)
    refined[touched] = accum[touched] / count[touched]

    coverage = float(touched.sum()) / float(band.sum()) if band.any() else 1.0
    unpad = tuple(slice(0, s) for s in crop_shape)
    return (refined[unpad], coarse_p[unpad], band[unpad], mask, spacing,
            len(jobs), coverage)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--crops-dir",   type=str, required=True)
    p.add_argument("--out-dir",     type=str, required=True)
    p.add_argument("--fold",        type=int, default=None)
    p.add_argument("--split",       type=str, default="val", choices=["train", "val"])
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--long",        type=int, default=LONG, help="must match training")
    p.add_argument("--thin",        type=int, default=THIN, help="must match training")
    p.add_argument("--overlap",     type=float, default=0.25,
                   help="fraction of each extent shared between neighbouring tiles")
    p.add_argument("--tile-batch",  type=int, default=4)
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
    print(f"Cases: {len(cids)}   slabs {a.long}x{a.long}x{a.thin}  overlap={a.overlap}")
    print(f"Model: {model_summary(model)}")

    coverages = []
    prof = Profiler("phase2_slab_predict")
    with prof:
        for i, cid in enumerate(cids, start=1):
            with prof.span("predict_case") as s:
                refined, coarse, band, mask, spacing, n_tiles, cov = predict_case(
                    model, os.path.join(a.crops_dir, f"{cid}.npz"),
                    device, a.long, a.thin, a.overlap, a.tile_batch)
                s["items"] = n_tiles
            coverages.append(cov)
            np.savez_compressed(os.path.join(a.out_dir, f"{cid}.npz"),
                                refined=refined, coarse=coarse, band=band.astype(np.uint8),
                                mask=mask, spacing=np.array(spacing, dtype=np.float32))
            print(f"[{i}/{len(cids)}] {cid[:40]:<40} tiles={n_tiles:>5}"
                  f"  band covered={100*cov:.1f}%", flush=True)

    print(f"\n{len(cids)} case(s) -> {a.out_dir}")
    print(f"band coverage: mean {100*np.mean(coverages):.1f}%  "
          f"worst {100*np.min(coverages):.1f}%")


if __name__ == "__main__":
    main()
