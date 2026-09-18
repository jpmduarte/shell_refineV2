"""Native-resolution crops around phase 1's predicted band, for phase 2 training/eval.

Reads phase 1's saved predictions (predict.py's output: <pred_dir>/<cid>.npz with
sdf + affine, already upsampled to native resolution) and cache/x1 + cache/label,
crops a tight box around the band plus a padding margin, and saves everything phase 2
needs in one place.

    python phase2/predict.py ...              # phase 1 predictions already exist
    python phase2/make_crops.py --pred-dir phase1/runs/fold0/preds_val --out-dir phase2/crops/fold0_val

    phase2/crops/<run>/<cid>.npz:
        img          native image, cropped, re-normalised within the crop
        sdf_prior    phase 1's predicted SDF, normalised [-1,1], cropped
        coarse_mask  sdf_prior < 0 — phase 1's fallback outside the refined band
        band         |sdf_prior_mm| < band_mm — where phase 2 is allowed to act
        mask         ground truth, cropped
        bbox         crop's (lo, hi) per axis in the native volume
        native_shape the case's full native shape (to place predictions back)
        spacing      native voxel spacing in mm
        trunc_mm     the SDF's truncation distance in mm
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import load as load_image, load_label  # noqa: E402

TRUNC_MM = 10.0


def normalize(img: np.ndarray) -> np.ndarray:
    mean, std = np.mean(img), np.std(img)
    if std == 0:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mean) / std).astype(np.float32)


def bbox_from_mask(mask: np.ndarray, padding: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return [(0, s) for s in mask.shape]
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    return [(max(0, int(lo) - padding), min(s, int(hi) + padding + 1))
           for lo, hi, s in zip(mins, maxs, mask.shape)]


def build_one(cid: str, cache_dir: str, pred_dir: str, out_dir: str,
             padding: int, band_mm: float, trunc_mm: float, verbose: bool = False):
    img_native, native_affine = load_image(cache_dir, cid, 1)
    label, _ = load_label(cache_dir, cid)
    mask_native = (label[0] if label.ndim == 4 else label).astype(bool)
    native_shape = img_native.shape
    spacing = np.abs(np.diag(native_affine[:3, :3]))

    pred = np.load(os.path.join(pred_dir, f"{cid}.npz"))
    sdf_prior = pred["sdf"].astype(np.float32)  # normalised [-1, 1], native resolution
    sdf_mm = sdf_prior * trunc_mm

    # Same rule as v1: crop around interior + band, so padding smaller than band_mm
    # still can't clip the outer half of the band.
    bbox = bbox_from_mask(sdf_mm < band_mm, padding)
    slices = tuple(slice(lo, hi) for lo, hi in bbox)

    img_crop   = normalize(img_native[slices])
    mask_crop  = mask_native[slices].astype(np.uint8)
    prior_crop = sdf_prior[slices].astype(np.float32)

    coarse_crop = (prior_crop < 0).astype(np.uint8)
    band_crop   = (np.abs(sdf_mm[slices]) < band_mm).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{cid}.npz"),
        img=img_crop, sdf_prior=prior_crop,
        coarse_mask=coarse_crop, band=band_crop, mask=mask_crop,
        bbox=np.array(bbox, dtype=np.int32),
        native_shape=np.array(native_shape, dtype=np.int32),
        spacing=np.array(spacing, dtype=np.float32),
        trunc_mm=np.float32(trunc_mm),
    )

    if verbose:
        crop_shape = tuple(hi - lo for lo, hi in bbox)
        print(f"  {cid[:44]:<44}  native={native_shape}  crop={crop_shape}"
             f"  band={band_crop.sum():,}vox", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir",  type=str, required=True, help="phase 1 predictions (predict.py output)")
    p.add_argument("--cache-dir", type=str, default=os.path.join(PARENT, "cache"))
    p.add_argument("--out-dir",   type=str, required=True)
    p.add_argument("--padding",   type=int, default=10, help="voxel margin around the band bbox")
    p.add_argument("--band-mm",   type=float, default=5.0)
    p.add_argument("--trunc-mm",  type=float, default=TRUNC_MM)
    a = p.parse_args()

    cids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.pred_dir, "*.npz")))
    print(f"{len(cids)} cases -> {a.out_dir}\npadding={a.padding}vox  band_mm={a.band_mm}\n", flush=True)

    for i, cid in enumerate(cids, start=1):
        print(f"[{i}/{len(cids)}]", end="", flush=True)
        build_one(cid, a.cache_dir, a.pred_dir, a.out_dir, a.padding, a.band_mm, a.trunc_mm, verbose=True)

    print(f"\n{len(cids)} crops created -> {a.out_dir}")


if __name__ == "__main__":
    main()
