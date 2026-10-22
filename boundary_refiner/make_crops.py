"""Native-resolution crops around a first pass's predicted band, everything the refiner needs per case.

The prior (--prior-dir) is any first pass's saved predictions, <prior-dir>/<case>.npz with a native
SDF and affine, e.g. make_priors.py's. Each crop is a tight box around the prior's interior and band
plus --padding voxels. <out-dir>/<case>.npz holds:
    img           native image, cropped, re-normalised within the crop
    sdf_prior     the prior's SDF, normalised, cropped
    sdf_true      the true SDF, cropped
    coarse_mask   sdf_prior < 0: the answer outside the refined band
    band          |sdf_prior| < band_mm: where the refiner may act
    mask          ground truth, cropped
    bbox          the crop's (lo, hi) per axis in the native volume
    native_shape, spacing, trunc_mm
"""
import argparse
import glob
import os

import numpy as np

from common.geometry import voxel_spacing
from common.image_cache import load as load_image, load_label
from common.sdf_targets import TRUNC_MM, load as load_sdf_target
from datasets import add_dataset_argument, get as get_dataset

from boundary_refiner.dataset import BAND_MM

USAGE = """\
example:
  python boundary_refiner/make_priors.py --seeds 42
  python boundary_refiner/make_crops.py --prior-dir boundary_refiner/priors/x4_s42 \\
      --out-dir boundary_refiner/crops/x4_s42"""


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


def build_one(cid: str, cache_dir: str, prior_dir: str, out_dir: str,
              padding: int, band_mm: float, trunc_mm: float, verbose: bool = False):
    img_native, native_affine = load_image(cache_dir, cid, 1)
    label, _ = load_label(cache_dir, cid)
    mask_native = label.astype(bool)
    native_shape = img_native.shape

    with np.load(os.path.join(prior_dir, f"{cid}.npz")) as prediction:
        sdf_prior = prediction["sdf"].astype(np.float32)
    sdf_mm = sdf_prior * trunc_mm

    # Around interior + band, so a padding smaller than band_mm still cannot clip the band's outer half.
    bbox = bbox_from_mask(sdf_mm < band_mm, padding)
    box = tuple(slice(lo, hi) for lo, hi in bbox)

    sdf_true, _ = load_sdf_target(cache_dir, cid, 1)
    prior_crop = sdf_prior[box].astype(np.float32)
    band_crop = (np.abs(sdf_mm[box]) < band_mm).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    # Uncompressed: zlib made a crop load in 484 ms instead of 142, more than the network takes.
    np.savez(
        os.path.join(out_dir, f"{cid}.npz"),
        img=normalize(img_native[box]), sdf_prior=prior_crop, sdf_true=sdf_true[box].astype(np.float32),
        coarse_mask=(prior_crop < 0).astype(np.uint8), band=band_crop,
        mask=mask_native[box].astype(np.uint8),
        bbox=np.array(bbox, dtype=np.int32),
        native_shape=np.array(native_shape, dtype=np.int32),
        spacing=np.array(voxel_spacing(native_affine), dtype=np.float32),
        trunc_mm=np.float32(trunc_mm),
    )

    if verbose:
        crop_shape = tuple(hi - lo for lo, hi in bbox)
        print(f"  {cid[:44]:<44}  native={native_shape}  crop={crop_shape}"
              f"  band={band_crop.sum():,}vox", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prior-dir", type=str, required=True,
                        help="the reference prior: a first pass's native SDF predictions (make_priors.py)")
    add_dataset_argument(parser)
    parser.add_argument("--cache-dir", type=str, default=None, help="default: the dataset's cache")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--padding", type=int, default=10, help="voxel margin around the band bbox")
    parser.add_argument("--band-mm", type=float, default=BAND_MM)
    parser.add_argument("--trunc-mm", type=float, default=TRUNC_MM)
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or get_dataset(args.dataset).cache

    priors = glob.glob(os.path.join(args.prior_dir, "*.npz"))
    cids = sorted(os.path.splitext(os.path.basename(f))[0] for f in priors)
    print(f"{len(cids)} cases from prior {args.prior_dir} -> {args.out_dir}\n"
          f"padding={args.padding}vox  band_mm={args.band_mm}\n", flush=True)

    for i, cid in enumerate(cids, start=1):
        print(f"[{i}/{len(cids)}]", end="", flush=True)
        build_one(cid, args.cache_dir, args.prior_dir, args.out_dir, args.padding, args.band_mm, args.trunc_mm,
                  verbose=True)

    print(f"\n{len(cids)} crops created -> {args.out_dir}")


if __name__ == "__main__":
    main()
