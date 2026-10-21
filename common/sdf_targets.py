"""Build the training targets: the signed distance field (SDF) of each label, in mm to the head's surface.

Positive outside the head, negative inside. It is computed once, exactly, on the native label grid
and then resampled like the image cache, so image and target land on the same grid.
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch
from monai.data import MetaTensor
from scipy.ndimage import distance_transform_edt

from common.geometry import FACTOR, downsample, voxel_spacing
from common.image_cache import FACTORS, case_ids, load_label
from common.paths import NATIVE_CACHE

warnings.filterwarnings("ignore", category=UserWarning)

TRUNC_MM = 10.0   # +-10 mm maps to +-1; farther out the distance says nothing about where the surface is


def signed_distance_mm(label, spacing):
    """Exact Euclidean SDF in mm; `spacing` is the voxel size per axis."""
    label = label.astype(bool)
    distance_to_inside = distance_transform_edt(~label, sampling=spacing)
    distance_to_outside = distance_transform_edt(label, sampling=spacing)
    return distance_to_inside - distance_to_outside   # one of the two is zero at every voxel


def normalise(sdf_mm, truncation_mm):
    return np.clip(sdf_mm, -truncation_mm, truncation_mm) / truncation_mm


def save(path, sdf, affine):
    without_channel = sdf[0]
    np.savez(path, sdf=np.asarray(without_channel.cpu()), affine=np.asarray(affine.cpu()))


def label_sdf_mm(cache_dir, cid):
    """The case's SDF in mm on the native grid, and that grid's affine."""
    label, affine = load_label(cache_dir, cid)
    return signed_distance_mm(label, voxel_spacing(affine)), affine


def as_resamplable(sdf, affine):
    """Wrap with its affine and a channel axis so MONAI resamples it like an image."""
    return {"sdf": MetaTensor(torch.from_numpy(sdf)[None].float(), affine=torch.from_numpy(affine))}


def build(cache_dir, out_dir, factors=FACTORS, truncation_mm=TRUNC_MM, limit=0):
    """The normalised SDF at every cached resolution: <out_dir>/x<f>/<case>.npz."""
    ids = case_ids(cache_dir)
    if limit:
        ids = ids[:limit]
    print(f"{len(ids)} cases -> {out_dir}  (truncation={truncation_mm}mm)\n", flush=True)

    for factor in factors:
        os.makedirs(os.path.join(out_dir, f"x{factor}"), exist_ok=True)

    for i, cid in enumerate(ids, start=1):
        start = time.time()
        sdf_mm, affine = label_sdf_mm(cache_dir, cid)
        item = as_resamplable(normalise(sdf_mm, truncation_mm), affine)

        sizes = []
        for factor in sorted(factors):
            down = downsample(item, factor, key="sdf") if factor > 1 else item
            save(os.path.join(out_dir, f"x{factor}", f"{cid}.npz"), down["sdf"], down["sdf"].affine)
            sizes.append(f"x{factor}={'x'.join(map(str, down['sdf'].shape[1:]))}")

        print(f"[{i}/{len(ids)}] {cid[:44]:<44} {time.time() - start:5.1f}s  "
              f"{'  '.join(sizes)}", flush=True)

    print(f"\n{len(ids)} cases done")


def load(cache_dir, cid, factor):
    """A normalised SDF: (array, affine)."""
    data = np.load(os.path.join(cache_dir, "sdf", f"x{factor}", f"{cid}.npz"))
    return data["sdf"], data["affine"]


def build_mm(cache_dir, factor=FACTOR, limit=0):
    """The same field untruncated, in mm: <cache_dir>/sdf/x<f>mm/<case>.npz, for the augmentation.

    Zooming by k scales every distance by k, which a truncated field cannot follow (14 mm stored as 10
    reads 7 after a 0.7 zoom, not 9.8), so the augmentation zooms this field and truncates afterwards.
    """
    ids = case_ids(cache_dir)[:limit or None]
    out = os.path.join(cache_dir, "sdf", f"x{factor}mm")
    os.makedirs(out, exist_ok=True)
    for i, cid in enumerate(ids, start=1):
        sdf_mm, affine = label_sdf_mm(cache_dir, cid)
        down = downsample(as_resamplable(sdf_mm, affine), factor, key="sdf")
        save(os.path.join(out, f"{cid}.npz"), down["sdf"], down["sdf"].affine)
        print(f"[{i}/{len(ids)}] {cid[:44]}", flush=True)


def load_mm(cache_dir, cid, factor=FACTOR):
    """An untruncated SDF in mm: (array, affine)."""
    data = np.load(os.path.join(cache_dir, "sdf", f"x{factor}mm", f"{cid}.npz"))
    return data["sdf"], data["affine"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the SDF targets from the cached labels: sdf/x1 and sdf/x4, normalised and truncated.")
    parser.add_argument("--cache-dir", default=NATIVE_CACHE)
    parser.add_argument("--out-dir", default=os.path.join(NATIVE_CACHE, "sdf"))
    parser.add_argument("--truncation-mm", type=float, default=TRUNC_MM)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    parser.add_argument("--untruncated", action="store_true",
                        help="write the untruncated mm field at x4 to <cache-dir>/sdf/x4mm instead")
    args = parser.parse_args()
    if args.untruncated:
        build_mm(args.cache_dir, FACTOR, args.limit)
    else:
        build(args.cache_dir, args.out_dir, truncation_mm=args.truncation_mm, limit=args.limit)
