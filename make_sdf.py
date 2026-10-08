# Builds the training targets: the signed distance field (SDF) of each label.
#
#     python make_sdf.py                 # caches/cache/sdf/x1 and x4: normalised, truncated
#     python make_sdf.py --untruncated   # caches/cache/sdf/x4mm: in millimetres, for augmentation
#
# The SDF gives, for every voxel, its distance in mm to the head's surface: positive
# outside the head, negative inside, zero on the surface. The network regresses it and the
# segmentation is where it is negative.
#
# The distance is computed once, exactly, on the native label grid with each case's own
# spacing, and then resampled to each resolution (the same resampling as the image cache,
# so image and target land on the same grid).

import argparse
import os
import time
import warnings

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore", category=UserWarning)

from monai.data import MetaTensor  # noqa: E402

from cache import FACTORS, case_ids, load_label  # noqa: E402
from geometry import downsample  # noqa: E402


# Exact Euclidean SDF in mm. `spacing` is the voxel size per axis in mm.
def signed_distance_mm(label, spacing):
    label = label.astype(bool)
    # For each outside voxel, the distance to the nearest inside voxel ...
    outside = distance_transform_edt(~label, sampling=spacing)
    # ... and for each inside voxel, the distance to the nearest outside voxel.
    inside = distance_transform_edt(label, sampling=spacing)
    # Exactly one of the two is non-zero per voxel: positive outside, negative inside.
    return outside - inside


# Clip to +-truncation_mm and scale to [-1, 1]. Far from the surface the exact distance
# carries no information about where the surface is, so the target saturates there.
def normalise(sdf_mm, truncation_mm):
    return np.clip(sdf_mm, -truncation_mm, truncation_mm) / truncation_mm


def save(path, sdf, affine):
    np.savez(path, sdf=np.asarray(sdf[0].cpu()), affine=np.asarray(affine.cpu()))


# The normalised SDF at every cached resolution: caches/cache/sdf/x<f>/<case>.npz.
def build(cache_dir, out_dir, factors=FACTORS, truncation_mm=10.0, limit=0):
    ids = case_ids(cache_dir)
    if limit:
        ids = ids[:limit]
    print(f"{len(ids)} cases -> {out_dir}  (truncation={truncation_mm}mm)\n", flush=True)

    for factor in factors:
        os.makedirs(os.path.join(out_dir, f"x{factor}"), exist_ok=True)

    for i, cid in enumerate(ids, start=1):
        t0 = time.time()
        label, affine = load_label(cache_dir, cid)
        spacing = np.abs(np.diag(affine[:3, :3]))

        # Labels are stored with a channel axis; drop it if present.
        sdf_mm = signed_distance_mm(label[0] if label.ndim == 4 else label, spacing)
        sdf = normalise(sdf_mm, truncation_mm)

        # Wrap it with its affine so it can be resampled like an image.
        item = {"sdf": MetaTensor(torch.from_numpy(sdf)[None].float(),
                                  affine=torch.from_numpy(affine))}

        sizes = []
        for factor in sorted(factors):
            down = downsample(item, factor, key="sdf") if factor > 1 else item
            save(os.path.join(out_dir, f"x{factor}", f"{cid}.npz"),
                 down["sdf"], down["sdf"].affine)
            sizes.append(f"x{factor}={'x'.join(map(str, down['sdf'].shape[1:]))}")

        print(f"[{i}/{len(ids)}] {cid[:44]:<44} {time.time() - t0:5.1f}s  "
              f"{'  '.join(sizes)}", flush=True)

    print(f"\n{len(ids)} cases done")


# Read a normalised SDF: returns (array, affine).
def load(cache_dir, cid, factor):
    d = np.load(os.path.join(cache_dir, "sdf", f"x{factor}", f"{cid}.npz"))
    return d["sdf"], d["affine"]


# The same field in mm and untruncated, at one resolution: caches/cache/sdf/x<f>mm/<case>.npz.
#
# Augmentation needs it. Zooming a volume by k multiplies every distance by k, and a
# truncated field cannot be rescaled: a voxel 14 mm out is stored as 10, and after a zoom
# of 0.7 it would read 7 mm where the truth is 9.8. So the augmentation rescales the field
# in mm and truncates only afterwards. Clipping this field to +-10 and dividing by 10
# reproduces caches/cache/sdf/x<f> exactly.
def build_mm(cache_dir, factor=4, limit=0):
    ids = case_ids(cache_dir)[:limit or None]
    out = os.path.join(cache_dir, "sdf", f"x{factor}mm")
    os.makedirs(out, exist_ok=True)
    for i, cid in enumerate(ids, start=1):
        label, affine = load_label(cache_dir, cid)
        spacing = np.abs(np.diag(affine[:3, :3]))
        sdf_mm = signed_distance_mm(label[0] if label.ndim == 4 else label, spacing)
        item = {"sdf": MetaTensor(torch.from_numpy(sdf_mm)[None].float(),
                                  affine=torch.from_numpy(affine))}
        down = downsample(item, factor, key="sdf")
        save(os.path.join(out, f"{cid}.npz"), down["sdf"], down["sdf"].affine)
        print(f"[{i}/{len(ids)}] {cid[:44]}", flush=True)


# Read an untruncated mm SDF: returns (array, affine).
def load_mm(cache_dir, cid, factor=4):
    d = np.load(os.path.join(cache_dir, "sdf", f"x{factor}mm", f"{cid}.npz"))
    return d["sdf"], d["affine"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Build the SDF targets from the cached labels.")
    p.add_argument("--cache-dir", default=os.path.join(here, "caches", "cache"))
    p.add_argument("--out-dir", default=os.path.join(here, "caches", "cache", "sdf"))
    p.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    p.add_argument("--truncation-mm", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    p.add_argument("--untruncated", action="store_true",
                   help="write the untruncated mm field at x4 to cache/sdf/x4mm instead")
    a = p.parse_args()
    if a.untruncated:
        build_mm(a.cache_dir, 4, a.limit)
    else:
        build(a.cache_dir, a.out_dir, tuple(a.factors), a.truncation_mm, a.limit)
