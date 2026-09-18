"""Signed distance field targets for phase 1, at the same factors as the image cache.

    python make_sdf.py

Reads cache/label/<case>.npz (built by cache.py) and writes
    cache/sdf/x1|x2|x4|x8/<case>.npz   sdf and its affine

Distance is computed once on the native label grid with scipy's exact Euclidean
transform, using the case's own mm spacing (labels are isotropic per case but spacing
varies case to case, same as the images). Sign convention: positive outside the head,
negative inside, matching geometry.upsample's "an SDF needs +1 outside" cval handling.
Truncated to +/-truncation_mm then normalised to [-1, 1], so phase 1 regresses a
bounded target instead of an unbounded distance.

Each factor's SDF is a separate resample of the native field (not a re-truncation of a
coarser one), using the same Spacingd/align_corners=False convention as the image cache,
so ascending an x_k SDF and ascending an x_k image land on the same grid.
"""

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


def signed_distance_mm(label, spacing):
    """label: binary array, spacing: mm per voxel per axis. +outside, -inside."""
    label = label.astype(bool)
    outside = distance_transform_edt(~label, sampling=spacing)
    inside = distance_transform_edt(label, sampling=spacing)
    return outside - inside


def normalise(sdf_mm, truncation_mm):
    return np.clip(sdf_mm, -truncation_mm, truncation_mm) / truncation_mm


def save(path, sdf, affine):
    np.savez(path, sdf=np.asarray(sdf[0].cpu()), affine=np.asarray(affine.cpu()))


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

        sdf_mm = signed_distance_mm(label[0] if label.ndim == 4 else label, spacing)
        sdf = normalise(sdf_mm, truncation_mm)

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


def load(cache_dir, cid, factor):
    d = np.load(os.path.join(cache_dir, "sdf", f"x{factor}", f"{cid}.npz"))
    return d["sdf"], d["affine"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=os.path.join(here, "cache"))
    p.add_argument("--out-dir", default=os.path.join(here, "cache", "sdf"))
    p.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    p.add_argument("--truncation-mm", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    build(a.cache_dir, a.out_dir, tuple(a.factors), a.truncation_mm, a.limit)
