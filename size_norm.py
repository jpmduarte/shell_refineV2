# Builds a size-normalised copy of the x4 cache, in which every head has the same
# apparent size.
#
#     python size_norm.py                                    # training copy -> caches/cache_norm4/
#     python size_norm.py --volumes est.json --dst caches/cache_norm4_est --reference caches/cache_norm4
#
# Head volumes in the dataset range from 35 to 291 mL, so a network on a fixed grid has to
# handle very different sizes, and the smallest heads are rare. Here each case gets its
# own spacing, proportional to its head's linear size:
#
#     spacing = 4 x native spacing x (V / V0)^(1/3)
#
# so every head spans about the same number of voxels; a small head gets finer voxels
# rather than fewer of them.
#
# V0, the canonical size, is the median head volume of the training data: computed here
# from the labels when the training copy is built, and stored in its norm.json. Every
# later use (the estimated-volume copies, deployment) reads it from there (--reference),
# so it comes from the data and training and inference cannot disagree. --v0-ml
# overrides it; the existing caches were built with 130 mL (the median, 128.3, rounded).
#
# Train on the copy built from the true volumes (from the labels). For evaluation and
# deployment, the volume comes from a first pass of the plain x4 network (--volumes).
#
# Output layout, same as caches/cache/ so the other scripts read it unchanged:
#     x4/, sdf/x4/, sdf/x4mm/   the normalised image and targets, each with its own affine
#     x1/, label/, sdf/x1/      links to caches/cache/: predictions go back to the native grid
#                               through the affine, and evaluation is untouched
#     norm.json                 each case's volume, scale and spacing
#
# The 96^3 cube: training batches need every scan to fit 96 voxels per axis, and
# enlarging a small head enlarges its whole scan. A case that would exceed 96 is
# coarsened just enough to fit (its head ends a little below the canonical size) rather
# than cropped. It never ends coarser than plain x4, which fits every scan. In this
# dataset it applied once, by 1%. Deployment (scripts/bench_pipeline.py) does not cap: it
# pads a larger grid to a multiple of 8 instead, which keeps the canonical size (on a scan
# 17% too large, the cap cost 0.028 Dice).

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from monai.data import MetaTensor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache import case_ids, load, load_label  # noqa: E402
from geometry import downsample  # noqa: E402
from make_sdf import signed_distance_mm  # noqa: E402

TRUNC_MM = 10.0    # SDF truncation, as in make_sdf.py
FACTOR = 4         # the grid normalised: x4
CUBE = 96          # the network's input size per axis


# Link a folder of caches/cache/ into the new cache instead of copying it. On Windows a junction,
# which needs no administrator rights (a directory symlink does).
def link(src: str, dst: str):
    if os.path.exists(dst):
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        os.symlink(src, dst, target_is_directory=True)


# The canonical head volume (mL) a size-normalised cache was built with.
def reference_v0(cache_dir: str) -> float:
    with open(os.path.join(cache_dir, "norm.json")) as f:
        return float(json.load(f)["v0_ml"])


# Head volume in mL of one case, from its label.
def label_volume_ml(src: str, cid: str) -> float:
    label, affine = load_label(src, cid)
    spacing = np.abs(np.diag(affine[:3, :3]))
    return float(label.astype(bool).sum() * np.prod(spacing) / 1000)


# Resample a native array to `factor` times its spacing (factor need not be an integer).
# Returns (array, affine).
def resample(arr: np.ndarray, affine: np.ndarray, factor: float):
    item = {"x": MetaTensor(torch.from_numpy(arr.astype(np.float32))[None],
                            affine=torch.from_numpy(affine))}
    out = downsample(item, factor, key="x")["x"]
    return np.asarray(out[0].cpu(), dtype=np.float32), np.asarray(out.affine.cpu())


def main():
    p = argparse.ArgumentParser(description="Build a size-normalised copy of the x4 cache.")
    p.add_argument("--src", default=os.path.join(HERE, "caches", "cache"))
    p.add_argument("--dst", default=os.path.join(HERE, "caches", "cache_norm4"))
    p.add_argument("--volumes", default=None,
                   help="JSON {case: mL} of estimated volumes; default: the label's")
    p.add_argument("--reference", default=None,
                   help="the training cache whose V0 to use (required with --volumes)")
    p.add_argument("--v0-ml", type=float, default=None,
                   help="override V0; default: the median head volume of the labels")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    a = p.parse_args()

    estimates = json.load(open(a.volumes)) if a.volumes else None
    ids = case_ids(a.src)[:a.limit or None]

    # The canonical volume V0: given, or the training cache's, or (building a training
    # cache) the median of the label volumes. An estimated-volume cache must use its
    # training cache's V0, never one of its own.
    if a.v0_ml is not None:
        v0 = a.v0_ml
    elif a.reference:
        v0 = reference_v0(a.reference)
    elif estimates is None:
        v0 = float(np.median([label_volume_ml(a.src, cid) for cid in ids]))
    else:
        p.error("--volumes needs --reference <training cache> (or --v0-ml)")
    print(f"V0 = {v0:.1f} mL", flush=True)

    xf = f"x{FACTOR}"
    for d in (xf, os.path.join("sdf", xf), os.path.join("sdf", xf + "mm")):
        os.makedirs(os.path.join(a.dst, d), exist_ok=True)
    link(os.path.join(a.src, "x1"), os.path.join(a.dst, "x1"))
    link(os.path.join(a.src, "label"), os.path.join(a.dst, "label"))
    link(os.path.join(a.src, "sdf", "x1"), os.path.join(a.dst, "sdf", "x1"))

    meta = {}
    for i, cid in enumerate(ids, start=1):
        t0 = time.time()
        img, affine = load(a.src, cid, 1)
        label, _ = load_label(a.src, cid)
        label = (label[0] if label.ndim == 4 else label).astype(bool)
        spacing = np.abs(np.diag(affine[:3, :3]))

        # Head volume in mL: from the label, or the estimate when one is given.
        true_ml = label.sum() * np.prod(spacing) / 1000
        # `is not None`: an empty estimates file must fail on the missing case, not fall
        # back silently to the true volume.
        vol_ml = float(estimates[cid]) if estimates is not None else true_ml

        # The linear scale between this head and the canonical one, and the resulting
        # resampling factor relative to the native spacing.
        scale = (vol_ml / v0) ** (1 / 3)
        factor = FACTOR * scale

        img4, affine4 = resample(img, affine, factor)

        # Too large for the cube: coarsen just enough to fit (+0.5% for shape rounding).
        if max(img4.shape) > CUBE:
            factor *= max(img4.shape) / CUBE * 1.005
            scale = factor / FACTOR
            img4, affine4 = resample(img, affine, factor)

        # The targets on exactly the same grid: the SDF in mm, then truncated and scaled.
        sdf_mm = signed_distance_mm(label, spacing)
        sdf4_mm, affine_sdf = resample(sdf_mm, affine, factor)
        assert np.allclose(affine4, affine_sdf) and img4.shape == sdf4_mm.shape, cid
        sdf4 = np.clip(sdf4_mm, -TRUNC_MM, TRUNC_MM) / TRUNC_MM

        np.savez(os.path.join(a.dst, xf, f"{cid}.npz"), img=img4, affine=affine4)
        np.savez(os.path.join(a.dst, "sdf", xf, f"{cid}.npz"), sdf=sdf4, affine=affine4)
        np.savez(os.path.join(a.dst, "sdf", xf + "mm", f"{cid}.npz"), sdf=sdf4_mm, affine=affine4)

        meta[cid] = {"volume_ml": vol_ml, "true_ml": float(true_ml), "scale": scale,
                     "spacing_mm": float(np.abs(affine4[0, 0])), "shape": list(img4.shape)}
        print(f"[{i}/{len(ids)}] {cid[:44]:<44} {vol_ml:6.1f} mL  spacing "
              f"{np.abs(affine4[0, 0]):.2f}mm  {'x'.join(map(str, img4.shape))}  "
              f"{time.time() - t0:4.1f}s", flush=True)

    with open(os.path.join(a.dst, "norm.json"), "w") as f:
        json.dump({"v0_ml": v0, "oracle": estimates is None, "cases": meta}, f, indent=1)
    shapes = np.array([m["shape"] for m in meta.values()])
    print(f"\nlargest shape per axis {shapes.max(0)} (cube {CUBE})")


if __name__ == "__main__":
    main()
