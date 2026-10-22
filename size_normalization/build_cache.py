"""Build the size-normalised training copy of the x4 cache (caches/cache_norm4), every head at the size V0.

Each case gets spacing = 4 x native spacing x (V / V0)^(1/3) from its label's volume V, so a small
head gets finer voxels rather than fewer. V0 is the labels' median (--v0-ml overrides it; the cache
was built with 130 mL) and is stored in norm.json, where inference reads it. Evaluation and
deployment never use this cache: infer.py normalises each scan with pass 1's estimate.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
from monai.data import MetaTensor

from common.geometry import CUBE, FACTOR, MM3_PER_ML, downsample, isotropic_spacing, voxel_spacing
from common.image_cache import case_ids, load, load_label
from common.sdf_targets import TRUNC_MM, signed_distance_mm
from datasets import add_dataset_argument, get as get_dataset

CUBE_FIT_MARGIN = 1.005   # for the shape's rounding when a case is coarsened to fit the cube
LINKED_FROM_NATIVE = ["x1", "label", os.path.join("sdf", "x1")]   # predictions go back to native via the affine

LAYOUT = """\
output, laid out like caches/cache/ so the other scripts read it unchanged:
  x4/, sdf/x4/, sdf/x4mm/   the normalised image and targets, each with its own affine
  x1/, label/, sdf/x1/      links to caches/cache/
  norm.json                 V0 and each case's volume, scale and spacing

A case too large for the 96^3 training cube is coarsened just enough to fit (once in this dataset,
by 1%). Inference does not cap: it pads instead (the cap cost 0.028 Dice on a scan 17% too large)."""


def link(src: str, dst: str):
    """A junction on Windows, which needs no administrator rights (a directory symlink does)."""
    src, dst = os.path.abspath(src), os.path.abspath(dst)   # mklink needs absolute Windows paths
    if os.path.exists(dst):
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True, stdout=subprocess.DEVNULL)
    else:
        os.symlink(src, dst, target_is_directory=True)


def reference_v0(cache_dir: str) -> float:
    """The canonical head volume (mL) a size-normalised cache was built with."""
    with open(os.path.join(cache_dir, "norm.json")) as f:
        return float(json.load(f)["v0_ml"])


def mask_volume_ml(label: np.ndarray, affine: np.ndarray):
    return label.astype(bool).sum() * np.prod(voxel_spacing(affine)) / MM3_PER_ML


def label_volume_ml(src: str, cid: str) -> float:
    return float(mask_volume_ml(*load_label(src, cid)))


def resample(arr: np.ndarray, affine: np.ndarray, factor: float):
    """A native array at `factor` (any real) times its spacing: (array, affine)."""
    with_channel = MetaTensor(torch.from_numpy(arr.astype(np.float32))[None], affine=torch.from_numpy(affine))
    resampled = downsample({"x": with_channel}, factor, key="x")["x"]
    without_channel = resampled[0]
    return np.asarray(without_channel.cpu(), dtype=np.float32), np.asarray(resampled.affine.cpu())


def normalise_case(src, cid, v0):
    """The case's image, SDF and SDF in mm on its normalised grid, that grid's affine, and its record."""
    img, affine = load(src, cid, 1)
    label, _ = load_label(src, cid)
    label = label.astype(bool)

    true_ml = mask_volume_ml(label, affine)
    scale = (float(true_ml) / v0) ** (1 / 3)
    factor = FACTOR * scale
    img4, affine4 = resample(img, affine, factor)

    largest_axis = max(img4.shape)
    if largest_axis > CUBE:
        factor *= largest_axis / CUBE * CUBE_FIT_MARGIN
        scale = factor / FACTOR
        img4, affine4 = resample(img, affine, factor)

    sdf4_mm, affine_sdf = resample(signed_distance_mm(label, voxel_spacing(affine)), affine, factor)
    assert np.allclose(affine4, affine_sdf) and img4.shape == sdf4_mm.shape, cid
    sdf4 = np.clip(sdf4_mm, -TRUNC_MM, TRUNC_MM) / TRUNC_MM

    record = {"volume_ml": float(true_ml), "true_ml": float(true_ml), "scale": scale,
              "spacing_mm": isotropic_spacing(affine4), "shape": list(img4.shape)}
    return img4, sdf4, sdf4_mm, affine4, record


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=LAYOUT,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_argument(parser)
    parser.add_argument("--src", default=None, help="default: the dataset's cache")
    parser.add_argument("--dst", default=None, help="default: the dataset's size-normalised cache")
    parser.add_argument("--v0-ml", type=float, default=None,
                        help="override V0; default: the median head volume of the labels")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.src = args.src or dataset.cache
    args.dst = args.dst or dataset.norm_cache

    ids = case_ids(args.src)[:args.limit or None]
    v0 = args.v0_ml if args.v0_ml is not None else float(np.median([label_volume_ml(args.src, cid) for cid in ids]))
    print(f"V0 = {v0:.1f} mL", flush=True)

    grid = f"x{FACTOR}"
    image_dir = os.path.join(args.dst, grid)
    sdf_dir = os.path.join(args.dst, "sdf", grid)
    sdf_mm_dir = os.path.join(args.dst, "sdf", grid + "mm")
    for d in (image_dir, sdf_dir, sdf_mm_dir):
        os.makedirs(d, exist_ok=True)
    for name in LINKED_FROM_NATIVE:
        link(os.path.join(args.src, name), os.path.join(args.dst, name))

    cases = {}
    for i, cid in enumerate(ids, start=1):
        start = time.time()
        img4, sdf4, sdf4_mm, affine4, record = normalise_case(args.src, cid, v0)
        np.savez(os.path.join(image_dir, f"{cid}.npz"), img=img4, affine=affine4)
        np.savez(os.path.join(sdf_dir, f"{cid}.npz"), sdf=sdf4, affine=affine4)
        np.savez(os.path.join(sdf_mm_dir, f"{cid}.npz"), sdf=sdf4_mm, affine=affine4)
        cases[cid] = record
        print(f"[{i}/{len(ids)}] {cid[:44]:<44} {record['volume_ml']:6.1f} mL  spacing "
              f"{record['spacing_mm']:.2f}mm  {'x'.join(map(str, img4.shape))}  "
              f"{time.time() - start:4.1f}s", flush=True)

    with open(os.path.join(args.dst, "norm.json"), "w") as f:
        json.dump({"v0_ml": v0, "oracle": True, "factor": FACTOR, "cube": CUBE, "cases": cases}, f, indent=1)
    shapes = np.array([record["shape"] for record in cases.values()])
    print(f"\nlargest shape per axis {shapes.max(0)} (cube {CUBE})")


if __name__ == "__main__":
    main()
