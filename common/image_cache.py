"""Build the image cache every other script reads: x1 (native), x4 (the network's grid) and label.

Every volume is reoriented to RAS and the image is z-scored over the whole volume.
"""
import argparse
import glob
import os
import time
import warnings

import numpy as np
from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad, LoadImaged,
                              NormalizeIntensityd, Orientationd)

from common.geometry import FACTOR, downsample
from common.paths import NATIVE_CACHE

warnings.filterwarnings("ignore", category=UserWarning)   # MONAI's deprecation notices; the defaults used are checked

FACTORS = (1, FACTOR)   # the resolutions cached
KEYS = ["image", "label"]
NIFTI_SUFFIXES = ("_0000.nii.gz", ".nii.gz")   # "_0000" is nnU-Net's channel suffix

PREPROCESS = Compose([
    LoadImaged(keys=KEYS),
    EnsureChannelFirstd(keys=KEYS),
    # Only flips and axis swaps, so nothing is interpolated; labels=None reads the axis names from the header.
    Orientationd(keys=KEYS, axcodes="RAS", labels=None),
    # The flips leave negative strides, which make every later pass ~100x slower.
    Lambdad(keys=KEYS, func=lambda t: t.contiguous()),
    NormalizeIntensityd(keys=["image"]),
])


def case_id(path):
    name = os.path.basename(path)
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def save(path, key, tensor):
    """Uncompressed on purpose: compressing z-scored speckle cost 3.3 s per case for 7% of the space."""
    without_channel = tensor[0]
    np.savez(path, **{key: np.asarray(without_channel.cpu())}, affine=np.asarray(tensor.affine.cpu()))


def labelled_pairs(images_dir, labels_dir):
    """Every image with the label of the same case; images without a label are skipped."""
    labels = {case_id(p): p for p in glob.glob(os.path.join(labels_dir, "*.nii.gz"))}
    images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    return [{"image": p, "label": labels[case_id(p)]} for p in images if case_id(p) in labels]


def shape_text(tensor):
    return "x".join(map(str, tensor.shape[1:]))


def build(images_dir, labels_dir, out_dir, factors=FACTORS, limit=0):
    pairs = labelled_pairs(images_dir, labels_dir)
    if limit:
        pairs = pairs[:limit]
    print(f"{len(pairs)} labelled cases -> {out_dir}\n", flush=True)

    for name in [f"x{f}" for f in factors] + ["label"]:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    for i, pair in enumerate(pairs, start=1):
        start = time.time()
        cid = case_id(pair["image"])
        item = PREPROCESS(pair)

        binary_label = (item["label"] > 0).byte()   # native resolution only, never resampled
        save(os.path.join(out_dir, "label", f"{cid}.npz"), "label", binary_label)

        sizes = []
        for factor in sorted(factors):
            down = downsample(item, factor) if factor > 1 else item
            save(os.path.join(out_dir, f"x{factor}", f"{cid}.npz"), "img", down["image"])
            sizes.append(f"x{factor}={shape_text(down['image'])}")

        print(f"[{i}/{len(pairs)}] {cid[:44]:<44} {time.time() - start:5.1f}s  "
              f"{'  '.join(sizes)}", flush=True)

    print(f"\n{len(pairs)} cases cached")


def load(cache_dir, cid, factor):
    """A cached image at one resolution: (array, affine)."""
    data = np.load(os.path.join(cache_dir, f"x{factor}", f"{cid}.npz"))
    return data["img"], data["affine"]


def load_label(cache_dir, cid):
    """A cached label, (D, H, W) at native resolution: (array, affine)."""
    data = np.load(os.path.join(cache_dir, "label", f"{cid}.npz"))
    return data["label"], data["affine"]


def case_ids(cache_dir):
    names = glob.glob(os.path.join(cache_dir, "x1", "*.npz"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the image and label cache: <out-dir>/x1, x4 and label, one .npz per case.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", default=NATIVE_CACHE)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    args = parser.parse_args()
    build(args.images_dir, args.labels_dir, args.out_dir, limit=args.limit)
