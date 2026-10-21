# Builds the image cache every other script reads.
#
#     python common/image_cache.py --images-dir <NIfTI images> --labels-dir <NIfTI labels>
#
# For each case it writes
#     caches/cache/x1/<case>.npz      the image at native resolution (0.4 mm), with its affine
#     caches/cache/x4/<case>.npz      the image at 4x the native spacing (1.6 mm)
#     caches/cache/label/<case>.npz   the label at native resolution, never resampled
#
# Every volume is reoriented to RAS and the image is z-scored (zero mean, unit std).

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np

# MONAI warns about deprecated defaults on every load; the defaults used here are checked.
warnings.filterwarnings("ignore", category=UserWarning)

from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad,  # noqa: E402
                              LoadImaged, NormalizeIntensityd, Orientationd)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import NATIVE_CACHE  # noqa: E402
from common.geometry import downsample  # noqa: E402

# The resolutions cached: native (x1) and the network's grid (x4).
FACTORS = (1, 4)

# Loading pipeline shared by image and label.
PREPROCESS = Compose([
    # Read the NIfTI files.
    LoadImaged(keys=["image", "label"]),
    # Add the channel axis MONAI expects: (1, X, Y, Z).
    EnsureChannelFirstd(keys=["image", "label"]),
    # Reorient both to RAS. It is only flips and axis swaps, so nothing is interpolated,
    # and it must be applied to the label too or the two would no longer line up.
    # labels=None reads the axis names from the image header (MONAI's new default);
    # checked identical on an LPS and a RAS case.
    Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
    # The flips leave a view with negative strides, which makes every later pass ~100x
    # slower; make the arrays contiguous again.
    Lambdad(keys=["image", "label"], func=lambda t: t.contiguous()),
    # Z-score the image over the whole volume.
    NormalizeIntensityd(keys=["image"]),
])


# The case id is the file name without its extension (and without nnU-Net's "_0000"
# channel suffix, when present).
def case_id(path):
    name = os.path.basename(path)
    for suffix in ("_0000.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


# Save one volume and its affine. Uncompressed on purpose: compressing z-scored speckle
# took 3.3 s per case to save 7% of the space.
def save(path, key, tensor):
    np.savez(path, **{key: np.asarray(tensor[0].cpu())},
             affine=np.asarray(tensor.affine.cpu()))


def build(images_dir, labels_dir, out_dir, factors=FACTORS, limit=0):
    # Pair every image with the label of the same case; images without a label are skipped.
    labels = {case_id(p): p for p in glob.glob(os.path.join(labels_dir, "*.nii.gz"))}
    pairs = [{"image": p, "label": labels[case_id(p)]}
             for p in sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
             if case_id(p) in labels]
    if limit:
        pairs = pairs[:limit]
    print(f"{len(pairs)} labelled cases -> {out_dir}\n", flush=True)

    for name in [f"x{f}" for f in factors] + ["label"]:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    for i, pair in enumerate(pairs, start=1):
        t0 = time.time()
        cid = case_id(pair["image"])
        item = PREPROCESS(pair)

        # The label is stored as a binary mask, at native resolution only.
        save(os.path.join(out_dir, "label", f"{cid}.npz"), "label",
             (item["label"] > 0).byte())

        # The image at every cached resolution.
        sizes = []
        for factor in sorted(factors):
            down = downsample(item, factor) if factor > 1 else item
            save(os.path.join(out_dir, f"x{factor}", f"{cid}.npz"), "img", down["image"])
            sizes.append(f"x{factor}={'x'.join(map(str, down['image'].shape[1:]))}")

        print(f"[{i}/{len(pairs)}] {cid[:44]:<44} {time.time() - t0:5.1f}s  "
              f"{'  '.join(sizes)}", flush=True)

    print(f"\n{len(pairs)} cases cached")


# Read a cached image at one resolution: returns (array, affine).
def load(cache_dir, cid, factor):
    d = np.load(os.path.join(cache_dir, f"x{factor}", f"{cid}.npz"))
    return d["img"], d["affine"]


# Read a cached label: returns (array, affine).
def load_label(cache_dir, cid):
    d = np.load(os.path.join(cache_dir, "label", f"{cid}.npz"))
    return d["label"], d["affine"]


# Every case id in a cache, sorted.
def case_ids(cache_dir):
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(cache_dir, "x1", "*.npz")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the image and label cache.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out-dir", default=NATIVE_CACHE)
    p.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    p.add_argument("--limit", type=int, default=0, help="only the first N cases (testing)")
    a = p.parse_args()
    build(a.images_dir, a.labels_dir, a.out_dir, tuple(a.factors), a.limit)
