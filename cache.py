"""Reoriented, normalised volumes at integer factors of the native spacing.

    python cache.py --images-dir ... --labels-dir ...

    cache/x1|x2|x4|x8/<case>.npz   image and its affine
    cache/label/<case>.npz         label and its affine, never resampled
"""

import argparse
import glob
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad,  # noqa: E402
                              LoadImaged, NormalizeIntensityd, Orientationd)

from geometry import downsample  # noqa: E402

FACTORS = (1, 2, 4, 8)

# Orientation goes on both keys: it is a flip and transpose, so it is lossless, and
# skipping it on the label would break alignment. Only the spacing change is withheld
# from the label. Lambdad is not cosmetic: the flips leave a negative-stride view, and
# every later pass over it runs about 100x slower.
PREPROCESS = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # labels=None opts into MONAI's new default, reading axis labels off the meta-tensor
    # space. Verified identical on both an LPS and a RAS case here.
    Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
    Lambdad(keys=["image", "label"], func=lambda t: t.contiguous()),
    NormalizeIntensityd(keys=["image"]),
])


def case_id(path):
    name = os.path.basename(path)
    for suffix in ("_0000.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def save(path, key, tensor):
    # Uncompressed everywhere: zlib on z-scored speckle costs 3.3s to save 7%.
    np.savez(path, **{key: np.asarray(tensor[0].cpu())},
             affine=np.asarray(tensor.affine.cpu()))


def build(images_dir, labels_dir, out_dir, factors=FACTORS, limit=0):
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

        save(os.path.join(out_dir, "label", f"{cid}.npz"), "label",
             (item["label"] > 0).byte())

        sizes = []
        for factor in sorted(factors):
            down = downsample(item, factor) if factor > 1 else item
            save(os.path.join(out_dir, f"x{factor}", f"{cid}.npz"), "img", down["image"])
            sizes.append(f"x{factor}={'x'.join(map(str, down['image'].shape[1:]))}")

        print(f"[{i}/{len(pairs)}] {cid[:44]:<44} {time.time() - t0:5.1f}s  "
              f"{'  '.join(sizes)}", flush=True)

    print(f"\n{len(pairs)} cases cached")


def load(cache_dir, cid, factor):
    d = np.load(os.path.join(cache_dir, f"x{factor}", f"{cid}.npz"))
    return d["img"], d["affine"]


def load_label(cache_dir, cid):
    d = np.load(os.path.join(cache_dir, "label", f"{cid}.npz"))
    return d["label"], d["affine"]


def case_ids(cache_dir):
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(cache_dir, "x1", "*.npz")))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--out-dir", default=os.path.join(here, "cache"))
    p.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    build(a.images_dir, a.labels_dir, a.out_dir, tuple(a.factors), a.limit)
