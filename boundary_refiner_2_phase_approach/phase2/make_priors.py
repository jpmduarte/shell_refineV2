# The refiner's priors: an existing phase-1 network's prediction for every case, which
# make_crops.py (--prior-dir) then cuts the band around. The refiner trains no phase 1
# of its own; it refines another approach's output.
#
#     python boundary_refiner_2_phase_approach/phase2/make_priors.py --seeds 42
#         -> boundary_refiner_2_phase_approach/priors/x4_s42/<case>.npz
#     python boundary_refiner_2_phase_approach/phase2/make_priors.py --seeds 42 43 44 --tta
#         -> .../priors/x4_ens_s42-43-44_tta/   the 3-seed ensemble with TTA
#
# By default the networks are the plain x4 ones (size_normalization_approach's first
# pass, no size normalisation). Any other phase-1 networks can be used through
# --checkpoint (a pattern with {s} and {f}) and --cache-dir (with {s} when each seed has
# its own cache), with --out-dir. The size-normalised priors:
#     --checkpoint "size_normalization_approach/runs/nnaug_x4norm_s{s}/fold{f}/checkpoints/phase1_last.pth"
#     --cache-dir "caches/cache_norm4_est_s{s}" --out-dir boundary_refiner_2_phase_approach/priors/x4norm_ens_tta
#
# Each file holds the predicted SDF at native resolution (sdf) and the native affine
# (affine), the format make_crops.py and size_normalization_approach/evaluate.py read.
#
# Out-of-fold: every case is predicted by the network of the fold where it was
# validation, so no prior comes from a network that trained on that case. Phase 2 then
# learns to correct the errors phase 1 really makes on unseen scans, not the smaller ones
# it makes on its own training data.

import argparse
import itertools
import os
import sys
import zipfile

import numpy as np
import torch
from numpy.lib import format as npy_format

HERE = os.path.dirname(os.path.abspath(__file__))
APPROACH = os.path.dirname(HERE)
ROOT = os.path.dirname(APPROACH)
sys.path.insert(0, ROOT)

from cache import case_ids, load as load_image  # noqa: E402
from geometry import apply_resample, resample_grid  # noqa: E402
from model import UNet3D  # noqa: E402
from nnunet_augmentation import pad_to_cube  # noqa: E402
from sdf_dataset import CUBE, FACTOR  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


# The native shape and affine of a case, read from the .npy header inside the .npz
# without decompressing the voxels.
def native_geometry(cache_dir: str, cid: str):
    path = os.path.join(cache_dir, "x1", f"{cid}.npz")
    with np.load(path) as d:
        affine = d["affine"]
    with zipfile.ZipFile(path) as z, z.open("img.npy") as f:
        major, _ = npy_format.read_magic(f)
        read_header = (npy_format.read_array_header_1_0 if major == 1
                       else npy_format.read_array_header_2_0)
        shape, _, _ = read_header(f)
    return tuple(shape), affine


# The 8 mirrorings for TTA: every subset of the three axes, from none to all three.
FLIPS = [[a for a, f in enumerate(fl) if f] for fl in itertools.product((0, 1), repeat=3)]


# Predict one case on its coarse grid and carry the SDF to the native grid (on the GPU).
# With TTA: the image is mirrored before padding (so the scan stays at the cube's origin),
# each prediction is mirrored back, and the 8 SDFs are averaged.
@torch.no_grad()
def predict_one(model, cache_dir: str, cid: str, device, tta: bool = False):
    img4, affine4 = load_image(cache_dir, cid, FACTOR)
    native_shape, affine1 = native_geometry(cache_dir, cid)
    img4 = img4.astype(np.float32)
    s = img4.shape

    preds = []
    for axes in (FLIPS if tta else [[]]):
        im = np.ascontiguousarray(np.flip(img4, axes)) if axes else img4
        # Pad to the network's cube, predict, cut the padding off.
        x = torch.from_numpy(pad_to_cube(im, CUBE, cval=0.0)[np.newaxis, np.newaxis]).to(device)
        y = model(x)[:, :, :s[0], :s[1], :s[2]]
        preds.append(torch.flip(y, [a + 2 for a in axes]) if axes else y)
    pred = preds[0] if len(preds) == 1 else torch.stack(preds).mean(0)

    # Back to native resolution through the two affines; beyond the scan, "far outside".
    grid = resample_grid(affine4, affine1, native_shape, device=device).float()
    native = apply_resample(pred, grid, img4.shape, cval=1.0)
    return native[0, 0], affine1


def main():
    p = argparse.ArgumentParser(description="Out-of-fold phase 1 priors for the refiner.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="one seed, or several: their SDFs are averaged (an ensemble)")
    p.add_argument("--tta", action="store_true", help="average the 8 mirrored predictions")
    p.add_argument("--checkpoint", default=os.path.join(
        ROOT, "size_normalization_approach", "runs", "nnaug_x4e500_s{s}", "fold{f}",
        "checkpoints", "phase1_last.pth"),
        help="checkpoint path pattern, with {s} for the seed and {f} for the fold")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"),
                   help="the grid the networks read; may contain {s} for a cache per seed")
    p.add_argument("--splits-json", default=os.path.join(ROOT, "splits.json"))
    p.add_argument("--out-dir", default=None,
                   help="default: priors/x4_s<seed> (or x4_ens_s42-43-44, with _tta) here")
    a = p.parse_args()

    tag = f"s{a.seeds[0]}" if len(a.seeds) == 1 else "ens_s" + "-".join(map(str, a.seeds))
    out_dir = a.out_dir or os.path.join(APPROACH, "priors", f"x4_{tag}" + ("_tta" if a.tta else ""))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = get_folds(case_ids(a.cache_dir.format(s=a.seeds[0])), n_folds=5, seed=42,
                      splits_json=a.splits_json)

    for k in range(len(folds)):
        # The fold-k networks of every seed, and the cases none of them trained on.
        models = []
        for seed in a.seeds:
            m = UNet3D().to(device)
            m.load_state_dict(torch.load(a.checkpoint.format(s=seed, f=k), map_location=device))
            models.append(m.eval())
        _, val_ids = fold_split(folds, k)

        for cid in val_ids:
            # Each seed on its own cache (the size-normalised grids differ per seed); the
            # SDFs are averaged at native resolution, where they share one grid.
            sdfs = [predict_one(m, a.cache_dir.format(s=seed), cid, device, a.tta)
                    for m, seed in zip(models, a.seeds)]
            sdf = sdfs[0][0] if len(sdfs) == 1 else torch.stack([s for s, _ in sdfs]).mean(0)
            np.savez(os.path.join(out_dir, f"{cid}.npz"),
                     sdf=sdf.cpu().numpy().astype(np.float32), affine=sdfs[0][1])
        print(f"fold {k}: {len(val_ids)} priors", flush=True)

    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
