# Where pass 1's head-volume error comes from.
#
#     python size_normalization_approach/evaluation/pass1_error_analysis.py
#
# Pass 1 (the plain x4 network) measures the head volume the size normalisation needs; its
# error is 4.3% median, with almost no overall bias, and systematic per case (pass 2 errs
# the same way on the same heads). Here every case's error is taken apart, at native
# resolution, against the label, for pass 1 and for pass 2 (one model, no TTA):
#
#     FN / FP          the head volume missed and added (mL)
#     thin vs thick    error voxels within 1.2 mm (3 voxels) of the true surface — the
#                      contour drawn slightly inside or outside — against those further
#                      away: whole pieces missed or added
#     field of view    error near the edge of the ultrasound sector (where the image is
#                      exactly its minimum: no echo recorded) against inside it
#     intensity        the image along the true surface where it was missed against where
#                      it was found: is the missed boundary darker (shadowing, dropout)?
#     shell offset     the uniform shift of the predicted SDF that would give the true
#                      volume: how much of the error a per-case threshold could fix
#
# Each case is scored by the networks of the fold that held it out; all 3 seeds for the
# volumes, seed 42 for the spatial breakdown. Writes runs/pass1_error_analysis.json.

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

HERE   = os.path.dirname(os.path.abspath(__file__))
SNA    = os.path.dirname(HERE)          # size_normalization_approach/: the method, its runs
ROOT   = os.path.dirname(SNA)
sys.path[:0] = [ROOT, SNA]

from cache import case_ids, load as load_image, load_label  # noqa: E402
from infer import load_net, segment, segment_plain  # noqa: E402
from make_sdf import TRUNC_MM  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402

SEEDS = (42, 43, 44)
THIN_MM = 1.2       # an error voxel this close to the true surface is a contour offset
EDGE_MM = 4.0       # this close to the sector's edge counts as "at the edge"


# The ultrasound sector: voxels with any echo, holes filled (dark fluid inside the sector
# can reach the minimum too), small specks outside removed.
def field_of_view(img: np.ndarray) -> np.ndarray:
    fov = img > img.min()
    fov = ndimage.binary_opening(fov, iterations=2)
    return ndimage.binary_fill_holes(fov)


# The uniform shift t (mm) of a predicted SDF (normalised, x TRUNC_MM = mm) whose level set
# {sdf < t} has the true volume; negative: the true surface lies inside the predicted one.
def shell_offset_mm(sdf: np.ndarray, true_vox: int) -> float:
    d = np.sort(sdf.ravel())
    return float(d[min(true_vox, d.size - 1)] * TRUNC_MM)


def spatial(pred, lab, fov, img, spacing):
    fn, fp = lab & ~pred, pred & ~lab
    surf = lab ^ ndimage.binary_erosion(lab)
    # Distance (mm) of every voxel to the true surface, and to the sector's edge.
    d_surf = ndimage.distance_transform_edt(~surf, sampling=spacing)
    d_edge = ndimage.distance_transform_edt(fov, sampling=spacing)
    vox_ml = float(np.prod(spacing)) / 1000
    out = {}
    for name, err in (("fn", fn), ("fp", fp)):
        n = int(err.sum())
        out[name] = {
            "ml": n * vox_ml,
            "thin_frac": float((d_surf[err] <= THIN_MM).mean()) if n else 0.0,
            "edge_frac": float((d_edge[err] <= EDGE_MM).mean()) if n else 0.0,
        }
    # The true surface: how much of it lies at the sector's edge (a cut head), and the
    # image there where pass 1 missed it (outside the prediction) against where it found it.
    missed = surf & ~pred
    found = surf & pred
    out["surface_at_edge_frac"] = float((d_edge[surf] <= EDGE_MM).mean())
    out["img_surface_missed"] = float(img[missed].mean()) if missed.any() else None
    out["img_surface_found"] = float(img[found].mean()) if found.any() else None
    out["surface_missed_frac"] = float(missed.sum() / surf.sum())
    return out


def main():
    p = argparse.ArgumentParser(description="Where pass 1's volume error comes from.")
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"))
    p.add_argument("--limit", type=int, default=0, help="only the first N cases per fold (testing)")
    p.add_argument("--out", default=os.path.join(SNA, "runs", "pass1_error_analysis.json"))
    a = p.parse_args()

    dev = torch.device("cuda")
    v0_ml = reference_v0(a.reference)
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42, splits_json=os.path.join(ROOT, "splits.json"))

    def net(run, seed, fold):
        return load_net(os.path.join(SNA, "runs", f"{run}_s{seed}", f"fold{fold}", "checkpoints",
                                     "phase1_last.pth"), dev)[0]

    rows = []
    for fold in range(5):
        p1 = {s: net("nnaug_x4e500", s, fold) for s in SEEDS}
        p2 = {s: net("nnaug_x4norm", s, fold) for s in SEEDS}
        for cid in fold_split(folds, fold)[1][:a.limit or None]:
            img, aff = load_image(a.cache_dir, cid, 1)
            img = np.asarray(img, np.float32).squeeze()
            lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
            spacing = np.abs(np.diag(aff[:3, :3]))
            vox_ml = float(np.prod(spacing)) / 1000
            true_ml = lab.sum() * vox_ml
            row = {"case": cid, "fold": fold, "true_ml": true_ml, "seeds": {}}
            fov = None
            for s in SEEDS:
                f1 = segment_plain(img, aff, [p1[s]], dev=dev).cpu().numpy()
                f2, info = segment(img, aff, p1[s], [p2[s]], v0_ml, dev=dev)
                f2 = f2.cpu().numpy()
                m1, m2 = f1 < 0, f2 < 0
                r = {"pass1_ml": m1.sum() * vox_ml, "pass2_ml": m2.sum() * vox_ml,
                     "pass1_est_ml": info["volume_ml"],
                     "pass1_offset_mm": shell_offset_mm(f1, int(lab.sum())),
                     "pass2_offset_mm": shell_offset_mm(f2, int(lab.sum()))}
                if s == 42:
                    fov = field_of_view(img)
                    r["pass1"] = spatial(m1, lab, fov, img, spacing)
                    r["pass2"] = spatial(m2, lab, fov, img, spacing)
                row["seeds"][s] = r
            rows.append(row)
            r42 = row["seeds"][42]
            print(f"fold {fold} {cid[:30]:<30} true {true_ml:6.1f}  pass 1 {r42['pass1_ml']:6.1f} "
                  f"(est {r42['pass1_est_ml']:6.1f})  pass 2 {r42['pass2_ml']:6.1f}  "
                  f"FN {r42['pass1']['fn']['ml']:5.1f} (thin {r42['pass1']['fn']['thin_frac']:.2f}, "
                  f"edge {r42['pass1']['fn']['edge_frac']:.2f})  FP {r42['pass1']['fp']['ml']:5.1f}  "
                  f"offset {r42['pass1_offset_mm']:+.2f} mm", flush=True)
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} cases -> {a.out}")


if __name__ == "__main__":
    main()
