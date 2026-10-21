"""Where pass 1's head-volume error (4.3% median, systematic per case) comes from, for pass 1 and pass 2.

Per case, at native resolution against the label (one model, no TTA):
    FN / FP        the head volume missed and added (mL)
    thin / thick   error voxels within 1.2 mm of the true surface (a contour drawn slightly off) or further
    field of view  error near the edge of the ultrasound sector (no echo recorded) or inside it
    intensity      the image along the true surface where it was missed vs found (shadowing, dropout?)
    shell offset   the uniform SDF shift that would give the true volume
Each case is scored by its fold's networks: all 3 seeds for the volumes, seed 42 for the breakdown.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import ndimage

from common.folds import N_FOLDS, SEED as FOLDS_SEED, fold_split, get_folds
from common.geometry import MM3_PER_ML, voxel_spacing
from common.image_cache import case_ids, load as load_image, load_label
from common.paths import NATIVE_CACHE, NORM_CACHE, PASS1_RUN, PASS2_RUN, SIZE_NORM_RUNS, SPLITS_JSON, run_checkpoint
from common.sdf_targets import TRUNC_MM
from size_normalization.build_cache import reference_v0
from size_normalization.infer import load_net, segment, segment_plain

SEEDS = (42, 43, 44)
BREAKDOWN_SEED = 42
THIN_MM = 1.2   # 3 voxels: an error voxel this close to the true surface is a contour offset
EDGE_MM = 4.0   # this close to the sector's edge counts as "at the edge"
OPENING_ITERATIONS = 2


def field_of_view(img: np.ndarray) -> np.ndarray:
    """The sector: voxels with any echo, small specks removed, holes (dark fluid) filled."""
    fov = img > img.min()
    fov = ndimage.binary_opening(fov, iterations=OPENING_ITERATIONS)
    return ndimage.binary_fill_holes(fov)


def shell_offset_mm(sdf: np.ndarray, true_voxels: int) -> float:
    """The shift t of the normalised SDF whose {sdf < t} has the true volume; negative: the true surface is inside."""
    ordered = np.sort(sdf.ravel())
    return float(ordered[min(true_voxels, ordered.size - 1)] * TRUNC_MM)


def spatial(pred, label, fov, img, spacing):
    false_negative, false_positive = label & ~pred, pred & ~label
    surface = label ^ ndimage.binary_erosion(label)
    to_surface_mm = ndimage.distance_transform_edt(~surface, sampling=spacing)
    to_edge_mm = ndimage.distance_transform_edt(fov, sampling=spacing)
    voxel_ml = float(np.prod(spacing)) / MM3_PER_ML
    out = {}
    for name, error in (("fn", false_negative), ("fp", false_positive)):
        n = int(error.sum())
        out[name] = {
            "ml": n * voxel_ml,
            "thin_frac": float((to_surface_mm[error] <= THIN_MM).mean()) if n else 0.0,
            "edge_frac": float((to_edge_mm[error] <= EDGE_MM).mean()) if n else 0.0,
        }
    missed, found = surface & ~pred, surface & pred
    out["surface_at_edge_frac"] = float((to_edge_mm[surface] <= EDGE_MM).mean())   # a cut head
    out["img_surface_missed"] = float(img[missed].mean()) if missed.any() else None
    out["img_surface_found"] = float(img[found].mean()) if found.any() else None
    out["surface_missed_frac"] = float(missed.sum() / surface.sum())
    return out


def seed_record(img, aff, label, voxel_ml, pass1, pass2, v0_ml, device, with_breakdown):
    field1 = segment_plain(img, aff, [pass1], device=device).cpu().numpy()
    field2, estimate = segment(img, aff, pass1, [pass2], v0_ml, device=device)
    field2 = field2.cpu().numpy()
    mask1, mask2 = field1 < 0, field2 < 0
    record = {"pass1_ml": mask1.sum() * voxel_ml, "pass2_ml": mask2.sum() * voxel_ml,
              "pass1_est_ml": estimate.volume_ml,
              "pass1_offset_mm": shell_offset_mm(field1, int(label.sum())),
              "pass2_offset_mm": shell_offset_mm(field2, int(label.sum()))}
    if with_breakdown:
        fov = field_of_view(img)
        spacing = voxel_spacing(aff)
        record["pass1"] = spatial(mask1, label, fov, img, spacing)
        record["pass2"] = spatial(mask2, label, fov, img, spacing)
    return record


def print_case(fold, cid, row):
    r = row["seeds"][BREAKDOWN_SEED]
    print(f"fold {fold} {cid[:30]:<30} true {row['true_ml']:6.1f}  pass 1 {r['pass1_ml']:6.1f} "
          f"(est {r['pass1_est_ml']:6.1f})  pass 2 {r['pass2_ml']:6.1f}  "
          f"FN {r['pass1']['fn']['ml']:5.1f} (thin {r['pass1']['fn']['thin_frac']:.2f}, "
          f"edge {r['pass1']['fn']['edge_frac']:.2f})  FP {r['pass1']['fp']['ml']:5.1f}  "
          f"offset {r['pass1_offset_mm']:+.2f} mm", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=NATIVE_CACHE)
    parser.add_argument("--reference", default=NORM_CACHE)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases per fold (testing)")
    parser.add_argument("--out", default=os.path.join(SIZE_NORM_RUNS, "pass1_error_analysis.json"))
    args = parser.parse_args()

    device = torch.device("cuda")
    v0_ml = reference_v0(args.reference)
    folds = get_folds(case_ids(args.cache_dir), n_folds=N_FOLDS, seed=FOLDS_SEED, splits_json=SPLITS_JSON)

    rows = []
    for fold in range(N_FOLDS):
        pass1 = {s: load_net(run_checkpoint(PASS1_RUN, s, fold), device) for s in SEEDS}
        pass2 = {s: load_net(run_checkpoint(PASS2_RUN, s, fold), device) for s in SEEDS}
        for cid in fold_split(folds, fold)[1][:args.limit or None]:
            img, aff = load_image(args.cache_dir, cid, 1)
            img = np.asarray(img, np.float32).squeeze()
            label = np.asarray(load_label(args.cache_dir, cid)[0]).squeeze().astype(bool)
            voxel_ml = float(np.prod(voxel_spacing(aff))) / MM3_PER_ML
            row = {"case": cid, "fold": fold, "true_ml": label.sum() * voxel_ml,
                   "seeds": {s: seed_record(img, aff, label, voxel_ml, pass1[s], pass2[s], v0_ml, device,
                                            with_breakdown=s == BREAKDOWN_SEED)
                             for s in SEEDS}}
            rows.append(row)
            print_case(fold, cid, row)
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} cases -> {args.out}")


if __name__ == "__main__":
    main()
