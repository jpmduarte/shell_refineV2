# Is the polar head good enough for biometry, and better than the SDF it comes with?
#
#     python size_normalization_approach/evaluation/polar_radii_test.py --run nnaug_x4normpolar --seeds 42
#
# For every case (each by its own fold's networks): pass 1 gives the head's volume and
# centre (the centroid of its mask), pass 2 runs on the normalised grid exactly as in
# infer.py, and from pass 1's centre the head's radius is read in the 64 directions of
# polar.py, three ways, all in mm:
#     polar    the polar head's output at the centre
#     sdf      rays traced on pass 2's own SDF: what the method gives without the head
#     label    rays traced on the label's SDF: the reference
# plus the label's rays from the label's own centroid, to see what pass 1's centre costs.
# Writes runs/polar_radii_<run>.json and prints the errors against the label.

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
SNA    = os.path.dirname(HERE)          # size_normalization_approach/: the method, its runs
ROOT   = os.path.dirname(SNA)
sys.path[:0] = [ROOT, SNA]

import polar  # noqa: E402
from cache import case_ids, load as load_image, load_label  # noqa: E402
from infer import forward, load_net, pad, to_grid_fast, volume_ml, zscore  # noqa: E402
from make_sdf import TRUNC_MM, load as load_sdf  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


# A point in mm (world) -> voxel coordinates of a grid with this affine.
def to_vox(affine, p_mm):
    return np.linalg.solve(affine, np.append(p_mm, 1.0))[:3]


# Radii (mm) traced from a voxel point on an SDF normalised by TRUNC_MM, on a grid of the
# given voxel size.
def traced_mm(sdf_norm, point_vox, spacing_mm, dirs):
    sdf_vox = np.asarray(sdf_norm, np.float32) * TRUNC_MM / spacing_mm
    return polar.ray_radii(sdf_vox, np.asarray(point_vox, np.float64)[None], dirs)[0] * spacing_mm


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Polar head radii against the label and the SDF.")
    p.add_argument("--run", default="nnaug_x4normpolar", help="the pass-2 run family (with a polar head)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(SNA, "runs", f"polar_radii_{a.run}.json")

    dev = torch.device("cuda")
    v0_ml = reference_v0(a.reference)
    dirs = polar.directions()
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42, splits_json=os.path.join(ROOT, "splits.json"))

    rows = []
    for seed in a.seeds:
        for fold in range(5):
            ck = lambda run: os.path.join(SNA, "runs", f"{run}_s{seed}", f"fold{fold}", "checkpoints", "phase1_last.pth")
            pass1 = load_net(ck("nnaug_x4e500"), dev)[0]
            net = load_net(ck(a.run), dev)[0]
            for cid in fold_split(folds, fold)[1]:
                img, aff1 = load_image(a.cache_dir, cid, 1)
                img = np.asarray(img, np.float32).squeeze()
                lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
                lab_sdf = np.asarray(load_sdf(a.cache_dir, cid, 1)[0], np.float32).squeeze()
                sp1 = float(np.abs(aff1[0, 0]))

                # Pass 1: volume and centre (mm), as infer.segment measures them.
                x1 = zscore(img, dev)
                x4, aff4 = to_grid_fast(x1, aff1, 4 * sp1)
                f4 = forward(pass1, x4, False)
                k = (volume_ml(f4, 4 * sp1) / v0_ml) ** (1 / 3)
                c_vox4 = torch.nonzero(f4 < 0).float().mean(0).cpu().numpy()
                centre_mm = (aff4 @ np.append(c_vox4, 1.0))[:3]

                # Pass 2 on the normalised grid, with its polar head.
                xn, affn = to_grid_fast(x1, aff1, 4 * sp1 * k)
                spn = float(np.abs(affn[0, 0]))
                sdf_n, radii_n = net.forward_with_polar(pad(xn, 96)[None, None])
                sdf_n = sdf_n[0, 0, :xn.shape[0], :xn.shape[1], :xn.shape[2]].cpu().numpy()
                radii_n = radii_n[0, :, :xn.shape[0], :xn.shape[1], :xn.shape[2]]

                # The polar head at the centre: trilinear read of its 64 channels.
                cn = to_vox(affn, centre_mm)
                grid = torch.tensor((2 * cn[::-1] + 1) / np.array(xn.shape[::-1]) - 1,
                                    dtype=torch.float32, device=dev).view(1, 1, 1, 1, 3)
                r_polar = torch.nn.functional.grid_sample(radii_n[None], grid, align_corners=False)
                r_polar = r_polar.view(-1).cpu().numpy() * polar.RADIUS_SCALE * spn

                r_sdf = traced_mm(sdf_n, cn, spn, dirs)
                c1 = to_vox(aff1, centre_mm)
                r_label = traced_mm(lab_sdf, c1, sp1, dirs)
                r_label_own = traced_mm(lab_sdf, np.argwhere(lab).mean(0), sp1, dirs)

                rows.append({"case": cid, "seed": seed, "fold": fold,
                             "centre_err_mm": float(np.linalg.norm((c1 - np.argwhere(lab).mean(0)) * sp1)),
                             "polar": r_polar.tolist(), "sdf": r_sdf.tolist(),
                             "label": r_label.tolist(), "label_own_centre": r_label_own.tolist()})
                e_p = np.abs(r_polar - r_label).mean()
                e_s = np.abs(r_sdf - r_label).mean()
                print(f"s{seed} f{fold} {cid[:30]:<30} centre off {rows[-1]['centre_err_mm']:4.1f} mm  "
                      f"radius error: polar {e_p:4.2f} mm, sdf {e_s:4.2f} mm", flush=True)
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f)

    # Per case, mean over the seeds; then over the cases.
    from scipy.stats import wilcoxon
    cases = sorted({r["case"] for r in rows})
    def err(key):
        return np.array([np.mean([np.abs(np.array(r[key]) - np.array(r["label"])).mean()
                                  for r in rows if r["case"] == c]) for c in cases])
    ep, es = err("polar"), err("sdf")
    print(f"\n{len(cases)} cases. Mean |radius error| against the label, from pass 1's centre:")
    print(f"  polar head  {ep.mean():.2f} mm (median {np.median(ep):.2f})")
    print(f"  traced SDF  {es.mean():.2f} mm (median {np.median(es):.2f})")
    print(f"  polar - SDF {ep.mean() - es.mean():+.2f} mm, Wilcoxon p = {wilcoxon(ep, es).pvalue:.2g}, "
          f"polar better on {(ep < es).sum()}/{len(cases)}")
    cerr = np.array([np.mean([r["centre_err_mm"] for r in rows if r["case"] == c]) for c in cases])
    shift = np.array([np.mean([np.abs(np.array(r["label"]) - np.array(r["label_own_centre"])).mean()
                               for r in rows if r["case"] == c]) for c in cases])
    print(f"  pass 1's centre is {np.median(cerr):.2f} mm (median) from the label's centroid; "
          f"the label's radii change by {np.median(shift):.2f} mm between the two centres")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
