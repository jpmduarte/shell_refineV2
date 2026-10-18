# v2: are the polar branch's radii good, and better than rays traced on the SDF?
#
#     python polar_approach/radii_test.py --run x4normpolar2 --seeds 42
#
# For every case (its own fold's networks): pass 1 gives the head's volume and centre,
# pass 2 (the U-Net of the run, with its polar branch) runs on the normalised grid as in
# infer.py, and from pass 1's centre the head's radius in the 64 directions is read three
# ways, in mm: the polar branch, rays traced on pass 2's SDF, and rays traced on the
# label's SDF (the reference). Writes runs/polar_radii_<run>.json.

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNA = os.path.join(ROOT, "size_normalization_approach")
sys.path[:0] = [ROOT, SNA, HERE]

import polar  # noqa: E402
from cache import case_ids, load as load_image, load_label  # noqa: E402
from infer import forward, load_net, pad, to_grid_fast, volume_ml, zscore  # noqa: E402
from make_sdf import TRUNC_MM, load as load_sdf  # noqa: E402
from polar_model import PolarNet  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def traced_mm(sdf_norm, point_vox, spacing_mm, dirs):
    sdf_vox = np.asarray(sdf_norm, np.float32) * TRUNC_MM / spacing_mm
    return polar.ray_radii(sdf_vox, np.asarray(point_vox, np.float64)[None], dirs)[0] * spacing_mm


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="v2 polar radii against the label and the SDF.")
    p.add_argument("--run", default="x4normpolar2")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--cache-dir", default=os.path.join(ROOT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(ROOT, "caches", "cache_norm4"))
    a = p.parse_args()
    out = os.path.join(SNA, "runs", f"polar_radii_{a.run}.json")

    dev = torch.device("cuda")
    v0_ml = reference_v0(a.reference)
    dirs = polar.directions()
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42, splits_json=os.path.join(ROOT, "splits.json"))

    rows = []
    for seed in a.seeds:
        for fold in range(5):
            ck = os.path.join(SNA, "runs", f"{a.run}_s{seed}", f"fold{fold}", "checkpoints")
            pass1 = load_net(os.path.join(SNA, "runs", f"nnaug_x4e500_s{seed}", f"fold{fold}",
                                          "checkpoints", "phase1_last.pth"), dev)[0]
            model = PolarNet().to(dev).eval()
            model.net.load_state_dict(torch.load(os.path.join(ck, "phase1_last.pth"), map_location=dev))
            model.mlp.load_state_dict(torch.load(os.path.join(ck, "polar_last.pth"), map_location=dev))
            for cid in fold_split(folds, fold)[1]:
                img, aff1 = load_image(a.cache_dir, cid, 1)
                img = np.asarray(img, np.float32).squeeze()
                lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
                lab_sdf = np.asarray(load_sdf(a.cache_dir, cid, 1)[0], np.float32).squeeze()
                sp1 = float(np.abs(aff1[0, 0]))

                x1 = zscore(img, dev)
                x4, aff4 = to_grid_fast(x1, aff1, 4 * sp1)
                f4 = forward(pass1, x4, False)
                k = (volume_ml(f4, 4 * sp1) / v0_ml) ** (1 / 3)
                centre_mm = (aff4 @ np.append(torch.nonzero(f4 < 0).float().mean(0).cpu().numpy(), 1.0))[:3]

                xn, affn = to_grid_fast(x1, aff1, 4 * sp1 * k)
                spn = float(np.abs(affn[0, 0]))
                cn = np.linalg.solve(affn, np.append(centre_mm, 1.0))[:3]
                xin = pad(xn, 96)[None, None]
                sdf_n, log_r = model(xin, torch.tensor(cn[None], dtype=torch.float32, device=dev))
                sdf_n = sdf_n[0, 0, :xn.shape[0], :xn.shape[1], :xn.shape[2]].cpu().numpy()
                r_polar = torch.exp(log_r[0]).cpu().numpy() * spn

                r_sdf = traced_mm(sdf_n, cn, spn, dirs)
                c1 = np.linalg.solve(aff1, np.append(centre_mm, 1.0))[:3]
                r_label = traced_mm(lab_sdf, c1, sp1, dirs)
                rows.append({"case": cid, "seed": seed, "fold": fold,
                             "polar": r_polar.tolist(), "sdf": r_sdf.tolist(), "label": r_label.tolist()})
                print(f"s{seed} f{fold} {cid[:30]:<30} radius error: polar "
                      f"{np.abs(r_polar - r_label).mean():4.2f} mm, sdf {np.abs(r_sdf - r_label).mean():4.2f} mm",
                      flush=True)
                torch.cuda.empty_cache()

    with open(out, "w") as f:
        json.dump(rows, f)
    from scipy.stats import wilcoxon
    cases = sorted({r["case"] for r in rows})
    def err(key):
        return np.array([np.mean([np.abs(np.array(r[key]) - np.array(r["label"])).mean()
                                  for r in rows if r["case"] == c]) for c in cases])
    ep, es = err("polar"), err("sdf")
    print(f"\n{len(cases)} cases. Mean |radius error| from pass 1's centre: polar {ep.mean():.2f} mm "
          f"(median {np.median(ep):.2f}), traced SDF {es.mean():.2f} mm (median {np.median(es):.2f}); "
          f"polar better on {(ep < es).sum()}/{len(cases)}, Wilcoxon p = {wilcoxon(ep, es).pvalue:.2g}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
