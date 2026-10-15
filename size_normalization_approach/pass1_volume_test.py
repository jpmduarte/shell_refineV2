# Better head volumes for the size normalisation: does a better estimate improve the result?
#
#     python size_normalization_approach/pass1_volume_test.py
#
# The normalised network sees each head at the reference size V0, using the head volume
# estimated by pass 1 (one pass of the plain x4 network). Here the volume fed to pass 2 is
# estimated in several ways, everything else fixed (pass 2: one model, no TTA, each seed
# on its own fold's networks), over all 90 cases and the 3 seeds:
#     single      pass 1 once (the deployed method)
#     tta         pass 1 averaged over its 8 mirrorings
#     ens_tta     pass 1 of the 3 seeds, each with TTA, averaged (same fold: no leak)
#     iter1/2     from `single`, re-estimate the volume on pass 2's own output and run
#                 pass 2 again, once or twice
#     ens_iter1   ens_tta, then one such iteration
#     oracle      the label's volume: the best any estimate can do
# For each: Dice / HD95 at native resolution and the error of the volume pass 2 was run
# with. Pass 2 resamples from the native volume in every variant (no double interpolation).

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, "scripts"))

from bench_pipeline import forward, to_grid_fast, to_native_fast  # noqa: E402
from cache import case_ids, load as load_image, load_label  # noqa: E402
from metrics import hd95  # noqa: E402
from model import UNet3D  # noqa: E402
from size_norm import reference_v0  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402

SEEDS = (42, 43, 44)
VARIANTS = ("single", "tta", "ens_tta", "iter1", "iter2", "ens_iter1", "oracle")


# Head volume (mL) of an SDF on a grid of voxel size `spacing` mm: voxels inside, times the
# voxel volume.
def volume_ml(field: torch.Tensor, spacing: float) -> float:
    return float((field < 0).sum()) * spacing ** 3 / 1000


# Pass 2 for a given head volume: native image -> grid that brings the head to V0 -> network
# -> SDF. Returns the SDF on the native grid and the volume measured on pass 2's own grid.
@torch.no_grad()
def pass2(x1, aff1, net, vol_ml, v0_ml):
    native_spacing = float(np.abs(aff1[0, 0]))
    k = (vol_ml / v0_ml) ** (1 / 3)
    xn, affn = to_grid_fast(x1, aff1, 4 * native_spacing * k)
    fn = forward(net, xn, False)
    return to_native_fast(fn, affn, aff1, x1.shape), volume_ml(fn, 4 * native_spacing * k)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def main():
    p = argparse.ArgumentParser(description="Pass-1 volume variants for the size normalisation.")
    p.add_argument("--cache-dir", default=os.path.join(PARENT, "caches", "cache"))
    p.add_argument("--reference", default=os.path.join(PARENT, "caches", "cache_norm4"))
    p.add_argument("--out", default=os.path.join(HERE, "runs", "pass1_volume_test.json"))
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v0_ml = reference_v0(a.reference)
    folds = get_folds(case_ids(a.cache_dir), n_folds=5, seed=42,
                      splits_json=os.path.join(PARENT, "splits.json"))

    def load_net(run, seed, fold):
        net = UNet3D().to(dev).eval().requires_grad_(False)
        path = os.path.join(HERE, "runs", f"{run}_s{seed}", f"fold{fold}", "checkpoints", "phase1_last.pth")
        net.load_state_dict(torch.load(path, map_location=dev))
        return net

    rows = []
    for fold in range(5):
        pass1 = {s: load_net("nnaug_x4e500", s, fold) for s in SEEDS}
        norm = {s: load_net("nnaug_x4norm", s, fold) for s in SEEDS}
        for cid in fold_split(folds, fold)[1]:
            img1, aff1 = load_image(a.cache_dir, cid, 1)
            lab = np.asarray(load_label(a.cache_dir, cid)[0]).squeeze().astype(bool)
            sp_mm = np.abs(np.diag(aff1[:3, :3]))
            native_spacing = float(sp_mm[0])
            true_ml = float(lab.sum()) * native_spacing ** 3 / 1000

            x1 = torch.from_numpy(np.asarray(img1, np.float32).squeeze()).to(dev)
            x1 = (x1 - x1.mean()) / x1.std()
            x4, _ = to_grid_fast(x1, aff1, 4 * native_spacing)
            vox4 = 4 * native_spacing
            with torch.no_grad():
                p1 = {s: (forward(pass1[s], x4, False), forward(pass1[s], x4, True)) for s in SEEDS}
            ens_ml = volume_ml(torch.stack([p1[s][1] for s in SEEDS]).mean(0), vox4)

            for s in SEEDS:
                vols = {"single": volume_ml(p1[s][0], vox4), "tta": volume_ml(p1[s][1], vox4),
                        "ens_tta": ens_ml, "oracle": true_ml}
                out = {}
                for name in ("single", "tta", "ens_tta", "oracle"):
                    out[name] = pass2(x1, aff1, norm[s], vols[name], v0_ml)
                # Iterations: the volume pass 2 measured on its own output, then pass 2 again.
                v = out["single"][1]
                vols["iter1"] = v
                out["iter1"] = pass2(x1, aff1, norm[s], v, v0_ml)
                vols["iter2"] = out["iter1"][1]
                out["iter2"] = pass2(x1, aff1, norm[s], vols["iter2"], v0_ml)
                vols["ens_iter1"] = out["ens_tta"][1]
                out["ens_iter1"] = pass2(x1, aff1, norm[s], vols["ens_iter1"], v0_ml)

                row = {"case": cid, "fold": fold, "seed": s, "true_ml": true_ml}
                for name in VARIANTS:
                    mask = (out[name][0] < 0).cpu().numpy()
                    row[name] = {"dice": dice(mask, lab), "hd95": hd95(mask, lab, sp_mm),
                                 "vol_err": vols[name] / true_ml - 1}
                rows.append(row)
            torch.cuda.empty_cache()
            print(f"fold {fold} {cid[:34]:<34} single {np.mean([r['single']['dice'] for r in rows[-3:]]):.4f}  "
                  f"ens_iter1 {np.mean([r['ens_iter1']['dice'] for r in rows[-3:]]):.4f}  "
                  f"oracle {np.mean([r['oracle']['dice'] for r in rows[-3:]]):.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)

    # Per case, the mean over the 3 seeds; then over the 90 cases.
    from scipy.stats import wilcoxon
    cases = sorted({r["case"] for r in rows})
    per = {v: np.array([np.mean([r[v]["dice"] for r in rows if r["case"] == c]) for c in cases]) for v in VARIANTS}
    hd = {v: np.mean([r[v]["hd95"] for r in rows]) for v in VARIANTS}
    print("\n| Volume fed to pass 2 | Dice | vs single | p | HD95 | Volume error, median abs | signed mean | max abs |")
    print("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        e = np.array([r[v]["vol_err"] for r in rows])
        pv = "—" if v == "single" else f"{wilcoxon(per[v], per['single']).pvalue:.2g}"
        print(f"| {v} | {per[v].mean():.4f} | {per[v].mean() - per['single'].mean():+.4f} | {pv} | {hd[v]:.2f} | "
              f"{100 * np.median(abs(e)):.1f}% | {100 * e.mean():+.1f}% | {100 * abs(e).max():.1f}% |")
    print(f"\n{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
