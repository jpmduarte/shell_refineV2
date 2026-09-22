"""How much of the true surface does the band actually contain?

Phase 2 may only rewrite voxels inside |sdf_prior| < band_mm; everywhere else keeps
phase 1's answer. So any part of the true boundary lying outside that band is
unreachable no matter how good the refiner is, and the band width sets a hard ceiling.

This was measured once before as a fraction of *voxels* and came to 0.065%, which
looked settled. But voxels are dominated by the easy majority of the surface: measured
per direction instead, the displacement from the prior to the truth had a p99 of 5.79mm
and a worst case of 10.61mm, which a 5mm band cannot reach. This counts ground-truth
surface voxels rather than volume, which is the quantity the ceiling actually depends
on.

    python phase2/band_coverage.py --crops-dir phase2/crops/p1v2
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

TRUNC_MM = 10.0
WIDTHS = (3.0, 5.0, 7.5, 10.0, 12.5)


def coverage(crop_path: str, widths=WIDTHS, trunc_mm: float = TRUNC_MM) -> dict:
    with np.load(crop_path) as d:
        prior = d["sdf_prior"].astype(np.float32)
        mask = d["mask"].astype(bool)

    # The true surface: mask voxels with a non-mask neighbour. Counting the shell rather
    # than the volume is what matters, since a band that misses a thin strip of boundary
    # loses that strip regardless of how much interior it contains.
    surface = mask & ~binary_erosion(mask)
    if not surface.any():
        return None

    prior_mm = np.abs(prior) * trunc_mm
    at_surface = prior_mm[surface]

    out = {"surface_voxels": int(surface.sum())}
    for w in widths:
        out[f"cov_{w}"] = float((at_surface < w).mean())
    # Where the prior places the surface against where it is: the distribution whose
    # tail the band has to cover.
    out["p50"] = float(np.percentile(at_surface, 50))
    out["p95"] = float(np.percentile(at_surface, 95))
    out["p99"] = float(np.percentile(at_surface, 99))
    out["max"] = float(at_surface.max())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops-dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    files = sorted(f for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    if a.limit:
        files = files[:a.limit]

    rows = []
    for f in files:
        r = coverage(os.path.join(a.crops_dir, f))
        if r:
            r["case"] = f[:-4]
            rows.append(r)

    hdr = "".join(f"{w:>8.1f}mm" for w in WIDTHS)
    print(f"{'case':<30}{hdr}{'p95':>8}{'p99':>8}{'max':>8}")
    print("-" * (30 + 10 * len(WIDTHS) + 24))
    for r in rows:
        cells = "".join(f"{100*r[f'cov_{w}']:9.2f}%" for w in WIDTHS)
        print(f"{r['case'][:28]:<30}{cells}{r['p95']:8.2f}{r['p99']:8.2f}{r['max']:8.2f}")
    print("-" * (30 + 10 * len(WIDTHS) + 24))

    print(f"\n{'band':<10}{'mean coverage':>16}{'worst case':>14}"
          f"{'cases below 99%':>18}")
    for w in WIDTHS:
        v = np.array([r[f"cov_{w}"] for r in rows])
        print(f"{w:<10.1f}{100*v.mean():15.3f}%{100*v.min():13.2f}%"
              f"{int((v < 0.99).sum()):13d}/{len(v)}")

    print(f"\ndistance from the prior's surface to the true one, over all cases:")
    for k, label in (("p50", "median"), ("p95", "p95"), ("p99", "p99"), ("max", "worst")):
        v = np.array([r[k] for r in rows])
        print(f"  {label:<8} {v.mean():6.2f} mm mean,  {v.max():6.2f} mm worst case")

    print("\n  A band whose coverage is already near 100% is not what limits the")
    print("  refiner, and widening it only adds voxels it has to get right. One that")
    print("  leaves a percent or two of surface outside is imposing a ceiling.")


if __name__ == "__main__":
    main()
