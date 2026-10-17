# Tables from pass1_error_analysis.py's output.
#
#     python size_normalization_approach/evaluation/pass1_error_summary.py

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

SNA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=os.path.join(SNA, "runs", "pass1_error_analysis.json"))
    a = p.parse_args()
    rows = json.load(open(a.json))
    true = np.array([r["true_ml"] for r in rows])

    def per_case(key):   # mean over the seeds
        return np.array([np.mean([s[key] for s in r["seeds"].values()]) for r in rows])

    print(f"{len(rows)} cases, volumes {true.min():.0f}-{true.max():.0f} mL\n")
    print("| Volume | median abs error | signed mean | under by > 5% | over by > 5% | worst |")
    print("|---|---|---|---|---|---|")
    for name, key in (("pass 1, x4 count (what k uses)", "pass1_est_ml"),
                      ("pass 1, native mask", "pass1_ml"), ("pass 2, native mask", "pass2_ml")):
        e = per_case(key) / true - 1
        print(f"| {name} | {100 * np.median(abs(e)):.1f}% | {100 * e.mean():+.1f}% | "
              f"{(e < -0.05).sum()} | {(e > 0.05).sum()} | {100 * e[np.argmax(abs(e))]:+.1f}% |")

    e1 = per_case("pass1_est_ml") / true - 1
    e2 = per_case("pass2_ml") / true - 1
    print(f"\npass 1 vs pass 2 signed error, Spearman rho {spearmanr(e1, e2)[0]:+.2f}")
    s42 = [r["seeds"]["42"] for r in rows]

    # Where the error is, seed 42, pass 1: missed (FN) and added (FP).
    for which in ("pass1", "pass2"):
        fn = np.array([s[which]["fn"]["ml"] for s in s42])
        fp = np.array([s[which]["fp"]["ml"] for s in s42])
        fn_thin = np.array([s[which]["fn"]["thin_frac"] for s in s42])
        fp_thin = np.array([s[which]["fp"]["thin_frac"] for s in s42])
        fn_edge = np.array([s[which]["fn"]["edge_frac"] for s in s42])
        fp_edge = np.array([s[which]["fp"]["edge_frac"] for s in s42])
        print(f"\n{which} (seed 42): per case, mL and the share of it")
        print("| | mean mL | thin (≤ 1.2 mm from the surface) | at the sector edge (≤ 4 mm) |")
        print("|---|---|---|---|")
        print(f"| missed (FN) | {fn.mean():.1f} | {np.average(fn_thin, weights=fn):.0%} | {np.average(fn_edge, weights=fn):.0%} |")
        print(f"| added (FP) | {fp.mean():.1f} | {np.average(fp_thin, weights=fp):.0%} | {np.average(fp_edge, weights=fp):.0%} |")
        # Thick missed pieces: the FN volume not in the thin shell.
        thick = fn * (1 - fn_thin)
        print(f"thick missed pieces: {thick.mean():.1f} mL per case on average, "
              f"{(thick > 5).sum()} cases with more than 5 mL")

    # What goes with an underestimate: a cut head, a dark boundary, the head's size.
    edge = np.array([s["pass1"]["surface_at_edge_frac"] for s in s42])
    dark = np.array([(s["pass1"]["img_surface_missed"] or 0) - (s["pass1"]["img_surface_found"] or 0)
                     for s in s42])
    off = per_case("pass1_offset_mm")
    print("\nSpearman with pass 1's signed volume error (over the 90 cases)")
    print("| | rho | p |")
    print("|---|---|---|")
    for name, x in (("share of the true surface at the sector edge (cut head)", edge),
                    ("image at missed minus found surface (darker = negative)", dark),
                    ("true volume", true)):
        r, pv = spearmanr(x, e1)
        print(f"| {name} | {r:+.2f} | {pv:.2g} |")
    print(f"\nshell offset that would give the true volume (pass 1, mean over seeds): median "
          f"{np.median(off):+.2f} mm, IQR {np.percentile(off, 25):+.2f} to {np.percentile(off, 75):+.2f} mm "
          f"(negative: the predicted surface lies outside the true one)")
    cut = edge > 0.05
    print(f"cut heads (> 5% of the surface at the edge): {cut.sum()}; pass-1 error median "
          f"{100 * np.median(e1[cut]):+.1f}% vs {100 * np.median(e1[~cut]):+.1f}% for the others")

    print("\nThe 10 largest pass-1 errors")
    print("| case | true mL | pass 1 error | pass 2 error | surface at edge | FN thick mL | FP mL |")
    print("|---|---|---|---|---|---|---|")
    for i in np.argsort(-abs(e1))[:10]:
        s = s42[i]["pass1"]
        print(f"| {rows[i]['case'][:30]} | {true[i]:.0f} | {100 * e1[i]:+.1f}% | {100 * e2[i]:+.1f}% | "
              f"{edge[i]:.0%} | {s['fn']['ml'] * (1 - s['fn']['thin_frac']):.1f} | {s['fp']['ml']:.1f} |")


if __name__ == "__main__":
    main()
