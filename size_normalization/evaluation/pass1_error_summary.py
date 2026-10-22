"""Tables from pass1_error_analysis.py's output."""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from datasets import add_dataset_argument, get as get_dataset

BREAKDOWN_SEED = "42"   # a JSON key
LARGE_ERROR = 0.05
THICK_PIECE_ML = 5
CUT_HEAD_SURFACE_FRAC = 0.05
N_WORST = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_argument(parser)
    parser.add_argument("--json", default=None, help="default: <the dataset's runs>/pass1_error_analysis.json")
    args = parser.parse_args()
    args.json = args.json or os.path.join(get_dataset(args.dataset).runs, "pass1_error_analysis.json")
    with open(args.json) as f:
        rows = json.load(f)
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
              f"{(e < -LARGE_ERROR).sum()} | {(e > LARGE_ERROR).sum()} | {100 * e[np.argmax(abs(e))]:+.1f}% |")

    e1 = per_case("pass1_est_ml") / true - 1
    e2 = per_case("pass2_ml") / true - 1
    print(f"\npass 1 vs pass 2 signed error, Spearman rho {spearmanr(e1, e2)[0]:+.2f}")
    s42 = [r["seeds"][BREAKDOWN_SEED] for r in rows]

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
        thick = fn * (1 - fn_thin)   # the FN volume outside the thin shell
        print(f"thick missed pieces: {thick.mean():.1f} mL per case on average, "
              f"{(thick > THICK_PIECE_ML).sum()} cases with more than {THICK_PIECE_ML} mL")

    # What goes with an underestimate: a cut head, a dark boundary, the head's size.
    edge = np.array([s["pass1"]["surface_at_edge_frac"] for s in s42])
    dark = np.array([(s["pass1"]["img_surface_missed"] or 0) - (s["pass1"]["img_surface_found"] or 0)
                     for s in s42])
    offset = per_case("pass1_offset_mm")
    print("\nSpearman with pass 1's signed volume error (over the 90 cases)")
    print("| | rho | p |")
    print("|---|---|---|")
    for name, x in (("share of the true surface at the sector edge (cut head)", edge),
                    ("image at missed minus found surface (darker = negative)", dark),
                    ("true volume", true)):
        rho, p_value = spearmanr(x, e1)
        print(f"| {name} | {rho:+.2f} | {p_value:.2g} |")
    print(f"\nshell offset that would give the true volume (pass 1, mean over seeds): median "
          f"{np.median(offset):+.2f} mm, IQR {np.percentile(offset, 25):+.2f} to {np.percentile(offset, 75):+.2f} mm "
          f"(negative: the predicted surface lies outside the true one)")
    cut = edge > CUT_HEAD_SURFACE_FRAC
    print(f"cut heads (> 5% of the surface at the edge): {cut.sum()}; pass-1 error median "
          f"{100 * np.median(e1[cut]):+.1f}% vs {100 * np.median(e1[~cut]):+.1f}% for the others")

    print(f"\nThe {N_WORST} largest pass-1 errors")
    print("| case | true mL | pass 1 error | pass 2 error | surface at edge | FN thick mL | FP mL |")
    print("|---|---|---|---|---|---|---|")
    for i in np.argsort(-abs(e1))[:N_WORST]:
        s = s42[i]["pass1"]
        print(f"| {rows[i]['case'][:30]} | {true[i]:.0f} | {100 * e1[i]:+.1f}% | {100 * e2[i]:+.1f}% | "
              f"{edge[i]:.0%} | {s['fn']['ml'] * (1 - s['fn']['thin_frac']):.1f} | {s['fp']['ml']:.1f} |")


if __name__ == "__main__":
    main()
