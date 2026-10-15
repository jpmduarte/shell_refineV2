# Real head-size test: does size normalisation help most on heads far from the median?
#
#     python size_normalization_approach/size_by_volume.py
#
# No resampling, unlike size_shift_test.py: the dataset's own spread of head sizes (35-291
# mL) and the cross-validation results already on disk. Per case, the Dice / HD95 / ASD of
# each method averaged over its 3 seeds (each case scored by the fold that held it out):
#     plain x4    runs/nnaug_x4e500_s<seed>/fold<k>/eval_val/summary.json
#     normalised  runs/nnaug_x4norm_s<seed>/fold<k>/eval_val/summary.json
# The head volume is the label's (caches/cache_norm4/norm.json, true_ml). Cases are
# grouped by volume and by distance from the reference size V0, |log(V / V0)|, and the two
# methods compared within each group with a paired Wilcoxon test; Spearman correlations
# say whether each method's Dice depends on the head's size.

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
SEEDS  = (42, 43, 44)
METRICS = ("dice_label", "hd95", "asd")


# {case: {metric: mean over seeds}} for one run family.
def per_case(run: str) -> dict:
    acc = {}
    for s in SEEDS:
        for k in range(5):
            path = os.path.join(HERE, "runs", f"{run}_s{s}", f"fold{k}", "eval_val", "summary.json")
            for c in json.load(open(path))["cases"]:
                acc.setdefault(c["case"], []).append([c[m] for m in METRICS])
    return {cid: dict(zip(METRICS, np.mean(v, axis=0))) for cid, v in acc.items()}


def group_table(title, groups, plain, norm, vol):
    print(f"\n{title}\n")
    print("| Group | n | Volume range | Plain Dice | Normalised Dice | Difference | p | Normalised better "
          "| Plain HD95 | Normalised HD95 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, ids in groups:
        P = np.array([plain[c]["dice_label"] for c in ids])
        N = np.array([norm[c]["dice_label"] for c in ids])
        Ph = np.mean([plain[c]["hd95"] for c in ids])
        Nh = np.mean([norm[c]["hd95"] for c in ids])
        v = [vol[c] for c in ids]
        print(f"| {name} | {len(ids)} | {min(v):.0f}–{max(v):.0f} mL | {P.mean():.4f} | {N.mean():.4f} | "
              f"{N.mean() - P.mean():+.4f} | {wilcoxon(N, P).pvalue:.2g} | {int((N > P).sum())}/{len(ids)} | "
              f"{Ph:.2f} | {Nh:.2f} |")


def main():
    p = argparse.ArgumentParser(description="Plain vs size-normalised x4 by real head size.")
    p.add_argument("--plain", default="nnaug_x4e500")
    p.add_argument("--norm", default="nnaug_x4norm")
    p.add_argument("--groups", type=int, default=3, help="volume groups (3: terciles)")
    a = p.parse_args()

    plain, norm = per_case(a.plain), per_case(a.norm)
    meta = json.load(open(os.path.join(PARENT, "caches", "cache_norm4", "norm.json")))
    v0 = float(meta["v0_ml"])
    vol = {c: m["true_ml"] for c, m in meta["cases"].items()}
    ids = sorted(set(plain) & set(norm) & set(vol))
    print(f"{len(ids)} cases, V0 = {v0:.0f} mL; overall Dice plain "
          f"{np.mean([plain[c]['dice_label'] for c in ids]):.4f}, normalised "
          f"{np.mean([norm[c]['dice_label'] for c in ids]):.4f}")

    # Groups of equal size by volume, and by distance from V0.
    by_vol = sorted(ids, key=lambda c: vol[c])
    by_dist = sorted(ids, key=lambda c: abs(np.log(vol[c] / v0)))
    names_v = ["small", "medium", "large"] if a.groups == 3 else [f"q{i + 1}" for i in range(a.groups)]
    names_d = ["near V0", "middle", "far from V0"] if a.groups == 3 else [f"d{i + 1}" for i in range(a.groups)]
    group_table("By head volume", list(zip(names_v, np.array_split(by_vol, a.groups))), plain, norm, vol)
    group_table("By distance from V0, |log(V / V0)|", list(zip(names_d, np.array_split(by_dist, a.groups))),
                plain, norm, vol)

    # Does Dice depend on size? Spearman over the 90 cases.
    v = np.array([vol[c] for c in ids])
    d = np.abs(np.log(v / v0))
    P = np.array([plain[c]["dice_label"] for c in ids])
    N = np.array([norm[c]["dice_label"] for c in ids])
    print("\nSpearman correlation (rho, p) over the cases\n")
    print("| | with head volume | with distance from V0 |")
    print("|---|---|---|")
    for name, y in (("Plain Dice", P), ("Normalised Dice", N), ("Normalised − plain", N - P)):
        r1, p1 = spearmanr(v, y)
        r2, p2 = spearmanr(d, y)
        print(f"| {name} | {r1:+.2f} (p = {p1:.2g}) | {r2:+.2f} (p = {p2:.2g}) |")


if __name__ == "__main__":
    main()
