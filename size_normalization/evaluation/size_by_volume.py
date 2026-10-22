"""Real head sizes: does size normalisation help most on heads far from the median? No resampling.

The dataset's own spread (35-291 mL) and the cross-validation results on disk
(runs/<run>_s<seed>/fold<k>/eval_val/summary.json), per case averaged over the 3 seeds. Cases are
grouped by the label's volume and by |log(V / V0)|, and the methods compared per group (paired
Wilcoxon); Spearman says whether each method's Dice depends on the head's size.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from common.folds import N_FOLDS
from datasets import add_dataset_argument, get as get_dataset

SEEDS = (42, 43, 44)
METRICS = ("dice_label", "hd95", "asd")


def per_case(dataset, run: str) -> dict:
    """{case: {metric: mean over seeds}} for one run family."""
    scores = {}
    for s in SEEDS:
        for k in range(N_FOLDS):
            path = os.path.join(dataset.runs, f"{run}_s{s}", f"fold{k}", "eval_val", "summary.json")
            with open(path) as f:
                cases = json.load(f)["cases"]
            for c in cases:
                scores.setdefault(c["case"], []).append([c[m] for m in METRICS])
    return {cid: dict(zip(METRICS, np.mean(v, axis=0))) for cid, v in scores.items()}


def dice_of(results, ids):
    return np.array([results[c]["dice_label"] for c in ids])


def group_table(title, groups, plain, norm, vol):
    print(f"\n{title}\n")
    print("| Group | n | Volume range | Plain Dice | Normalised Dice | Difference | p | Normalised better "
          "| Plain HD95 | Normalised HD95 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, ids in groups:
        plain_dice, norm_dice = dice_of(plain, ids), dice_of(norm, ids)
        plain_hd95 = np.mean([plain[c]["hd95"] for c in ids])
        norm_hd95 = np.mean([norm[c]["hd95"] for c in ids])
        volumes = [vol[c] for c in ids]
        print(f"| {name} | {len(ids)} | {min(volumes):.0f}–{max(volumes):.0f} mL | {plain_dice.mean():.4f} | "
              f"{norm_dice.mean():.4f} | {norm_dice.mean() - plain_dice.mean():+.4f} | "
              f"{wilcoxon(norm_dice, plain_dice).pvalue:.2g} | {int((norm_dice > plain_dice).sum())}/{len(ids)} | "
              f"{plain_hd95:.2f} | {norm_hd95:.2f} |")


def group_names(n_groups, three_names, prefix):
    return three_names if n_groups == 3 else [f"{prefix}{i + 1}" for i in range(n_groups)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_argument(parser)
    parser.add_argument("--plain", default=None, help="default: the dataset's pass-1 run")
    parser.add_argument("--norm", default=None, help="default: the dataset's pass-2 run")
    parser.add_argument("--groups", type=int, default=3, help="volume groups (3: terciles)")
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)

    plain = per_case(dataset, args.plain or dataset.pass1_run)
    norm = per_case(dataset, args.norm or dataset.pass2_run)
    with open(os.path.join(dataset.norm_cache, "norm.json")) as f:
        meta = json.load(f)
    v0 = float(meta["v0_ml"])
    vol = {c: m["true_ml"] for c, m in meta["cases"].items()}
    ids = sorted(set(plain) & set(norm) & set(vol))
    print(f"{len(ids)} cases, V0 = {v0:.0f} mL; overall Dice plain "
          f"{np.mean([plain[c]['dice_label'] for c in ids]):.4f}, normalised "
          f"{np.mean([norm[c]['dice_label'] for c in ids]):.4f}")

    by_volume = sorted(ids, key=lambda c: vol[c])
    by_distance = sorted(ids, key=lambda c: abs(np.log(vol[c] / v0)))
    volume_names = group_names(args.groups, ["small", "medium", "large"], "q")
    distance_names = group_names(args.groups, ["near V0", "middle", "far from V0"], "d")
    group_table("By head volume", list(zip(volume_names, np.array_split(by_volume, args.groups))),
                plain, norm, vol)
    group_table("By distance from V0, |log(V / V0)|",
                list(zip(distance_names, np.array_split(by_distance, args.groups))), plain, norm, vol)

    volumes = np.array([vol[c] for c in ids])
    distance = np.abs(np.log(volumes / v0))
    plain_dice, norm_dice = dice_of(plain, ids), dice_of(norm, ids)
    print("\nSpearman correlation (rho, p) over the cases\n")
    print("| | with head volume | with distance from V0 |")
    print("|---|---|---|")
    for name, y in (("Plain Dice", plain_dice), ("Normalised Dice", norm_dice),
                    ("Normalised − plain", norm_dice - plain_dice)):
        rho_volume, p_volume = spearmanr(volumes, y)
        rho_distance, p_distance = spearmanr(distance, y)
        print(f"| {name} | {rho_volume:+.2f} (p = {p_volume:.2g}) | {rho_distance:+.2f} (p = {p_distance:.2g}) |")


if __name__ == "__main__":
    main()
