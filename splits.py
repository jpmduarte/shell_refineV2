"""5-fold case splits, cached to disk so every run (and every phase) sees the same
folds.

    splits.json: {"folds": [[case_id, ...] x5]}

Fold k's val set is folds[k]; its train set is every case in the other 4 folds.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_JSON = os.path.join(HERE, "splits.json")


def make_folds(case_ids: list[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    ids = sorted(case_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return [sorted(ids[i] for i in order[k::n_folds]) for k in range(n_folds)]


def get_folds(case_ids: list[str] = None, n_folds: int = 5, seed: int = 42,
             splits_json: str = None) -> list[list[str]]:
    path = splits_json or SPLITS_JSON

    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        print(f"Loaded folds from {path}")
        return saved["folds"]

    if case_ids is None:
        raise RuntimeError(f"{path} does not exist and no case_ids were given to build it")

    folds = make_folds(case_ids, n_folds, seed)
    with open(path, "w") as f:
        json.dump({"folds": folds}, f, indent=2)
    print(f"Saved {n_folds} folds to {path}  ({[len(f) for f in folds]} cases each)")
    return folds


def fold_split(folds: list[list[str]], fold: int) -> tuple[list[str], list[str]]:
    val = folds[fold]
    train = [cid for k, f in enumerate(folds) if k != fold for cid in f]
    return train, val
