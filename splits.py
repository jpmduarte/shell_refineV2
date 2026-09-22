"""5-fold case splits, grouped by patient, cached so every run and every phase agrees.

    splits.json: {"folds": [[case_id, ...] x5], "grouped_by_patient": true}

Fold k's val set is folds[k]; its train set is every case in the other four.

Grouping matters here: 90 volumes come from 82 patients, so six patients contributed
more than one scan. Two scans of the same fetus taken in one session are far more alike
than two different fetuses, and splitting them across a fold boundary means testing on
something the model has effectively already seen. Shuffling case ids put five of those
six patients on both sides, which was worth +0.0011 Dice — under the seed noise, but
the sort of thing a reader is entitled to see ruled out rather than argued about.

Folds come out slightly uneven as a result, because patients rather than volumes are
what gets dealt out. That is the correct trade.
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_JSON = os.path.join(HERE, "splits.json")


def patient_of(case_id: str) -> str:
    """The patient a case belongs to.

    Two naming schemes are present. Most ids are '<patient>_I<series>', and the prefix
    repeats across scans of one patient: 21_I0000006, 21_I0000008, 21_I0000028. The rest
    are DICOM UIDs with no visible patient field, so each is treated as its own patient.
    That is the conservative reading — assuming otherwise would merge cases that may be
    unrelated, which is a worse error than leaving a group unmerged.
    """
    m = re.match(r"^(\d+)_I\d+$", case_id)
    return f"P{m.group(1)}" if m else case_id


def make_folds(case_ids: list[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    """Deal patients into folds, largest group first.

    Greedy rather than random: with a handful of three-volume patients, a random
    assignment can leave one fold noticeably larger. Placing the biggest groups first
    into whichever fold is currently smallest keeps the sizes close without needing to
    retry, and the shuffle beforehand keeps it from depending on id order.
    """
    groups: dict[str, list[str]] = {}
    for cid in sorted(case_ids):
        groups.setdefault(patient_of(cid), []).append(cid)

    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    names.sort(key=lambda p: len(groups[p]), reverse=True)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for p in names:
        smallest = min(range(n_folds), key=lambda k: len(folds[k]))
        folds[smallest].extend(groups[p])

    return [sorted(f) for f in folds]


def get_folds(case_ids: list[str] = None, n_folds: int = 5, seed: int = 42,
             splits_json: str = None) -> list[list[str]]:
    path = splits_json or SPLITS_JSON

    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if not saved.get("grouped_by_patient"):
            raise RuntimeError(
                f"{path} predates patient grouping: it was built by shuffling case ids, "
                f"so scans of one patient can sit on both sides of a fold. Delete it and "
                f"let it be rebuilt, and regenerate anything derived from it.")
        return saved["folds"]

    if case_ids is None:
        raise RuntimeError(f"{path} does not exist and no case_ids were given to build it")

    folds = make_folds(case_ids, n_folds, seed)
    n_patients = len({patient_of(c) for c in case_ids})
    with open(path, "w") as f:
        json.dump({"folds": folds, "grouped_by_patient": True,
                   "n_patients": n_patients, "seed": seed}, f, indent=2)
    print(f"Saved {n_folds} patient-grouped folds to {path}")
    print(f"  {len(case_ids)} volumes from {n_patients} patients, "
          f"{[len(f) for f in folds]} per fold")
    return folds


def fold_split(folds: list[list[str]], fold: int) -> tuple[list[str], list[str]]:
    val = folds[fold]
    train = [cid for k, f in enumerate(folds) if k != fold for cid in f]
    return train, val
