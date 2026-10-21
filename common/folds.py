# The 5-fold cross-validation split, grouped by patient and saved to splits.json so that
# every run uses the same folds.
#
#     splits.json: {"folds": [[case_id, ...] x 5], "grouped_by_patient": true}
#
# Fold k validates on folds[k] and trains on the other four.
#
# Why group by patient: the 90 volumes come from 82 patients, so some fetuses were scanned
# more than once. Two scans of the same fetus from one session are far more alike than two
# different fetuses; with one in training and the other in validation, the model is tested
# on something it has effectively already seen. The folds come out slightly uneven as a
# result, because whole patients are dealt out rather than single volumes.

import json
import os
import re

import numpy as np

from common.paths import SPLITS_JSON


# The patient a case belongs to.
# Most ids look like "<patient>_I<series>" (21_I0000006, 21_I0000008: same patient).
# The rest are DICOM UIDs with no patient field, so each counts as its own patient:
# merging cases that may be unrelated would be the worse mistake.
def patient_of(case_id: str) -> str:
    m = re.match(r"^(\d+)_I\d+$", case_id)
    return f"P{m.group(1)}" if m else case_id


# Deal whole patients into folds, largest patient first, each into whichever fold is
# currently smallest. That keeps the fold sizes close without retrying; the shuffle
# beforehand stops the result depending on the id order.
def make_folds(case_ids: list[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    # patient -> their cases
    groups: dict[str, list[str]] = {}
    for cid in sorted(case_ids):
        groups.setdefault(patient_of(cid), []).append(cid)

    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    # Stable sort: patients with the same number of scans keep their shuffled order.
    names.sort(key=lambda p: len(groups[p]), reverse=True)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for p in names:
        smallest = min(range(n_folds), key=lambda k: len(folds[k]))
        folds[smallest].extend(groups[p])

    return [sorted(f) for f in folds]


# The folds: read from the saved file if it exists, built and saved otherwise.
def get_folds(case_ids: list[str] = None, n_folds: int = 5, seed: int = 42,
              splits_json: str = None) -> list[list[str]]:
    path = splits_json or SPLITS_JSON

    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        # An old split built by shuffling case ids would leak patients across folds.
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


# (training cases, validation cases) for one fold.
def fold_split(folds: list[list[str]], fold: int) -> tuple[list[str], list[str]]:
    val = folds[fold]
    train = [cid for k, f in enumerate(folds) if k != fold for cid in f]
    return train, val
