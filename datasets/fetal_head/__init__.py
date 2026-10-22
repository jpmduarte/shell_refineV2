"""Fetal head in 3D ultrasound: 90 volumes of 82 patients, isotropic 0.4 mm.

Its caches and runs keep the locations they had before the repository held several datasets.
"""
import os
import re

from common.paths import ROOT
from datasets.base import Dataset

PATIENT_SERIES_ID = re.compile(r"^(\d+)_I\d+$")   # "21_I0000006" and "21_I0000008": patient 21


def patient_of(case_id: str) -> str:
    """A DICOM UID id has no patient field, so it counts as its own patient."""
    match = PATIENT_SERIES_ID.match(case_id)
    return f"P{match.group(1)}" if match else case_id


DATASET = Dataset(
    name="fetal_head",
    cache=os.path.join(ROOT, "caches", "cache"),
    norm_cache=os.path.join(ROOT, "caches", "cache_norm4"),
    splits_json=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json"),
    runs=os.path.join(ROOT, "size_normalization", "runs"),
    patient_of=patient_of,
    pass1_run="nnaug_x4e500",
    pass2_run="nnaug_x4norm",
)
