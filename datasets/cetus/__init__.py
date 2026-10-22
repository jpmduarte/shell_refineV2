"""CETUS (MICCAI 2014): the left ventricle in 3D echocardiography, 45 patients at end-diastole and end-systole."""
from datasets.base import Dataset


def patient_of(case_id: str) -> str:
    """patient01_ED and patient01_ES are the same patient."""
    return case_id.rsplit("_", 1)[0]


DATASET = Dataset.standard("cetus", patient_of)
