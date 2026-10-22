"""What the pipeline needs to know about a dataset: where its data and runs live, and the few choices that differ."""
import os
from dataclasses import dataclass
from typing import Callable

from common.paths import DATA_ROOT, ROOT, RUNS_ROOT

LAST_CHECKPOINT = "phase1_last.pth"


@dataclass(frozen=True)
class Dataset:
    name: str
    cache: str                          # native and x4 images, labels, SDFs (common/image_cache.py layout)
    norm_cache: str                     # the size-normalised training cache (size_normalization/build_cache.py)
    splits_json: str                    # the fixed fold assignment, tracked in git
    runs: str                           # <runs>/<run>_s<seed>/fold<k>/
    patient_of: Callable[[str], str]    # case id -> patient id, so scans of one patient share a fold
    pass1_run: str = "x4"               # the plain x4 network: pass 1, and the plain method
    pass2_run: str = "x4norm"           # the size-normalised network: pass 2
    k_range: tuple[float, float] = (0.5, 2.0)   # accepted scale factors; outside, pass 1's result is kept

    @classmethod
    def standard(cls, name: str, patient_of: Callable[[str], str], **choices) -> "Dataset":
        """A dataset in the default layout: data/<name>/, runs/<name>/, datasets/<name>/splits.json."""
        data = os.path.join(DATA_ROOT, name)
        return cls(name=name, cache=os.path.join(data, "cache"), norm_cache=os.path.join(data, "cache_norm4"),
                   splits_json=os.path.join(ROOT, "datasets", name, "splits.json"),
                   runs=os.path.join(RUNS_ROOT, name), patient_of=patient_of, **choices)

    def checkpoint(self, run: str, seed: int, fold: int) -> str:
        """The reported checkpoint of one seed and fold of a run."""
        return os.path.join(self.runs, f"{run}_s{seed}", f"fold{fold}", "checkpoints", LAST_CHECKPOINT)
