"""The datasets the pipeline runs on; every script picks one with --dataset (fetal_head by default)."""
from datasets.base import Dataset
from datasets.cetus import DATASET as CETUS
from datasets.fetal_head import DATASET as FETAL_HEAD

DATASETS = {d.name: d for d in (FETAL_HEAD, CETUS)}
DEFAULT = FETAL_HEAD.name


def get(name: str) -> Dataset:
    return DATASETS[name]


def add_dataset_argument(parser):
    parser.add_argument("--dataset", default=DEFAULT, choices=sorted(DATASETS),
                        help="which dataset's caches, folds and runs to use (default: %(default)s)")
