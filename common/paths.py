"""The repository's top-level folders; each dataset's own paths are in datasets/."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")   # data/<dataset>/ for every dataset but fetal_head
RUNS_ROOT = os.path.join(ROOT, "runs")   # runs/<dataset>/ likewise
SIZE_NORM_DIR = os.path.join(ROOT, "size_normalization")
REFINER_DIR = os.path.join(ROOT, "boundary_refiner")
