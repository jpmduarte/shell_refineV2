"""Where the repository keeps its data, so no script works out these paths on its own."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_JSON = os.path.join(ROOT, "splits.json")
NATIVE_CACHE = os.path.join(ROOT, "caches", "cache")        # native and x4 images, labels, SDFs
NORM_CACHE = os.path.join(ROOT, "caches", "cache_norm4")    # the size-normalised training cache
SIZE_NORM_DIR = os.path.join(ROOT, "size_normalization")
SIZE_NORM_RUNS = os.path.join(SIZE_NORM_DIR, "runs")
REFINER_DIR = os.path.join(ROOT, "boundary_refiner")

PASS1_RUN = "nnaug_x4e500"   # the plain x4 network: pass 1, and the plain method
PASS2_RUN = "nnaug_x4norm"   # the size-normalised network: pass 2
LAST_CHECKPOINT = "phase1_last.pth"


def run_checkpoint(run: str, seed: int, fold: int) -> str:
    """The reported checkpoint of one seed and fold of a size-normalisation run."""
    return os.path.join(SIZE_NORM_RUNS, f"{run}_s{seed}", f"fold{fold}", "checkpoints", LAST_CHECKPOINT)
