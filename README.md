# shell_refineV2: fetal head segmentation in 3D ultrasound

A small 3D U-Net (1.40 M parameters) regresses the signed distance to the fetal head's
surface on a 1.6 mm grid. The method runs in two passes: the first estimates the head
volume, the volume is resampled so every head has the training median size (130 mL),
and a second network segments it there. Results on 90 volumes (5-fold cross-validation
grouped by patient) are in [results.md](results.md); how each step works and why is in
[docs/pipeline.md](docs/pipeline.md).

## Layout

| Folder | What it holds |
|---|---|
| `datasets/` | One folder per dataset (`fetal_head/`, `cetus/`): where its caches and runs live, how its cases group into patients, its fixed fold assignment (`splits.json`), and its converter to NIfTI if it needs one |
| `common/` | Building blocks shared by both methods: the image and SDF caches, the folds, geometry (resampling), the U-Net, losses, metrics, the training augmentation, seeding |
| `size_normalization/` | The main method: its training cache, training, inference (`infer.py`), deployment on a new scan (`deploy/`), and every evaluation with its results (`evaluation/`) |
| `boundary_refiner/` | A second version: a refiner at native resolution on top of a first network's prediction. Tested and not kept (no gain at 10–30× the cost) |
| `scripts/` | Inference and SDF benchmarks, GPU memory probe, resource profiler, the local job queue |
| `tests/` | Checks that the code still reproduces the reported numbers |
| `docs/` | `pipeline.md` (the method in detail), `final_architecture.md` (the planned extensions), `profiling/` (benchmark outputs), `archive/` (documents from before the folds were grouped by patient) |

Data are not in the repository. The fetal head keeps its original locations, `caches/`
(built from the NIfTI volumes) and `size_normalization/runs/`; every other dataset uses
`data/<dataset>/` and `runs/<dataset>/`. Each dataset's `splits.json` is tracked so every run
uses the same folds. All these paths are defined once, in `datasets/`.

## Running

Python with PyTorch, MONAI, nibabel, SciPy, NumPy and psutil (the conda environment
`fetal-head`). The three folders of code are one installable package, so `from common...`
works from anywhere; install it once, editable, so changes to the code apply right away:

```
pip install -e . --no-deps
```

The commands below run from the repository root, on the fetal head. Every script takes
`--dataset <name>` (default `fetal_head`) and then reads and writes that dataset's caches,
folds and runs. Each script's `--help` shows its options and examples.

```
# caches, once
python common/image_cache.py --images-dir <images> --labels-dir <labels>   # caches/cache: x1, x4, label
python common/sdf_targets.py                                               # caches/cache/sdf: x1, x4
python common/sdf_targets.py --untruncated                                 # caches/cache/sdf/x4mm
python size_normalization/build_cache.py                                   # caches/cache_norm4

# training, per fold k and seed s: pass 1 (plain x4), then pass 2 (size-normalised)
python size_normalization/train.py --fold k --seed s --plain --run-dir size_normalization/runs/x4_s<s>/fold<k>
python size_normalization/train.py --fold k --seed s --run-dir size_normalization/runs/x4norm_s<s>/fold<k>

# evaluation of a fold, the method as deployed
python size_normalization/evaluation/predict.py --fold k --out-dir <out>/preds \
    --pass1 size_normalization/runs/x4_s<s>/fold<k>/checkpoints/phase1_last.pth \
    --checkpoint size_normalization/runs/x4norm_s<s>/fold<k>/checkpoints/phase1_last.pth
python size_normalization/evaluation/evaluate.py --pred-dir <out>/preds --out-dir <out>/eval

# a new scan: NIfTI in, mask out
python size_normalization/deploy/segment.py scan.nii.gz mask.nii.gz [--tta] [--sdf sdf.nii.gz]

# checks
python -m unittest discover tests -v
```

`predict.py` also takes `--tta` (8 mirrorings), several `--checkpoint` (the ensemble) and
`--oracle` (the label's volume instead of pass 1's estimate). The evaluations behind each
claim, with their commands, are in `size_normalization/evaluation/*.md`.

## Adding a dataset

1. Create `datasets/<name>/__init__.py` with `DATASET = Dataset.standard("<name>", patient_of)`,
   where `patient_of` maps a case id to its patient (scans of one patient share a fold).
   Pass `k_range=` or the run names if they differ from the defaults.
2. Register it in `datasets/__init__.py`.
3. If the data are not NIfTI image and label pairs, add a converter that writes them, one
   isotropic volume and one binary mask per case, named by case id.
4. Build the caches and train with `--dataset <name>`. The first run writes the dataset's
   `splits.json`; commit it so later runs keep the same folds.
