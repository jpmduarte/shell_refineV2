# How the method works

Fetal head segmentation in 3D ultrasound. A small 3D U-Net predicts the signed distance
to the head's surface rather than a mask, on a coarse grid in which every head has
been resized to the same apparent size. Two passes: the first measures the head, the
second segments it at the normalised size. Result: Dice 0.9487, HD95 3.01 mm, ASD 1.01 mm
over 5 folds, tied with nnU-Net (1000 epochs, no TTA) at 1/11 of the parameters and
~12-77x the speed. The numbers are in [results.md](../results.md).

Every reference below is `file:line` and opens the code in the editor.

Where the code lives, under `size_normalization/`:

| | |
|---|---|
| the method | `infer.py` (inference), `train.py`, `dataset.py`, `runs/` (checkpoints) |
| deployment | `deploy/segment.py`: NIfTI in, mask out; nothing about folds, labels or scores |
| evaluation | `evaluation/`: `predict.py` (a fold through `infer.py`), `evaluate.py` (the metrics), the size tests and their results |

Deployment and evaluation share only `infer.py`, so what is scored is what is deployed.
Shared building blocks (caches, SDF, augmentation, model, geometry) are at the repo root.

```
 DATA (offline, once)                       TRAINING                     DEPLOYMENT (per scan)
 ────────────────────                       ────────                     ─────────────────────
 NIfTI image + label                                                     native image
   │ common/image_cache.py         RAS, z-score,                                        │ z-score
   │                  native + x4 grid                                     ▼
   ▼                                                                     PASS 1: plain x4 network
 caches/cache/x1, x4, label                                                       │ head volume V
   │ common/sdf_targets.py      signed distance                                      ▼
   ▼                  to the surface                                     resample so the head
 caches/cache/sdf/x1, x4, x4mm                                                    is V0 (spacing ∝ V^1/3)
   │                                                                       ▼
   ├──────────────────────────────────► train plain x4 network ──┐      PASS 2: normalised network
   │                                     (pass 1)                │        (+ 8 flips TTA, + 3 seeds)
   │ size_normalization/build_cache.py     resample each                              │         │ SDF
   ▼                  head to V0 (the                            │         ▼
   │                  training median)                           │         │
 cache_norm4 ─────────────────────────► train normalised network ┤      back to native grid
                                         (pass 2)                │         │ SDF < 0
                                                                 └────►    ▼
                                                                         head mask
```

---

## 1. The image cache: `common/image_cache.py`

Every scan is read once and stored as plain numpy arrays, so training never touches NIfTI.

- **Loading pipeline**, [common/image_cache.py:24](../common/image_cache.py#L24). Each volume is reoriented to RAS
  (flips and axis swaps only, nothing interpolated, applied to image *and* label so they
  stay aligned), made contiguous (the flips leave slow negative-stride views), and the
  image is z-scored. *Why:* one fixed orientation and intensity scale for every scan,
  whatever the scanner wrote in the header.
- **Two resolutions**, [common/image_cache.py:20](../common/image_cache.py#L20): native (x1, 0.4 mm) and x4
  (1.6 mm). x4 is what the network sees; x1 is what we evaluate at. *Why x4:* a whole head
  fits in 96³ voxels, so the network sees all of it at once with no tiling. At 3.2 mm (x8)
  small heads became unstable; at 1.6 mm they did not.
- **The label is never resampled**, [common/image_cache.py:74](../common/image_cache.py#L74). *Why:* resampling a
  mask moves its surface; all metrics are computed against this original label.

## 2. The target, a signed distance field: `common/sdf_targets.py`

- **`signed_distance_mm`**, [common/sdf_targets.py:25](../common/sdf_targets.py#L25): for every voxel, the
  exact Euclidean distance in mm to the surface, positive outside and negative inside,
  computed on the native label with the case's own spacing.
- **Truncated to ±10 mm and scaled to [-1, 1]**, [common/sdf_targets.py:33](../common/sdf_targets.py#L33).
- *Why an SDF instead of a mask:* a mask says only inside/outside; on a 1.6 mm grid that
  quantises the surface to whole voxels. A distance field varies smoothly, so the zero
  crossing (the surface) can fall *between* voxels, and interpolating it back to 0.4 mm
  keeps that sub-voxel position. *Why truncate:* far from the head the exact distance says
  nothing about where the surface is, and would dominate the loss.
- **The untruncated mm version**, [common/sdf_targets.py:86](../common/sdf_targets.py#L86) (`sdf/x4mm`).
  *Why:* the augmentation zooms volumes, and zooming by k multiplies every distance by k.
  A field already clipped at 10 mm cannot be rescaled correctly, so the augmentation works
  on the mm field and truncates afterwards ([common/augmentation.py:162](../common/augmentation.py#L162)).

## 3. The folds: `common/folds.py`

- **Grouped by patient**, [common/folds.py:19](../common/folds.py#L19) and
  [common/folds.py:25](../common/folds.py#L25). 90 volumes come from 82 patients; scans of the same
  fetus are always in the same fold. *Why:* two scans of one fetus are nearly the same
  image, so one in training and one in validation would inflate the score.
- Saved once in `splits.json` and reused by every run, so all numbers compare like for
  like. nnU-Net was trained on the same folds.

## 4. Size normalisation: `size_normalization/build_cache.py`

The idea of the method. Head volumes range from 35 to 291 mL. On a fixed grid the network
has to cope with heads of very different sizes, and the smallest are rare.

- **Each case gets its own spacing**: `scale = (V / V0)^(1/3)`, spacing = 4 × native ×
  scale. With V0 ≈ 130 mL, a 35 mL head is sampled at ~1.0 mm and a 291 mL head at
  ~2.1 mm, and both span about the same number of voxels. *Why:* the network only ever
  sees one head size, and a small head gets *finer* voxels rather than fewer of them.
- **V0, the canonical size, comes from the data**: the median head volume of the training
  labels, computed when the training cache is built and stored in its `norm.json`
  (`size_normalization/build_cache.py`; `--v0-ml` overrides it). Inference (evaluation and deployment) reads it
  from there with `--reference`, never from a constant. *Why:* on another dataset the reference size adapts by itself, and
  training and inference cannot disagree on it. The caches in this repo were built with
  130 mL (this dataset's median, 128.3, rounded).
- **V0 is a dataset-level choice**, taken from all 90 labels (validation folds
  included), in the same way nnU-Net computes its plans (target spacing, patch size,
  normalisation) from the fingerprint of every training case. Per fold, from training
  labels only, it would have been 127.0–130.2 mL: at most 1% in linear scale, for a
  canonical size that any fixed value serves as long as training and testing share it.
  Deliberately kept, so the comparison with nnU-Net is like for like.
- **The image and both SDF targets are resampled to that grid**,
  [size_normalization/build_cache.py:78](../size_normalization/build_cache.py#L78) and saved with their own affine
  ([size_normalization/build_cache.py:122](../size_normalization/build_cache.py#L122)). The affine records the spacing, so a
  prediction can always be carried back to native.
- **Links instead of copies**, [size_normalization/build_cache.py:36](../size_normalization/build_cache.py#L36): `x1`, `label` and
  `sdf/x1` point to `caches/cache/`, so prediction and evaluation read the new cache unchanged.
- **The 96³ cap**, [size_normalization/build_cache.py:81](../size_normalization/build_cache.py#L81). Training batches need every
  scan to fit 96 voxels per axis, and enlarging a small head enlarges its whole scan. A
  case that would not fit is coarsened just enough. It happened once in 90, by 1%.
  Deployment does not cap (section 8).
- **Which volume:** the training cache uses the true volume from the label (`cache_norm4`).
  Evaluation and deployment never use a normalised cache: they run the whole method from
  the native image (section 8), with the volume pass 1 estimates. The estimate costs
  ~0.0006 Dice against the true volume (results.md).

## 5. The network: `common/model.py`

- **3D U-Net**, [common/model.py:37](../common/model.py#L37): three pooling levels
  ([common/model.py:14](../common/model.py#L14)), 16 → 32 → 64 channels, a 128-channel bottleneck
  ([common/model.py:50](../common/model.py#L50)), skip connections, 1.40 M parameters.
- **Building block**, [common/model.py:18](../common/model.py#L18): two 3×3×3 convolutions with batch
  norm and ReLU. nnU-Net's instance norm and leaky ReLU were tried and were worse here.
- **Output**: a 1×1×1 convolution and `tanh` ([common/model.py:60](../common/model.py#L60)), matching
  the SDF's [-1, 1] range.
- *Why this small:* at x4 with normalised heads the task is narrow. A fourth level
  (5.6 M parameters) gave nothing. The same architecture serves both passes.

## 6. Training: `size_normalization/train.py`, `size_normalization/dataset.py`, `common/augmentation.py`, `common/loss.py`

- **Data**, [dataset.py:22](../size_normalization/dataset.py#L22): each item is the x4 image and
  its SDF, padded at the far end of each axis to 96³ ([common/augmentation.py:28](../common/augmentation.py#L28)).
  Image padding is 0 (background after z-scoring); SDF padding is +1 (outside), because 0
  would create a fake surface.
- **Augmentation: nnU-Net's, adapted to an SDF**, in the shared root module so every
  approach can use it: [common/augmentation.py:150](../common/augmentation.py#L150).
  Mirroring ([common/augmentation.py:69](../common/augmentation.py#L69)), rotation and scaling
  ([common/augmentation.py:76](../common/augmentation.py#L76)), then noise, blur, brightness, contrast,
  low resolution and gamma ([common/augmentation.py:130](../common/augmentation.py#L130)): the same
  transforms, probabilities and ranges as nnU-Net v2. Two adaptations: the SDF is
  interpolated linearly and divided by the zoom factor, because distances scale with
  the image. *Why nnU-Net's:* it is the reference we compare against, and adopting its
  augmentation was a clear gain over our earlier simple one.
- **Loss**, [train.py:55](../size_normalization/train.py#L55): L1 on the SDF + 0.5 × a
  **surface-weighted L1** ([common/loss.py:10](../common/loss.py#L10)), whose weight is
  `exp(-distance / 1.5 mm)` ([common/loss.py:18](../common/loss.py#L18)). *Why:* most voxels are far from
  the surface, where the target is a flat ±1; plain L1 spends its gradient there. The
  weighting concentrates it on the boundary. It was the largest single gain (+0.0030 Dice).
- **Optimiser**: Adam, lr 5e-4, cosine decay to 0 over 500 epochs
  ([train.py:194](../size_normalization/train.py#L194)), batch 4.
- **Checkpoint: the last epoch**, [train.py:226](../size_normalization/train.py#L226). Validation is
  logged for curves only. *Why:* picking the best epoch on the validation fold is picking on
  the test data; that inflated Dice by 0.0024 before we stopped. Choosing it without the test data, on an
  inner validation set held out of training (`--inner-val 5`), is no better than the last
  epoch (−0.0004, p = 0.34; results.md).
- **Seeds**, [train.py:170](../size_normalization/train.py#L170): a seed fixes weights, batch order,
  augmentation and GPU kernels; a seeded run reproduces bit for bit. Every result is the
  per-case mean of 3 seeds (42, 43, 44), because one seed's noise (~0.002 Dice) is as large
  as most of the effects we measure.
- **Two networks are trained per fold**: the plain x4 network on `cache` (pass 1) and the
  normalised network on `cache_norm4` (pass 2). Same code; only `--cache-dir` differs.

## 7. Prediction and evaluation: `size_normalization/evaluation/predict.py`, `size_normalization/evaluation/evaluate.py`, `common/metrics.py`

- **Predict**, [predict.py](../size_normalization/evaluation/predict.py): runs the method of
  section 8 on each cached native volume and saves the native SDF. `--pass1` makes it the
  size-normalised method; several `--checkpoint` form the ensemble; `--tta` adds the 8
  mirrorings; `--oracle` uses the label's volume. The SDF is resampled to native, not a
  mask, so the surface keeps its sub-voxel position. Outside the scan the SDF is +1.
- **Evaluate**, [evaluate.py:24](../size_normalization/evaluation/evaluate.py#L24): mask = SDF < 0, scored
  against the original label: Dice, HD95 and ASD in mm ([common/metrics.py:30](../common/metrics.py#L30),
  [common/metrics.py:55](../common/metrics.py#L55), [common/metrics.py:64](../common/metrics.py#L64)). nnU-Net's
  predictions were scored by the same functions against the same labels.

## 8. The method at inference: `size_normalization/infer.py`

[`segment`](../size_normalization/infer.py) is the method on one scan, entirely on
the GPU, native image in, native SDF out. Everything that runs the networks calls it:
`evaluation/predict.py` (evaluation), `deploy/segment.py` (a new NIfTI), `scripts/benchmark_inference.py`
(timing) and the size tests, so what is scored is what is deployed.

1. **z-score** the native volume.
2. **Pass 1**: resample to x4, run the plain x4 network, and measure the head as
   voxels with SDF < 0 × voxel volume.
3. **Guard**: the scale k = (V / V0)^(1/3) is used only within 0.5–2 (a head of about
   16–1040 mL). Outside it, including a pass 1 that finds no head, the case keeps pass 1's
   own segmentation. On the 90 cases every k lies in 0.65–1.27.
4. **Normalise**: resample the native volume at the spacing that makes this head V0, read
   from the training cache's `norm.json` (`--reference`, default `cache_norm4`).
5. **Pass 2**: the normalised network. Optional TTA: the 8 mirror images of the input,
   each prediction mirrored back and averaged (`forward`). Optional ensemble: the SDFs
   of 3 seeds averaged. Averaging SDFs (not masks) keeps the result a smooth field.
6. **Back to native** and threshold at 0. Optional blend (`--blend 0.25`): pass 1's SDF,
   also carried to native, is mixed in as 0.25 × pass 1 + 0.75 × pass 2. Pass 1 sees the
   head at its real size and pass 2 at the canonical size, so their errors partly cancel.
   On 90 cases: +0.0010 Dice without TTA for ~6 ms, +0.0004 with TTA (where pass 1 then
   needs its 8 flips too, ~+95 ms).

*Why two passes:* normalisation needs the head volume, which is unknown on a new scan;
pass 1 estimates it (median error 4.3% of the volume, 1.4% of the scale k). The estimate is
good enough: normalising with the true volume instead gains only ~0.0006 Dice.

**A new scan**, [deploy/segment.py](../size_normalization/deploy/segment.py): NIfTI in, mask out.
The volume is reoriented to RAS for the networks (as `common/image_cache.py` does for training), and the
mask is reoriented back and written in the input's own orientation, on its grid, with its
affine and header. Needs isotropic voxels.

    python size_normalization/deploy/segment.py scan.nii.gz mask.nii.gz [--tta] [--sdf sdf.nii.gz]

**Scans larger than the training cube** (`pad`):
each axis is padded to 96 or to the next multiple of 8, whichever is larger. The network is
fully convolutional and only needs sizes divisible by 8 (three poolings), so a bigger scan
is processed whole, never cropped and never coarsened. Coarsening to fit cost 0.028 Dice
in a test.

Cost on an RTX 5070, per case: 35 ms (one model), 115 ms (+TTA), 300 ms (3 seeds + TTA),
0.7 GB of GPU memory. nnU-Net: 1.35 s, or 8.9 s with TTA.

## 9. Resampling: `common/geometry.py`

Every grid carries an affine (voxel index → mm). Resampling = for each output voxel, find
its position in the input through the two affines, and interpolate.

- **`downsample`**: MONAI's `Spacingd`, used to build the caches.
- **`resample_grid` + `apply_resample`**: the same map as an explicit per-voxel grid
  (`infer.py --resampler geometry`, kept for comparison).
- **`spacing_grid` + `affine_resample`**: used at inference (`infer.py`). Between our
  grids the map is only a scale and a shift per axis, so PyTorch's `affine_grid` builds the
  sampling grid from one 3×4 matrix on the GPU. Same voxels (checked to 7e-5 against the
  caches), ~70 ms less per case.
- `FACTOR` (4) and `CUBE` (96), the network's grid, are defined here once; `TRUNC_MM`
  (10 mm), the SDF's truncation, in `common/sdf_targets.py`.
- *Why through the affines and not a plain resize:* a resize by voxel count stretches each
  axis by its own ratio, which distorts the anatomy; the affine keeps millimetres
  millimetres.

## 10. Measuring cost: `scripts/`

| Script | What it measures |
|---|---|
| `benchmark_inference.py` | our inference (`infer.py`): time per stage, GPU memory, RAM, CPU, Dice check |
| `profile_run.py` | any command (e.g. training): GPU memory, RAM, CPU and GPU use over time |
| `probe_vram.py` | one training step's peak GPU memory, for the job queue |

Results of the last measurement are in `docs/profiling/`.

## 11. Running everything from scratch

```
python common/image_cache.py --images-dir <images> --labels-dir <labels>    # caches/cache/x1, x4, label
python common/sdf_targets.py                                             # caches/cache/sdf/x1, x4
python common/sdf_targets.py --untruncated                               # caches/cache/sdf/x4mm
python size_normalization/build_cache.py                                            # cache_norm4 (true volumes; V0 = median)

# per fold k and seed s:
python size_normalization/train.py --fold k --seed s --cache-dir caches/cache --run-dir size_normalization/runs/x4_s<s>/fold<k>
python size_normalization/train.py --fold k --seed s --run-dir size_normalization/runs/x4norm_s<s>/fold<k>

# evaluation, the method as deployed (pass 1's volume, from the native image):
R=size_normalization/runs
python size_normalization/evaluation/predict.py --fold k --out-dir <out>/preds \
    --pass1 $R/x4_s<s>/fold<k>/checkpoints/phase1_last.pth \
    --checkpoint $R/x4norm_s<s>/fold<k>/checkpoints/phase1_last.pth
#   + --tta for TTA; three --checkpoint (seeds 42 43 44) for the ensemble; --oracle for the
#   label's volume; no --pass1 and the x4 checkpoint for the plain x4 method
python size_normalization/evaluation/evaluate.py --pred-dir <out>/preds --out-dir <out>/eval

# the checks (seeded training, a stored score, the guard, segment.py, a rebuilt cache):
python -m unittest discover tests -v
```
