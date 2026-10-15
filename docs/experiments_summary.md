# Every experiment, in one place

Two protocols, **not comparable with each other**:

- **Current** (from 28 Sep 2026): patient-grouped 5-fold CV, last epoch, per-case mean of
  3 seeds, 90 cases, paired Wilcoxon. Details in [results.md](../results.md).
- **Early** (before): folds that leaked patients, best epoch picked on the validation
  fold (+0.0024 optimistic), mostly one seed and often 1–2 folds. Details in
  [results.md](results.md) and [experiments.md](experiments.md).

Dice is on the native 0.4 mm label. "Same folds" means the arm ran on fewer folds and is
compared with the reference on those folds only.

## Current protocol

| Experiment | Kind | Result (Dice) | Verdict |
|---|---|---|---|
| x8 SDF network (3.2 mm), nnU-Net augmentation | resolution | 0.9412 | superseded by x4 |
| x8 + antialiased downsampling | preprocessing | 0.9425 | superseded by x4 |
| x8 + dual head (SDF + mask, Dice+BCE) | multi-task head | 0.9426 | small gain at x8; never tried at x4 |
| x8 + 4th level | architecture | 0.9422 vs 0.9417 (fold 3) | no gain |
| x8 + size normalisation | normalisation | 0.9470 vs 0.9417 (fold 3) | large gain, led to x4 + norm |
| **x4 SDF network** (1.6 mm) | resolution | **0.9447** | kept, as pass 1 |
| x4 + 4th level (5.65 M params) | architecture | 0.9445 | no gain |
| x4, 1000 epochs instead of 500 | training length | not better (seed 42) | kept 500 |
| x4 + joint 1D point head | multi-task head | 0.9454 (+0.0003, p = 0.84, seed-matched) | no robust gain |
| x4 + joint 2D head | multi-task head | 0.9459 (1 seed, 3 folds) | dropped |
| **x4 + size normalisation** (V0 = 130 mL) | normalisation | **0.9462** (+0.0015, p = 0.055; HD95 −0.20 mm) | **kept, main method** |
| x4 + pose + size canonical frame | normalisation | 0.9465 vs 0.9480 size-only (fold 3) | worse than size-only; on standby |
| + mirroring TTA (8 flips) | inference | x4 +0.0025 (p = 1e-8); norm 0.9480 | kept |
| + ensemble of 3 seeds | inference | x4 +0.0021; **norm + TTA 0.9487** | kept, optional |
| + blend pass 1's SDF (0.25) | inference | +0.0010 without TTA, +0.0004 with | optional, cheap without TTA |
| train on estimated vs true volumes (60/5/25 held-out) | protocol | −0.0008 (p = 0.33), HD95 +0.16 | train on true volumes |
| refiner: cascade at x2 (`phase2_v2`) | phase 2 | no gain on the x4 prior | dropped |
| refiner: 1D surface profiles (+ context) | phase 2 | no gain | dropped |
| refiner: native band, 64³ cubes, on ens+TTA priors | phase 2 | ±0.0000 to −0.0004 (seed 42, 2 folds); 2–10 s per case | dropped: no gain at 10–30× cost |
| x4 norm + Gaussian contour head + consistency | multi-task head | −0.0002 (p = 0.81), 3 seeds | no gain |
| x4 norm + soft-mask Dice + BCE on the SDF | loss | −0.0005 (p = 0.30); ens + TTA −0.0003 | no gain |
| x4 norm + 2× fine input (8 channels, space-to-depth) | input | −0.0003 (p = 0.90); ens + TTA −0.0002 | no gain |
| resolution × loss, size-normalised, seed 42 (vs x4 SDF 0.9466) | resolution / loss | x8 SDF 0.9442, x8 mask 0.9423, x4 mask 0.9453 (p = 0.057; ties with TTA, 0.9480 vs 0.9477); x2 fold 0 only: SDF 0.9371, mask 0.9359 vs 0.9437 | x4 SDF kept; x2 stopped (27 GB to train at batch 4, 3.3–6.7× slower inference) |
| representation ceiling (perfect coarse target → native) | analysis | SDF x8 / x4 / x2: 0.9961 / 0.9979 / 0.9995; mask 0.9755 / 0.9874 / 0.9932 | the coarse grid is not the limit; the plain-x4 0.9967 is an artefact of the integer factor 4 (4.04 gives 0.9980) |
| x4 norm, 32 base channels (5.6M params) | architecture | 0.9433 vs 0.9437 (seed 42, fold 0 only) | cancelled: no sign of gain, ~2× training memory |
| nnU-Net 3d_fullres, 250 / 1000 epochs | reference | 0.9484 / 0.9497; + TTA 0.9496 / **0.9510** | our best: −0.0010 vs 1000 (n.s.), −0.0022 vs 1000 + TTA |

## Early protocol

**Phase 1 (x8 SDF network)**

| Experiment | Kind | Result (Dice) | Verdict |
|---|---|---|---|
| plain L1 on the SDF | loss | 0.9417 | baseline |
| **L1 + 0.5 × surface-weighted L1** (σ 1.5 mm) | loss | **0.9447** (+0.0030, 5 folds) | **kept** (the largest single gain) |
| surface weight 1.0 / 2.0, σ 0.5 mm, weighted-only | loss | 0.9431 / 0.9396 / 0.9290 / 0.9423 | worse |
| log-cosh instead of L1 | loss | 0.9435 (1 fold) | no |
| gradient losses (slope L2, direction cosine) | loss | level with sw05 | no |
| learned attention gates on skips | architecture | 0.9414 vs 0.9439 (2 folds) | worse |
| SDF-driven gates (exp(−\|sdf\|/σ)) | architecture | 0.9427 vs 0.9447 (5 folds); looked best on 2 | worse |
| InstanceNorm + LeakyReLU (nnU-Net's) | architecture | −0.0018 to −0.0024 | worse |
| mirroring TTA, 4 / 8 flips | inference | +0.0002 / −0.0023 | not then (helps now, with nnU-Net augmentation) |
| SWA weight averaging | training | compared against an inflated baseline | void |
| shifting the 0 threshold | inference | optimum exactly at 0 | no |
| largest component + hole filling (nnU-Net post-processing) | post-processing | +0.00001 (the SDF gives one closed region) | no |

**Phase 2 refiner** (gain over the prior; fold 0 at 0.9417 unless noted)

| Experiment | Kind | Result (Δ Dice) | Verdict |
|---|---|---|---|
| cubes 64³ / 96³ / 128³ | patch shape | +0.0018 / +0.0005 / +0.0009 | 64³ best cube |
| band-masked loss (score only the band) | loss | +0.0021 vs +0.0010 unmasked | kept |
| slabs 128²×16 / 32 / **48** / 64 | patch shape | +0.0016 / +0.0034 / **+0.0039** / +0.0016 | 48 best |
| band 3 / 5 / 7.5 / 10 mm | band | +0.0021 / +0.0021 / +0.0014 / −0.0004 | 5 mm kept |
| Gaussian tile blending | inference | +0.0023 (5 folds) | kept |
| surface-weighted BCE (swwbce) | loss | 0.9448 on folds 1, 4 | kept |
| residual SDF correction instead of classes | output | −0.0001 to +0.0007 | no |
| polar / spherical refiner | representation | +0.0033, best HD95 / ASD of any arm | dropped (resampling cost, star-convexity) |
| polar width, context size, FiLM conditioning | architecture | null | no |
| 4-level refiner (5.6 M params) | architecture | +0.0015 vs +0.0023 | no |
| weight decay + dropout + strong augmentation | regularisation | +0.0005 vs +0.0007 | no (reduced overfitting, no Dice) |
| uncertainty / hybrid loss weighting | loss | level with swwbce | no |
| sampling biased to hard patches | sampling | level | no |
| joint training of both phases | training | degraded phase 1 (0.9417 → 0.9319) | no |
| sparse convolution on the band | efficiency | exact, but 1.2× slower than dense | parked |
| adaptive slab shape | patch shape | not built: the band is locally isotropic | no |
| third phase on uncertain voxels | phase 3 | not built: ceiling analysis gave +0.001–0.002 | no |
| iterating phase 2 on its own output | inference | only 0.4% of the band changes | no |
| best pipeline: sw05 + swwbce + blend | full method | 0.9454 (best-on-val) | superseded |

## What the two eras taught

- **What moved the result:** the surface-weighted loss, x4 resolution, size normalisation,
  TTA and seed ensembles.
- **Never helped, whatever the form:** capacity (a 4th pooling level; 32 base channels on
  one fold), attention or gates, InstanceNorm, regularisation, gradient losses, boundary
  losses (contour head, soft mask), finer input, a finer grid (x2), post-processing.
- **SDF against a mask target:** the SDF wins without TTA (x8 +0.0019, x4 +0.0013) and
  loses 4–6× less at the representation ceiling; with TTA the x4 mask ties (seed 42).
- **The refiner's gain shrinks as the prior improves.** It reaches zero at a prior Dice of
  ~0.946, and our priors are now 0.946–0.949.
