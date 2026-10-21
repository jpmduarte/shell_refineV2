# Results

Everything since the folds were regrouped by patient (commit `5f3bc7d`). The older numbers in
`docs/results.md` used folds that leaked patients and are not comparable with these.

## Protocol

- 90 volumes, 5-fold cross-validation grouped by patient (`splits.json`); nnU-Net trained on
  the same folds (`nnunet/runs/grouped_*`).
- Dice, HD95 (mm), ASD (mm) at native resolution against the label.
- Our numbers are the last epoch (nothing selected on the reported fold), and the
  **per-case mean over 3 seeds** (42, 43, 44). A method is compared on the same 90 cases with
  a paired Wilcoxon test; one seed is never enough to conclude (seed-to-seed spread is
  ~0.002 Dice).
- The last epoch is not worse than the best one. Size-normalised x4, 3 seeds × 5 folds,
  each trained with 1/5 of its training cases held out by patient (`train.py --inner-val 5`)
  to choose the epoch with the best Dice; both scored on the outer fold: best 0.9426 / 3.53 /
  1.127, last 0.9430 / 3.51 / 1.118 (Dice −0.0004, p = 0.34, best better on 43/90). The
  chosen epoch ranged from 245 to 415, and the gap per run from −0.0069 to +0.0021
  (`size_normalization_approach/runs/bestepoch_x4norm_s*`).
- Evaluation runs the method as deployed (`size_normalization_approach/infer.py`, through
  `predict.py`): from the native image, with pass 1's volume. The size-normalised numbers
  below were first computed from estimated-volume caches; re-run through `infer.py`, all
  90 cases × 3 seeds give the same means (one model: Dice 0.9462, HD95 3.17, ASD 1.059
  against 1.060). Per case they differ by up to 0.012 Dice, because the caches measured
  pass 1's volume slightly differently (median 0.1%) and shrank one scan to fit 96³.
- nnU-Net: `3d_fullres`, final checkpoint, 250 and 1000 epochs.

## The method

1. **Pass 1**: a 3D UNet (1.40 M params) regresses a truncated signed distance (10 mm) on a
   1.6 mm grid (x4) in one 96³ pass. The head volume comes from its zero level set.
2. **Size normalisation**: the image is resampled so the head has the dataset's median
   size V0, the training median (130 mL here): spacing = 4 × native × (V / V0)^(1/3).
3. **Pass 2**: a second network of the same size, trained on normalised heads (with the true
   volume), predicts the SDF. Optionally with mirroring TTA (8 flips) and/or the 3-seed
   ensemble, averaged as SDFs.
4. The SDF goes back to the native grid through the affine; mask = SDF < 0.

Loss: L1 + 0.5 × surface-weighted L1 (σ 1.5 mm). nnU-Net-style augmentation adapted to the
SDF. 500 epochs, lr 5e-4, batch 4.

## Main result (5 folds, 90 cases)

| Method | Dice | HD95 | ASD |
|---|---|---|---|
| x4, one model | 0.9447 | 3.37 | 1.089 |
| x4, one model + TTA | 0.9472 | 3.17 | 1.040 |
| x4, ensemble of 3 + TTA | 0.9479 | 3.11 | 1.026 |
| **size-norm x4, one model** | 0.9462 | 3.17 | 1.060 |
| **size-norm x4, one model + TTA** | **0.9480** | **3.06** | 1.024 |
| **size-norm x4, ensemble of 3** | 0.9480 | 3.04 | 1.024 |
| **size-norm x4, ensemble of 3 + TTA** | **0.9487** | **3.01** | 1.010 |
| nnU-Net 250 ep | 0.9484 | 3.36 | 1.031 |
| nnU-Net 250 ep + TTA | 0.9496 | 3.19 | 1.000 |
| nnU-Net 1000 ep | 0.9497 | 3.09 | 0.990 |
| nnU-Net 1000 ep + TTA | **0.9510** | 3.03 | **0.964** |

Our best (size-norm, ensemble + TTA) against nnU-Net 1000:

| | ΔDice [95% CI] | p (Dice) | ΔHD95 | p (HD95) |
|---|---|---|---|---|
| vs nnU-Net 1000, no TTA | −0.0010 [−0.0030, +0.0011] | 0.37 (n.s.) | −0.08 | 0.89 |
| vs nnU-Net 1000 + TTA | −0.0022 [−0.0043, −0.0002] | 0.019 | −0.02 | 0.81 |

**Tied with nnU-Net 1000 without TTA, and slightly better than nnU-Net 250 + TTA on HD95;
0.002 Dice behind nnU-Net 1000 + TTA**, at a small fraction of the cost (below).

What each step adds (paired, 90 cases):

| Comparison | ΔDice | cases better | p | ΔHD95 |
|---|---|---|---|---|
| TTA vs plain (x4, one model) | +0.0025 | 69/90 | 1e-8 | −0.20 |
| ensemble vs one model (x4) | +0.0021 | 90/90 | 2e-16 | −0.16 |
| size-norm vs x4, one model | +0.0015 | 49/90 | 0.055 | −0.20 |
| size-norm vs x4, one model + TTA | +0.0008 | 46/90 | 0.35 | −0.11 |
| size-norm vs x4, ensemble + TTA | +0.0008 | 48/90 | 0.40 | −0.10 |

Normalisation's Dice gain shrinks once TTA and the ensemble are added, but it improves HD95
consistently and fixes the small heads that used to fail at x8.

## Size normalisation: is the estimated volume good enough?

**Cross-validation** (90 cases, one model, 3 seeds): normalising the test head with the
volume pass 1 estimates gives 0.9462; with the true volume (oracle) 0.9468. Pass 1's
volume error is 4.3% median (20% at worst); the error of the scale k it gives is 1.4%.

## Other arms since the restart (one model, no TTA, 3 seeds)

| Arm | Folds | Dice | HD95 | ASD | Reference on the same folds |
|---|---|---|---|---|---|
| x8 (3.2 mm) | 5 | 0.9412 | 3.45 | 1.150 | — |
| x8 antialiased | 5 | 0.9425 | 3.38 | 1.128 | x8 0.9412 |
| x8 dual head (SDF + mask) | 5 | 0.9426 | 3.38 | 1.124 | x8 0.9412 |
| x8 depth 4 | fold 3 | 0.9422 | 3.17 | 1.102 | x8 0.9417 / 3.18 |
| x8 size-norm | fold 3 | 0.9470 | 2.75 | 1.017 | x8 0.9417 / 3.18 |
| x4 depth 4 (5.65 M params) | 5 | 0.9445 | 3.36 | 1.090 | x4 0.9447 / 3.37 |
| x4 joint 1D point head | 5 | 0.9454 | 3.29 | 1.074 | x4 0.9447 (+0.0003 seed-matched, p=0.84) |
| x4 joint 1D, 1000 ep | 5, seed 42 | 0.9454 | 3.37 | 1.074 | worse than 500 ep for us |
| x4 pose + size canonical | fold 3 | 0.9465 | 3.22 | 1.033 | size-norm 0.9480 / 2.83 |
| x4 pose + size canonical, oracle pose | fold 3 | 0.9571 | 2.24 | 0.822 | leaks the label; not usable |

Set aside: depth 4 (no gain for 4× params), joint training (no robust gain over 3 seeds),
2D joint head, 1000 epochs (overfits), pose canonicalisation (realistic pose worse than
size-only; on standby), and the phase 2 refiners (cascade `phase2_v2`, 1D surface profiles
`phase2_profile`), which did not improve the x4 prior.

On the size-normalised network (against nnaug_x4norm, 3 seeds × 5 folds, paired Wilcoxon):
a Gaussian contour head −0.0002 (p = 0.81), soft-mask Dice + BCE −0.0005 (p = 0.30), a 2×
finer input in 8 channels −0.0003 (p = 0.90); with ensemble + TTA all within ±0.0003. 32
base channels (5.6 M params): 0.9433 against 0.9437 on seed 42 fold 0, then cancelled.

## Resolution and loss (size-normalised, seed 42, 5 folds)

Same network and recipe; only the grid (`--factor`) and the target (`--loss sdf|mask`)
change. The mask arm is Dice + BCE on the head mask (nnU-Net's binary loss); its
probability is carried to the native grid and cut at 0.5. Paired Wilcoxon against x4 SDF.
Inference on all 90 cases, 3 repeats (`bench_pipeline.py --pass2 <the arm's runs>`; the grid
comes from each run's config.json).

| Grid | Loss | Dice | Dice + TTA | HD95 | ASD | ms, one model / + TTA |
|---|---|---|---|---|---|---|
| x8 (48³) | SDF | 0.9442 (p = 0.013) | 0.9467 | 3.29 | 1.096 | 27 / 40 |
| x8 | mask | 0.9423 (p < 0.001) | 0.9453 | 3.23 | 1.121 | 27 / 40 |
| **x4 (96³)** | **SDF** | **0.9466** | 0.9477 | **3.14** | **1.049** | 36 / 115 |
| x4 | mask | 0.9453 (p = 0.057) | 0.9480 | 3.20 | 1.069 | 36 / 115 |
| x2 (192³), fold 0 only | SDF | 0.9371 (x4: 0.9437) | 0.9377 | 3.97 | 1.243 | 118 / 774 |
| x2, fold 0 only | mask | 0.9359 | — | 5.09 | 1.338 | 118 / 774 |

x2 needs 27 GB to train at batch 4; on the 12 GB card it ran at batch 1 with 4-batch
gradient accumulation (`--accum 4`), 6.2 h per fold, and was stopped after fold 0. The SDF
beats the mask without TTA at x8 and x4; with TTA the x4 mask ties.

**Representation ceiling**: the true target on the coarse grid, carried to native and
thresholded, as a perfect network would predict it (90 cases):

| Grid | SDF | mask |
|---|---|---|
| x8 normalised | 0.9961 | 0.9755 |
| x4 normalised | 0.9979 | 0.9874 |
| x2 normalised | 0.9995 | 0.9932 |

The coarse grid is not the limit. The plain-x4 ceiling (0.9967) is lower only because the
integer factor 4 samples the native voxel centres, where the voxel-wise SDF jumps ±0.4 mm;
factors 3.96 or 4.04 give 0.9980.

## Efficiency (same machine)

RTX 5070 12 GB, Ryzen 7 8700F (8 cores / 16 threads), 32 GB RAM, Windows 11, PyTorch 2.12.
The GPU ran nothing else.

### Inference

All 90 cases (each fold's validation cases with that fold's networks), 3 timed repeats after
2 warm-up cases, GPU synchronised at both ends of every stage, nothing else on the GPU.
Time from the native volume in memory to the native mask in host memory, for both methods
(z-score, all passes, all resampling, threshold, copy back); disk reading excluded. nnU-Net
through `nnUNetPredictor` (crop, z-score, sliding window with Gaussian blending, step 0.5,
export), the 250-epoch checkpoints (same network and plans as 1000 epochs, so the same cost).
`scripts/bench_pipeline.py --folds 0 1 2 3 4 --repeats 3`, and nnU-Net with the same flags;
results in `docs/profiling/bench_ours_90.json`, `bench_nnunet_90.json`. The medians of
the 3 repeats agree to the millisecond.

| | Time / case, median | p5–p95 | Networks only | Pipeline around them | Network passes | Params |
|---|---|---|---|---|---|---|
| x4, one model | 23 ms | 21–31 | 11.5 | 11.8 | 1 | 1.40 M |
| x4, one model + TTA | 103 ms | 101–111 | 91.2 | 11.9 | 8 | 1.40 M |
| x4, ensemble of 3 | 46 ms | 44–54 | 34.1 | 11.6 | 3 | 4.20 M |
| x4, ensemble of 3 + TTA | 284 ms | 282–292 | 272.1 | 11.8 | 24 | 4.20 M |
| **size-norm, one model** | **35 ms** | 33–43 | 22.9 | 12.4 | 2 | 2.80 M |
| **size-norm, one model + TTA** | **115 ms** | 113–123 | 102.2 | 12.6 | 9 | 2.80 M |
| **size-norm, ensemble of 3** | **58 ms** | 56–66 | 45.6 | 12.6 | 4 | 5.61 M |
| **size-norm, ensemble of 3 + TTA** | **296 ms** | 294–304 | 283.6 | 12.6 | 25 | 5.61 M |
| nnU-Net | 1339 ms | 935–2345 | — | — | sliding window | 30.8 M |
| nnU-Net + TTA | 8870 ms | 5956–15733 | — | — | sliding window × 8 | 30.8 M |

- Size-norm one model is 38× faster than nnU-Net; with TTA, 12× faster than nnU-Net
  without TTA and 77× faster than nnU-Net + TTA; the ensemble + TTA is 30× faster than
  nnU-Net + TTA. Ours barely varies with the scan (always a 96³ cube); nnU-Net's sliding
  window scales with the volume.
- "Pipeline around them" is everything but the networks: z-score 4.5, to x4 grid 0.7, head
  volume 0.2, to the normalised grid 0.6, SDF to the native grid 5.2, threshold + copy to
  host 1.3 (median ms). About 12 ms in every configuration.
- GPU memory peak: ours 0.95 GB allocated, nnU-Net 1.9 GB. CPU per case: ours 0.03–0.17 s,
  nnU-Net 0.62 s (1.85 s with TTA). RAM includes the PyTorch/CUDA runtime (~1.4 GB).
- The contour-head, soft-mask and fine-input arms use the same inference network, so they
  cost the same (fine input would add the 2× grid resampling).
- Turning a label into the SDF training target (`scripts/bench_sdf.py`): 1.7 s per case
  median (1.3–3.4 s), 3 min for the 90 cases, almost all in the exact distance transform at
  native resolution. A one-off cache-building cost; inference never does it.
- Resampling uses `geometry.affine_resample` (analytic grid on the GPU). The older path
  (`Spacingd` on the CPU, `resample_grid`'s per-voxel float64 grid) gives identical Dice but
  costs 107 / 185 / 365 ms, 1.56 GB allocated and 1.2–1.9 s CPU per case.
- Fold 0 is harder than average; the 5-fold Dice is in the main table.

### Training

One job alone on the card. Ours: 10 epochs of the size-norm x4 recipe. nnU-Net: 5 epochs of
`nnUNetTrainer` with the plans of the 2ai run (patch 112×160×128, batch 2), on 20 cases
written locally; an nnU-Net epoch is a fixed 250 iterations, so its cost does not depend on
the dataset size. Memory and CPU sampled every second (`scripts/profile_run.py`); GPU
memory is the card's total minus its idle level (Windows reports no per-process figure).

| | Ours | nnU-Net |
|---|---|---|
| Params | 1.40 M | 30.8 M |
| Batch × input | 4 × 96³ (1.6 mm, normalised) | 2 × 112×160×128 (0.4 mm) |
| Time / epoch | 18.0 s (train 10.8 + full-volume val 7.2) | 67.4 s (250 + 50 val iterations) |
| Epochs | 500 | 1000 (or 250) |
| **Time / fold** | **2.5 h** | **18.7 h** (4.7 h at 250 ep) |
| 5-fold CV | 12.5 h per seed | 94 h |
| GPU memory, PyTorch peak allocated | 3.5 GB | — |
| GPU memory on the card, median / peak | 5.4 / 5.5 GB | 7.7 / 11.0 GB |
| RAM, process + workers, median / peak | 3.4 / 3.6 GB | 19.6 / 20.1 GB |
| CPU, cores busy, median / p95 | 1.9 / 2.4 | 3.2 / 4.5 |
| GPU utilisation, mean | 21% | 98% |

Our training is data-loader bound (GPU busy ~21% of the time, 2 workers; 6 workers were
slower on this machine), so it could get faster. nnU-Net keeps the GPU saturated and needs
~20 GB of RAM for its augmentation workers.

## Reproduce

```
python scripts/bench_pipeline.py --folds 0                     # ours, inference
python scripts/bench_pipeline.py --folds 0 --resampler geometry
python scripts/profile_run.py --out prof.json -- <training command>
```
