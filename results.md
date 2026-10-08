# Results

Everything since the folds were regrouped by patient (commit `5f3bc7d`). The older numbers in
`docs/results.md` used folds that leaked patients and are not comparable with these.

## Protocol

- 90 volumes, 5-fold cross-validation grouped by patient (`splits.json`); nnU-Net trained on
  the same folds (`nnunet/runs/grouped_*`).
- Dice, HD95 (mm), ASD (mm) at native resolution against the label.
- Our numbers are the **last epoch** (nothing selected on the reported fold), and the
  **per-case mean over 3 seeds** (42, 43, 44). A method is compared on the same 90 cases with
  a paired Wilcoxon test; one seed is never enough to conclude (seed-to-seed spread is
  ~0.002 Dice).
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
volume pass 1 estimates gives 0.9462; with the true volume (oracle) 0.9468. The pass-1
volume error is 1.4% median, ~6% at worst.

**Held-out split** `splits_60525.json`: 60 train / 5 val / 25 test, patient-grouped,
stratified by head size, 3 seeds per arm. Test normalised with the estimated volume:

| Training heads normalised with | Test, estimated volume | Test, true volume |
|---|---|---|
| A: the true volume | **0.9460** / 3.31 / 1.074 | 0.9465 / 3.36 |
| B: the estimated volume | 0.9452 / 3.47 / 1.093 | 0.9452 / 3.49 |

B − A: Dice −0.0008 (p = 0.33), HD95 +0.16 (p = 0.045). **Train with the true volume**;
the estimate at test time costs 0.0005 Dice.

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

## Efficiency (same machine)

RTX 5070 12 GB, Ryzen 7 8700F (8 cores / 16 threads), 32 GB RAM, Windows 11, PyTorch 2.12.
The GPU ran nothing else.

### Inference

Fold 0, 18 cases, after 2 warm-up cases. Time from the native volume in memory to the native
mask (z-score, both passes, all resampling, threshold); disk reading excluded for both.
nnU-Net through `nnUNetPredictor` (crop, z-score, sliding window with Gaussian blending,
step 0.5, export). `scripts/bench_pipeline.py`, `scripts/bench_nnunet.py`.

| | Time / case (median) | GPU memory peak (allocated / reserved) | RAM peak (process) | CPU time / case | Params | Network passes | Dice, fold 0 |
|---|---|---|---|---|---|---|---|
| **Ours, one model** | **35 ms** | 0.70 / 1.70 GB | 1.46 GB | 0.07 s | 2.80 M | 2 | 0.9438 |
| **Ours, one model + TTA** | **115 ms** | 0.70 / 1.71 GB | 1.52 GB | 0.14 s | 2.80 M | 9 | 0.9454 |
| **Ours, ensemble of 3 + TTA** | **309 ms** | 0.70 / 1.71 GB | 1.52 GB | 0.22 s | 5.61 M | 25 | 0.9460 |
| nnU-Net | 1348 ms | 1.43 / 1.82 GB | 1.96 GB | 0.71 s | 30.8 M | sliding window | 0.9481 |
| nnU-Net + TTA | 8903 ms | 1.44 / 1.82 GB | 2.01 GB | 2.19 s | 30.8 M | sliding window × 8 | 0.9481 |

- One model + TTA is **77× faster** than nnU-Net + TTA and **12× faster** than nnU-Net
  without TTA; the full ensemble + TTA is 29× faster than nnU-Net + TTA. 11× fewer
  parameters (2.8 M vs 30.8 M), half the GPU memory.
- RAM includes the PyTorch/CUDA runtime (~1.4 GB for both).
- Our stages (one model + TTA, median ms): z-score 5, pass 1 12, normalise 1, pass 2 91,
  back to native 6.
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
python scripts/bench_pipeline.py --fold 0                      # ours, inference
python scripts/bench_pipeline.py --fold 0 --resampler geometry
conda activate nnunet-bench
python scripts/bench_nnunet.py --fold 0                        # nnU-Net, inference
python scripts/profile_run.py --out prof.json -- <training command>
```
