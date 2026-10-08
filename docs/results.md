# Results

Every arm evaluated so far, with what it changed and what it measured. Numbers are
validation means over the held-out fold. Dice is against the native-resolution ground
truth label; HD95 and ASD are in mm.

**Every number in this file up to the EMA rerun was produced with the checkpoint chosen by
validation score on the fold being reported, and is optimistic by about 0.0024 Dice.** See
[Evaluation protocol](#evaluation-protocol) below. Differences between arms are still
meaningful, since all share the bias; absolute numbers and the comparison with the nnU-Net
are not.

Folds are patient-grouped: 90 volumes from 82 patients, so no patient appears in both
train and validation. Fold difficulty is uneven — under the baseline, fold 1 scores
0.9452 and fold 4 scores 0.9364. **A difference smaller than ~0.009 between two arms on
different folds means nothing.** Two-fold arms use 1 and 4 (easiest and hardest) as
triage; only arms worth it go to five.

---

## Phase 1 — SDF prior

Regresses a signed distance field on the downsampled volume. Truncated at 10 mm,
normalised to [-1, 1], positive outside. All arms: 300 epochs, lr 5e-4, seed 42, x8
cache (3.2 mm/voxel) unless stated.

| Arm | folds | MAE | Dice | HD95 | ASD | What it changes |
|---|---|---|---|---|---|---|
| baseline | 5 | 0.0445 | 0.9417 | 3.502 | 1.152 | Plain L1 on the SDF |
| **sw05** | **5** | 0.0469 | **0.9447** | **3.312** | **1.087** | **L1 + 0.5 × surface-weighted L1, σ=1.5 mm** |
| sw | 5 | 0.0507 | 0.9431 | 3.531 | 1.122 | L1 + 1.0 × surface-weighted L1 |
| x4 | 5 | 0.0428 | 0.9418 | 3.647 | 1.153 | Same loss, 1.6 mm/voxel instead of 3.2 |
| swonly | 2 | 0.0712 | 0.9423 | 3.516 | 1.128 | Surface-weighted only, base_coef=0 |
| sw20 | 2 | 0.0619 | 0.9396 | 3.788 | 1.182 | L1 + 2.0 × surface-weighted |
| sw05s | 2 | 0.0787 | 0.9290 | 4.662 | 1.381 | sw=1.0 with σ=0.5 mm instead of 1.5 |
| logcosh | 1 | 0.0555 | 0.9435 | 3.290 | 1.100 | log-cosh base loss instead of L1 |

Two-fold means are over folds 1 and 4, whose baseline mean is 0.9408 — **not** comparable
with the five-fold 0.9417.

### Per-fold, the arms that matter

| Fold | baseline | sw | sw05 | x4 |
|---|---|---|---|---|
| 0 | 0.9389 | 0.9414 | 0.9414 | 0.9370 |
| 1 | 0.9452 | 0.9463 | **0.9473** | 0.9457 |
| 2 | 0.9441 | 0.9445 | 0.9464 | **0.9484** |
| 3 | 0.9438 | 0.9432 | **0.9479** | 0.9391 |
| 4 | 0.9364 | 0.9400 | **0.9404** | 0.9388 |

### What phase 1 established

**The surface-weighted term works, and 0.5 is its optimum.** Plain L1 treats a 9 mm error
as nine times a 1 mm one, but everything downstream depends only on where the field
crosses zero, and most voxels sit in the saturated far field at ±1 — they dominate the
gradient while carrying no information about the boundary. Weighting by
`exp(-|sdf|/σ)` says that directly.

The sweep is unimodal: 0.5 > 1.0 > 2.0, and 2.0 is already worse than plain L1. So 1.0,
which was the first value tried, was not the right one. sw05 is the only arm that wins
Dice, ASD and HD95 simultaneously.

**Two informative negatives.** `swonly` (base_coef=0, dropping the plain term) scores
below sw05: the far field does need supervising. The band is defined as
`|sdf_prior| < 5 mm`, so a field that is wrong out there puts the band in the wrong
place. And `sw05s` (σ=0.5 mm) collapses on fold 4 — 0.9149, HD95 5.9 mm — concentrating
supervision too tightly on the surface destabilises training.

**Resolution is not the lever.** x4 halves the voxel size and improves MAE (0.0428 vs
0.0445) but moves Dice by +0.0001 and makes HD95 worse. It also costs 4× the memory.

**log-cosh does not replace L1.** 0.9435 on fold 1 against 0.9452 for the baseline —
worse, on the one fold measured so far.

---

## Phase 2 — band refiner

Takes native-resolution crops around phase 1's predicted band and refines inside it.
Normal-aligned slabs, 128×128×48, 4 per case, 100 epochs. All arms below sit on the
**baseline** prior, so `coarse` is what phase 1 gave and `refined` is what phase 2
returned.

| Arm | folds | coarse | refined | Δ Dice | HD95 | ASD | What it changes |
|---|---|---|---|---|---|---|---|
| **sw05reg** | **5** | **0.9447** | **0.9453** | +0.0006 | **3.318** | **1.072** | **sw05 prior + weight decay 1e-4, dropout 0.1, strong augmentation + blend** |
| blend | 5 | 0.9417 | 0.9440 | **+0.0023** | 3.440 | 1.102 | Gaussian tile weighting, σ=1/8 of extent |
| band5 | 5 | 0.9417 | 0.9438 | +0.0021 | 3.446 | 1.104 | 5 mm band, uniform tile averaging |
| band3 | 5 | 0.9417 | 0.9438 | +0.0021 | 3.457 | 1.105 | 3 mm band |
| prop_band3 | 4 | 0.9430 | 0.9452 | +0.0022 | 3.383 | 1.075 | 3 mm proportional band, thin=32 |
| band75 | 5 | 0.9417 | 0.9431 | +0.0014 | 3.470 | 1.115 | 7.5 mm band |
| band10 | 5 | 0.9417 | 0.9413 | **−0.0004** | 3.538 | 1.151 | 10 mm band |
| deep | 2 | 0.9408 | 0.9424 | +0.0015 | 3.495 | 1.128 | Four pooling levels; the refiner goes 1.4M → 5.6M params |
| deepblend | 2 | 0.9408 | 0.9423 | +0.0014 | 3.501 | 1.131 | deep + Gaussian blending |

Read the Δ column against the `coarse` it starts from, not across rows. `sw05reg` has the
smallest Δ and the highest result: its refiner is handed a prior 0.0030 better and has
correspondingly less left to fix.

### Regularisation: the right diagnosis, the wrong lever

The training logs said overfitting. Across 25 unregularised runs the best checkpoint
arrived at 60% of the 100-epoch budget, only 1 run in 30 peaked in the final 10%, and
validation loss drifted +0.0062 past its minimum while training loss fell 0.51.

Weight decay 1e-4, Dropout3d 0.1 on the two deepest levels, and wider appearance
augmentation moved all three signals in the predicted direction:

| | unregularised | regularised |
|---|---|---|
| best epoch, fraction of budget | 0.60 | **0.69** |
| validation-loss drift past minimum | +0.0062 | **+0.0050** |
| training-loss fall | 0.5052 | **0.4689** |

**And it bought no Dice.** On the same prior and the same fold, the unregularised refiner
gained +0.0007 and the regularised one +0.0005. The overfitting was real and reducing it
changed nothing downstream, so it was not what limited the result.

That closes the phase 2 capacity question. Three hypotheses, three eliminated: capacity
(`deep`, 4× the parameters, −0.0011 on the hard fold), epoch budget (1 run in 30 peaks
late), and overfitting (measured, corrected, no gain). What remains is the prior.

### The saturation shows up fold by fold

Within `sw05reg`, the gain tracks how much error the prior left:

| fold | prior | Δ from phase 2 |
|---|---|---|
| 3 | 0.9479 | +0.0004 |
| 1 | 0.9473 | +0.0003 |
| 2 | 0.9464 | +0.0009 |
| 0 | 0.9414 | +0.0005 |
| 4 | **0.9404** | **+0.0011** |

The worst prior gets the largest refinement and the best priors the smallest. This is the
same curve the degraded-prior experiment traced, now visible across folds of one arm
rather than across arms.

### Band width has an optimum

3–5 mm works, 7.5 mm is weaker, **10 mm makes Dice worse than not refining at all**. A
wider band means more easy interior voxels diluting the supervision, and the crop's
bounding box grows with it. This is the clearest signal in the phase 2 sweep.

### Depth was measured and rejected

`deep` quadruples the parameters for +0.0003 over band5 on fold 1, and on fold 4 it
**loses 0.0011** — while also making HD95 (3.910 vs 3.858) and ASD (1.229 vs 1.211)
worse. Fold 4 has the weaker prior, so it is where added capacity had the most error to
find, and it is where depth did worst.

The signature is overfitting, not saturation: `deep_f4` has std 0.0207 and min 0.8813,
against 0.0116 / 0.9244 on fold 1. With 72 training cases, 5.6M parameters is too many.

This matches the earlier width result — the polar refiner went 177k → 1.58M parameters
for 0.02 mm of MAE.

### Depth and blending do not add

`deepblend` was built to test whether they compose, since they act on different things
(what a tile predicts vs how tiles are combined). They do not: fold 1 is identical to
`deep` alone (0.9468), fold 4 is 0.0001 worse. Blending corrects stitching
discontinuities, and a deeper model with more receptive field already smooths them —
correcting the same thing twice gains nothing.

### Blending is the best phase 2 arm

+0.0023 over the prior, best HD95 and ASD, **zero extra parameters and no measurable
inference cost** — it only changes how overlapping tiles are averaged. A voxel at a tile
border was decided with its context truncated on one side; weighting each tile by a
Gaussian over its own extent lets the tile that saw most around a voxel decide it.

---

## Saturation

Phase 2's gain depends on how much error the prior left behind:

| prior Dice | phase 2 gain |
|---|---|
| 0.8550 (deliberately degraded) | **+0.0384** |
| 0.9364 (fold 4) | +0.0028 |
| 0.9417 (five-fold mean) | +0.0023 |
| 0.9452 (fold 1) | +0.0014 |

A line through these four points crosses zero at prior Dice **0.9464**. sw05 puts fold 1
at 0.9473 — above that crossing.

**This is a four-point fit dominated by the degraded-prior point, and should not be
trusted over a measurement.** It is the reason `sw05 + blend` is worth running rather
than assuming: if the curve holds, phase 2 adds nothing on top of a good prior and the
honest result is phase 1 alone. Fold 4 under sw05 (0.9404) sits below the crossing while
fold 1 (0.9473) sits above it, so the two folds should disagree if the curve describes
something real.

What the saturation experiment does establish: **the refiner works when there is error
to fix.** +0.0384 at a degraded prior is not the behaviour of a network that cannot
learn. Phase 2's small gains are a statement about the prior's quality, not the
refiner's capacity.

---

## Where this leaves the method

| | Dice | HD95 | ASD | params | inference |
|---|---|---|---|---|---|
| nnU-Net 250ep, with TTA, final checkpoint, leaky folds | **0.9493 ± 0.0048** | 3.375 | **1.022** | 88.2 M | 5.63 s |
| v1 ShellRefine, leaky folds | 0.9457 ± 0.0045 | **3.202** | 1.063 | 2.80 M | 1.05 s |
| v2, sw05 + swwbce, best-on-val | 0.9454 ± 0.0034 | 3.313 | 1.072 | 2.80 M | ~1.16 s |
| v2, sw05 phase 1 alone, best-on-val | 0.9447 ± 0.0035 | 3.313 | 1.087 | **1.40 M** | **0.10 s** |
| v2, sw05 phase 1 alone, **last epoch** | 0.9422 | 3.598 | — | 1.40 M | 0.10 s |
| v2, sw05 + swwbce, **EMA** | *running* | | | 2.80 M | ~1.16 s |
| v2, sw05 phase 1 alone, **EMA** | *running* | | | 1.40 M | 0.10 s |
| v2, baseline + blend, best-on-val | 0.9440 ± 0.0039 | 3.440 | 1.102 | 2.80 M | 1.16 s |

The rows are **not** directly comparable; see [Evaluation protocol](#evaluation-protocol).
The v2 best-on-val rows are optimistic by about 0.0024 Dice, and the nnU-Net and v1 were
evaluated on folds that split patients across train and validation. The honest v2 number
is the EMA row once it lands.

On the numbers as they stand, v2 is 0.0003 short of v1 with a tighter spread (± 0.0034
against ± 0.0045) and 0.0039 short of the nnU-Net. **An earlier version of this section
said v2 beats the nnU-Net on HD95 (3.313 against 3.375). That does not hold:** on the 76
cases whose patient has no scan in the nnU-Net's training fold, the nnU-Net's HD95 is
3.166 against v2's 3.286. The apparent win came from nnU-Net outliers on the 14 leaked
cases.

Almost all of the improvement came from phase 1. The prior went 0.9417 → 0.9447 (+0.0030,
from sw05) while the pipeline went 0.9440 → 0.9453 (+0.0013) — phase 2's own contribution
shrank by more than half as the prior improved, which is the saturation working as
measured. Phase 1 alone, at 1.40 M parameters and 0.10 s, is within 0.0006 of the two-phase
pipeline.

Parameter counts are for the whole pipeline: phase 1 is 1,401,265 and phase 2 is
1,401,697 (the extra 432 are the second input channel, which carries the prior), so the
two-phase method is 2.80 M — **the same as v1**, not half of it. The size advantage is
over the nnU-Net (31×), not over v1.

Phase 1 alone is the exception: one network, 1.40 M, and on five folds it already beats
the full baseline+blend pipeline (0.9447 against 0.9440) at a tenth of the inference cost.
That is a statement about how little phase 2 currently adds, not about phase 1 being a
finished method.

**Caveats that matter.** The nnU-Net number is *with* mirror TTA (8 passes, 5.63 s against
0.75 s without), and its quality without TTA was never evaluated — so "faster without it"
cannot be claimed against a number measured with it. The fold and checkpoint issues are in
the next section.

---

## Evaluation protocol

Two things measured on 2026-09-28 make the comparisons above less clean than they looked.

### Checkpoint selection on the reported fold

Both phases saved the epoch with the best validation score, and that validation fold is
the one the numbers are reported on. Choosing the maximum of 300 noisy scores on 18 cases
picks up luck as well as merit. Scoring the sw05 phase 1 at its **last** epoch instead:

| fold | best-on-val | last epoch | difference |
|---|---|---|---|
| 0 | 0.9414 | 0.9398 | −0.0016 |
| 1 | 0.9473 | 0.9449 | −0.0024 |
| 2 | 0.9464 | 0.9438 | −0.0026 |
| 3 | 0.9479 | 0.9451 | −0.0028 |
| 4 | 0.9404 | 0.9375 | −0.0030 |
| **mean** | **0.9447** | **0.9422** | **−0.0024** |

HD95 goes 3.31 → 3.60 mm. The drop is on every fold, so it is a bias, not noise. The
nnU-Net reports its final checkpoint, so the comparison was tilted towards v2.

**The fix, now running:** an exponential moving average of the weights (decay 0.999, about
1000 steps: ~55 epochs in phase 1, ~14 in phase 2), reported at the end of training.
Validation is still computed every epoch, but only logged. The best and last checkpoints
are scored alongside, and last ≤ EMA ≤ best is the expected order. Jobs `B00`–`B02`,
runs `fold*_sw05ema` and `swwbceema_f*`.

### The nnU-Net's folds leak patients, and are not ours

The nnU-Net inherited v1's folds, which predate patient grouping. Mapped through its
`case_map.json`, **all six patients with more than one scan are split across folds: 14 of
90 validation volumes have a sibling scan in training.** None of its five folds matches
`splits.json`.

The leak does not explain the gap. Paired case by case against v2 (sw05 + swwbce,
best-on-val):

| | cases | nnU-Net Dice | v2 Dice | nnU-Net better on | nnU-Net HD95 | v2 HD95 |
|---|---|---|---|---|---|---|
| all | 90 | 0.9493 | 0.9454 | 60 | 3.376 | 3.313 |
| leaked | 14 | 0.9507 | 0.9465 | 8 | 4.514 | 3.462 |
| clean | 76 | 0.9491 | 0.9452 | 52 | **3.166** | **3.286** |

On clean cases the nnU-Net still leads on Dice (+0.0038) and also leads on HD95.

### Where the gap is

The gap is not spread evenly. **Ten cases carry 86% of it.** By quintile of v2 Dice, the
nnU-Net's lead is +0.0183 on the hardest 18 cases and roughly zero on the other 72. Those 18
are phase 1 failures: coarse Dice 0.9193, phase 2 adds +0.0006, HD95 5.36 mm. Their error
lies mostly inside the band (0.064% of GT disagreement outside it), so the band is not what
stops phase 2. The prior's shape is wrong at a scale a local refiner does not correct.

### Pending

- **The nnU-Net on `splits.json`**, trained in an isolated tree on 2ai, so both methods are
  evaluated on the same patient-grouped folds. Postponed until v2 reaches about 0.9485
  Dice under the EMA protocol; until then 0.9493 is an approximate target, not a fair one.

### Biometry: considered and dropped

Clinical head circumference is measured on the transthalamic plane, which is defined by
structures inside the brain (thalami, cavum septi pellucidi, falx). The label is the whole
head as one solid region, so that plane cannot be found from a segmentation; the
anthropometric measures (biocular, orbitotragial, frontotemporale) need landmarks as well.
Both would need a second, plane- or landmark-finding network, so evaluation stays on Dice,
HD95 and ASD.

Two dataset facts found while prototyping it are worth reporting:

- **The label covers the whole head, face and ears included**, not only the cranial vault.
- **Most heads are cut by the ultrasound sector.** About half of each volume lies outside
  the scan cone. In 80 of 90 labels more than 0.5% of the head's surface lies along the
  sector edge (median about 6%, up to 13%): the annotation stops where the image stops. This
  truncation goes with *higher* Dice (Spearman +0.27, p = 0.009), since a straight sector
  edge is easy to predict; the hard cases are the less truncated heads.

---

## What phase 2 costs on individual cases

The five-fold means hide something that matters: **phase 2 does not improve every case, and on
HD95 it is close to a coin flip.**

| metric | better | worse | unchanged | mean |
|---|---|---|---|---|
| Dice | 58 | **32** | 0 | +0.0007 |
| ASD | 56 | **34** | 0 | −0.0154 |
| **HD95** | **39** | **38** | 13 | **+0.0003** |

On HD95 the total harm is **+6.02 mm** against **−5.99 mm** of help, spread over 38 cases —
not concentrated in a few, so no small gate would fix it. The worst single case goes 5.075 →
5.571 mm; the best goes 4.596 → 3.600 mm.

There is a structural reason, and it is consistent with the measurement. HD95 is the 95th
percentile of surface distance, so it is set by the worst points — and **0.65% of the true
surface lies outside the band**, where phase 2 cannot reach. It cannot fix those points, but
rewriting the band beside them perturbs the boundary there. So phase 2's HD95 changes are
largely noise injected near the outliers rather than signal. The cases it hurts have **3×
more true surface outside the band** (0.028% against 0.0095%) and a worse prior to begin with
(HD95 3.68 against 3.09) — both consistent with that account, and neither observable at
inference time, since both need the ground truth.

An oracle that picked per case whichever of prior and refinement was better would give Dice
0.9458, HD95 3.246, ASD 1.063. That is a ceiling, not a proposal: the pick needs the ground
truth.

**This is a real weakness to state rather than smooth over.** A refinement stage that leaves
one metric flat and degrades a third of cases on it is hard to defend as strictly better. The
defensible framings are either that phase 2 buys ASD and Dice at no cost in HD95, or that
phase 1 alone is the method and phase 2 is a measured extension whose value the prior's
quality has largely exhausted.

---

## Tried and dropped

- **SWA** — weight averaging over the last epochs. Measured, did not beat best-checkpoint selection. **That comparison is void:** best-checkpoint selection was made on the reported fold and is optimistic by ~0.0024 (see [Evaluation protocol](#evaluation-protocol)), so SWA was compared against an inflated baseline. The EMA rerun is the fair version of this test.
- **Sparse convolution on the band** — submanifold sparse conv in pure PyTorch. Parked in `phase2/sparse_experiment/`; the dense tiled refiner was simpler and no slower at this band width.
- **Polar / spherical coordinates** — resampling the band into a polar frame. Round-trip resampling error and the star-convexity assumption (which will not hold once this generalises past the fetal head) outweighed the gain.
- **Proportional band 7.5 / 10 mm** — dropped before running. Fixed-width 7.5 and 10 mm already gave +0.0014 and −0.0004; with a better prior they would give less.
- **Deep supervision** — auxiliary losses at intermediate decoder resolutions. Not run: the network is 3–4 levels deep, so gradients already reach the early layers, and `deep` showed the refiner overfits rather than underfits.
- **A third phase on phase 2's uncertain voxels** — refine only where phase 2's probability sits in (0.3, 0.7). The selection is genuinely informative: those voxels are 8.7% of the band and hold **37.4% of all phase 2 error**, at a 41.8% error rate against 3% in the confident extremes, and they form **one connected shell** per case rather than scattered fragments — phase 2 is well calibrated and knows where it does not know. The oracle ceiling is **+0.0207**, the largest measured in the project. It was dropped anyway, on three numbers: a 16³ patch centred on an uncertain voxel is only **16.8% uncertain** (8.5% at 32³), so the selection does not concentrate at patch level; phase 1's sign is right **56.4%** of the time there against phase 2's **58.2%**, so the prior adds nothing a third network could condition on; and a trivial "always inside" classifier already scores **54.4%**, meaning phase 2's edge over trivial is 3.8 points and the information in that zone is close to exhausted. Reaching 0.949 would need the accuracy there to go 58.2% → ~65%, roughly doubling the margin phase 2 extracted, from the same image at the same resolution. Realistic capture is +0.001–0.002 for a whole extra phase, its own crops, training and a third inference pass. The concentration it found is better attacked inside phase 2 by weighting the loss with the model's own uncertainty.
- **Iterating phase 2 on its own output** (`phase2/recrop.py`) — rebuild the band from the refined mask and run the same checkpoint again. Measured on `sw05reg_f0`: only **0.4% of the new band is voxels the old band did not cover** (max 0.8%). Phase 2 moves the boundary by a fraction of a voxel and the band is ~12 voxels thick at 0.4 mm, so a second pass looks at the same voxels. The script was removed after cleanup; the measurement is the useful part.
- **Shifting either threshold** — phase 1 decides by the zero crossing (`sdf < 0`), phase 2 by `p ≥ 0.5`. Swept both. Phase 1: Dice 0.9447 at 0 mm, 0.9427 at ±0.25 mm, 0.9369 at ±0.5 mm — symmetric, peaked exactly at zero, because phase 1 has no systematic bias (predicted/true volume 1.0041, over-segmenting in 48 of 90 cases and under in 42). Phase 2: 0.9453 at 0.5, 0.9446 at 0.6, **0.9410 at 0.3**. Both are already at their optimum, and the well-calibrated symmetry is why.
- **Gated skip connections in phase 1**, in two forms. The motivation was measured and is the strongest case in the project for an attention mechanism: of the padded 48³ cube phase 1 sees, **77.6% is padding, 15.8% saturated far field, 3.6% head interior, and only 2.9% lies within 5 mm of the boundary** — the decision the network exists to make lives in under 3% of its input. (Discounting padding, which a convolution ignores without help, the ratio of irrelevant to relevant real volume is still about 7:1.) And the same idea already works in the loss, where surface weighting is the method's largest single gain. Both forms lost.
  - **Learned attention gates** (Oktay et al.: the decoder signal produces a sigmoid map scaling the encoder skip, 8,179 parameters, +0.58%). **0.9414** against sw05's 0.9439 on two folds. Sigmoid rather than softmax deliberately — softmax over space would make attention zero-sum across the volume, so attending to one part of the boundary would suppress another. (The USegNet-DS-HAS reference implementation applies softmax along the channel axis, which silently makes its "spatial" attention channel attention.)
  - **SDF-driven gates** — gate each skip by `exp(−|sdf|·trunc/σ)` read off the decoder's own provisional field, the same expression the loss uses. **115 parameters, 0.01%**, and no learned attention map: the argument being that a network regressing a distance field does not need to learn where the boundary is, because the quantity it predicts says so directly. On folds 1 and 4 it looked like the best phase 1 arm in the project — 0.9459 against 0.9439, better ASD and HD95, with fold 4 gaining **+0.0049**. **On five folds it loses: 0.9427 against 0.9447**, winning one fold and losing four, with ASD 1.164 against 1.087.
  
  The diagnosis of why is worth keeping, because it is a lesson about the triage method rather than about gates. The gate **trains worse in four of five folds on the checkpoint-selection metric itself** (validation boundary Dice: −0.0106, −0.0038, −0.0119, −0.0002, and +0.0071 only on fold 4), and its curves are **2–3× noisier** in the last 20 epochs — mechanically explicable, since the gate decides where to look from a prediction the network has not yet made, which is a feedback loop. Roughly **half of the per-fold difference comes from 3 cases of 18** in every fold examined, and on fold 4 a single case worth +0.0525 accounts for 60% of that fold's apparent gain. **Two folds are not enough when one of them behaves atypically** — the arm was presented as a clear win on that evidence and was not one. A σ sweep (1.5, 2.0, 4.5, 6.0 mm against the original 3.0) found nothing above sw05 at any value, so no setting rescues it.
  
  Phase 2 on the sgate prior did behave as the saturation curve predicts — it gained **+0.0011** against +0.0007 on the better prior, confirming that a worse prior is more refinable — but recovered only 0.0004 of the 0.0020 the prior had lost, finishing at 0.9438 against 0.9454.
- **Mirror test-time augmentation** — average the field over flips of the input, un-flipping each output. Exact here, because a signed distance field is a scalar: reflecting moves where each value sits without changing the value, so un-flipping is the whole of the inverse with no interpolation. Motivated by the reference nnU-Net number being produced *with* 8-way TTA (5.63 s per case against 0.75 s without, its quality without never evaluated), so part of the gap might have been averaging rather than modelling. It is not. **4-way gives +0.0002** (noise: three folds better, two worse) and **8-way gives −0.0023**, with HD95 degrading 3.313 → 3.675. 8-way being worse than 4-way locates the cause: the diagonal flips, which combine two or three axes, are what hurt — the head is not symmetric under them and phase 1's augmentation shows the network single-axis flips only. So TTA's benefit to the nnU-Net is not a measurement artefact we can match.
- **InstanceNorm / LeakyReLU** (the nnU-Net's own choices, read from its `nnUNetPlans.json` for this dataset) — the one architectural difference with a mechanism rather than just more capacity. Phase 2 trains at batch size 1, and the mechanism is real: measured, with BatchNorm in training mode the same sample gives an output differing by **0.79** depending on which samples share its batch, against **2×10⁻⁶** for InstanceNorm. Parameter count is identical. It **degrades the prior**: fold 1 goes 0.9473 → 0.9462, fold 4 0.9404 → 0.9391, both below not refining at all, and 0.0018–0.0024 below the BatchNorm arm. Third instance of the same pattern as the regularisation and the SDF gate: the diagnosis was correct, the lever was wrong.
- **Loss weighting by phase 2's own uncertainty**, instead of by distance to the prior. `swwbce` (the kept arm) weights the BCE by `exp(−|sdf_prior|/σ)` — voxels near the predicted boundary count more. Two alternatives: **uncw** weights instead by `4p(1−p)`, the model's own predictive uncertainty (peaked at p=0.5), and **hybw** multiplies the two weightings together. On folds 1 and 4: uncw gives 0.9476/0.9417 (mean 0.9447), hybw gives 0.9479/0.9415 (mean 0.9447), both essentially tied with swwbce's 0.9480/0.9415 (mean 0.9448) and inside noise of each other. Uncertainty weighting is not a distinct lever once distance-to-prior is already in the loss — the two correlate too closely near the boundary to add independent signal. Distance was kept for simplicity, having been established first and measured on all five folds.
- **Sample re-weighting toward hard patches** (`hardw`) — bias patch sampling during phase 2 training toward crops where phase 1's prior was locally wrong, rather than uniform sampling over the band. 0.9474/0.9416 on folds 1 and 4 (mean 0.9445), essentially level with swwbce (0.9448) and not worth the extra sampling machinery.
- **Gradient-based phase 1 losses**, two forms, both on top of sw05 (L1 + surface weighting). **sw05gd** adds `gradient_difference_loss` (an L2 penalty on the SDF's spatial gradient, encouraging locally consistent slope): 0.9446/0.9417 on folds 1 and 4, essentially level with sw05 alone. **sw05gcos** adds a cosine-similarity penalty between predicted and true gradient direction: 0.9445/0.9395, slightly below sw05 on fold 4. Neither gradient term earns its complexity once surface weighting already concentrates supervision at the boundary, which is presumably where a slope penalty would matter most anyway.
- **nnU-Net-style post-processing** — largest-connected-component retention and hole filling, which nnU-Net applies by default. Worth **+0.00001**, which is nothing. The reason is structural and worth keeping: **phase 1 produces zero masks with multiple components and zero with interior holes, across all 90 cases**, because the sign of a distance field defines a closed, simply-connected region by construction. Phase 2's output has extra components in 12% of cases but with a median of **7 voxels** — threshold noise at tile seams. There is nothing for this post-processing to correct.
- **nnU-Net-style preprocessing** — also not the difference. Its plans target 0.4 mm isotropic spacing, which this dataset already is in all 90 cases, so **it resamples nothing**; the step that usually matters most in that pipeline does nothing here. Normalisation is per-image z-score, the same as ours.
- **Hybrid attention (USegNet-DS-HAS)** — reviewed, not ported. Its softmax normalises over channels, not space, so it is channel attention despite the paper describing spatial; the encoder's attention output is computed then discarded; and the reference implementation cannot run as published. More fundamentally it suppresses uninformative context in a whole-volume segmentor, and phase 2 already has its context restricted geometrically to a 3–5 mm band — `band10` losing Dice shows more context hurts this refiner.

---

## Inference cost

Phase 1 went 580 ms → **97 ms** per case with zero voxels changed (max difference 9.5e-07,
float32 against float64): resample on the GPU instead of CPU float64, using a
`grid_sample` with an affine-derived grid verified against `SpatialResample` to 3.6e-07.

Phase 2 went 1270 ms → 1067 ms. Full pipeline: **1.16 s** per case.







v1 ShellRefine	0.9457	3.202	1.063	5
nnU-Net (TTA)	0.9493	3.375	1.022	5
