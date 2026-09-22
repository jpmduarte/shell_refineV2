# Phase 2 refinement: what was tried, and what it did

Fetal head segmentation in 3D ultrasound, in two stages. Phase 1 regresses a signed
distance field on a downsampled volume and gives a boundary. Phase 2 sharpens that
boundary at native resolution. This is a record of everything tried, the implementation
choices each rests on, and what each measured.

> **These numbers are from volume-level folds and are superseded.**
> 90 volumes come from 82 patients, and the original splits shuffled case ids with no
> notion of patient, so five patients had scans on both sides of a fold boundary. The
> effect was measured at **+0.0011 Dice** — smaller than the ~0.005 seed noise, and the
> relative ordering of the arms below should hold — but every absolute figure here is
> optimistic by that much. The splits are now patient-grouped and results are being
> regenerated. Kept because the ordering, the negative results and the diagnosis cost
> weeks and do not depend on the split.

---

## The dataset

90 volumes from 82 patients, GE Voluson, gestational age 19+1 to 24+4 weeks, 0.4 mm
isotropic, roughly 216–285 × 221–292 × 169–205 voxels, manual reference masks. Five-fold
cross-validation, 72 train / 18 validation.

At 0.4 mm isotropic the geometry choices have physical meaning worth stating: a **5 mm
band is ±12.5 voxels**, and a **128×128×48 slab is 51.2 × 51.2 × 19.2 mm**. The slab is
not an arbitrary patch size — it is roughly 19 mm of native-resolution context across
the boundary and 51 mm along it.

## The setup everything shares

**Phase 1.** A 1.4M-parameter 3D U-Net on the x8 volume (~3.2 mm per voxel), padded to a
fixed 48³ cube, predicting a signed distance field truncated at 10 mm and normalised to
[-1, 1]. Trained 250–300 epochs, L1 loss, lr 5e-4.

Two choices worth naming. The checkpoint is selected on **boundary Dice**, not MAE: MAE
is dominated by the saturated far field, where the field is ±1 and easy, so it can
improve while the boundary moves the wrong way. And the input is **padded** to 48³
rather than resized, because resizing would resample an already-isotropic cached grid a
second time.

**The prior phase 2 refines.** Each case's prior comes from the fold that held that case
out, so phase 2 never sees a prior produced by a model that trained on the same case.
Without this, phase 2 trains against priors cleaner than the ones it meets at inference
and its measured gain is inflated.

**The band.** `|sdf_prior| < 5 mm`, about 2–3 million voxels per case. Phase 2 may only
rewrite inside it; everywhere else keeps phase 1's answer.

**Evaluation.** Native resolution, against the never-resampled label. `coarse` is phase
1 alone, `refined` is after phase 2, plus HD95 and ASD in mm.

---

## The arms

All on fold 0 against a 0.9417 prior unless stated.

### Cartesian cubes — 64³, 96³, 128³

Patch counts set so every arm sees the same ~2.1M supervised voxels per case: 8 patches
at 64³, 2 at 96³, 1 at 128³. Without that, a bigger patch would win by seeing more data.

Patch centres drawn with **Poisson-disk rejection** (minimum separation of half a patch).
Plain uniform draws put two of eight centres within a voxel of each other in 56% of
epochs; rejection cuts that to 8% without changing the overall spread.

| | Δ Dice | HD95 |
|---|---|---|
| 64³ (3 seeds) | +0.0018 ± 0.0003 | 3.51 |
| 96³ | +0.0005 | 3.63 |
| 128³ | +0.0009 | 3.57 |

### Band-masked loss

Score only band voxels rather than the whole patch. The network still **sees** the whole
patch — it needs the context to decide — it is only **scored** where refinement may act.
v1 had tested the weak form (down-weighting to a floor) and found nothing; this is the
strong form.

Masked +0.0021, unmasked +0.0010.

### Slabs — patches shaped like the band

A cube is the wrong shape for a shell, which is thin across and extended along. A slab
of `long × long × thin` matches it: more tangential context per voxel, and a higher
share supervised (a 64³ cube is 44% band, a 128³ cube only 31%).

The normal varies over the surface, so an axis-aligned slab cannot match it everywhere.
Rather than rotate the volume into the local frame — which would interpolate, in a
pipeline whose point is not to disturb the metric — each patch picks **whichever of the
three axes is nearest the local normal** and makes that one thin. Worst case is a normal
along the cube diagonal, ~54.7° off, where effective thickness is 25/cos(54.7°) ≈ 43
voxels: worse than aligned, still better than a cube.

| thickness | Δ Dice | HD95 |
|---|---|---|
| 128×128×16 | +0.0016 | 3.54 |
| 128×128×32 | +0.0034 | 3.43 |
| **128×128×48** | **+0.0039** | **3.40** |
| 128×128×64 | +0.0016 | 3.56 |
| 64×64×32 | +0.0006 | 3.61 |

A clear optimum at 48 (±9.6 mm), which matches the measured displacement tail
independently: p99 of 5.79 mm and worst case 10.61 mm.

The 64×64×32 arm keeps the thin section but drops the tangential extent and loses nearly
all the gain — both are needed. *Caveat:* it ran with half the supervised voxels of the
others (1.0M against 2.1M), an arithmetic slip when the job was written.

### Residual SDF — regress a correction, not a class

Predict `Δ` and output `clamp(prior + Δ, -1, 1)`. The head is `0.5·tanh`, bounding the
correction at the band half-width, and the final layer is zero-initialised so training
starts from "change nothing".

Five runs, all between −0.0001 and +0.0007. A clean negative. It did make the model
conservative — 274k flips against the slab's 1.3M — but the flips it makes are right
**51.6%** of the time, which is chance. Restraint without judgement.

### Polar — the band as a slab by construction

In spherical coordinates anchored on the prior's surface, the band *is* an axis-aligned
slab everywhere, with the radial axis along the normal by definition. The output is one
radius per direction, so it cannot have holes, islands or a doubled boundary.

Four things measured before building anything:

- **Star-convexity.** 0.02% of rays cross the surface more than once, by 0.19 mm on
  average, against an ASD of ~1.1 mm. The representation can express this anatomy.
- **Round-trip cost.** 0.9990 Dice at 128×256×128, but only 0.9940 at 64×128×64. Since
  phase 2's whole contribution is ~+0.004, the coarse grid would have swallowed more
  than there was to win.
- **Shell coverage.** ±5 mm leaves 2.83% of directions unreachable; **±10 mm** leaves
  0.02%.
- **Ceiling.** Rasterising the *true* radii gives **+0.0686** Dice.

φ wraps and θ does not, so convolutions pad circularly on one axis and by replication on
the other. The encoder strides the radial axis away and the decoder works on the sphere,
so the output has one dimension fewer than the input — 177k parameters against 1.4M
elsewhere.

Result: **+0.0033** Dice, but **HD95 3.29 mm and ASD 1.08 mm, the best surface metrics of
any arm**. It predicts a surface, so it gets the surface right; Dice counts volume, which
is dominated by an interior every arm already has.

### Capacity, encoder, context, conditioning

- **Width 32 → 64 → 96** (177k → 1.58M parameters): null. Training loss moved 5% across
  a 9× parameter range; validation MAE stayed between 1.136 and 1.155 mm.
- **`keep_r=6`** (stop the radial striding early, fold the rest into channels): null.
  It has *fewer* parameters than the baseline, so it separates architecture from
  capacity — and neither mattered.
- **Global context**: patch 32×64 → 64×128 → 128×256 gave +0.0033 → **+0.0039** →
  +0.0029. More context helps, but the whole sphere is worse than a quarter of it.
- **FiLM** (per-case conditioning on channel statistics): +0.0027 and +0.0025. No help.

### Joint training

Train both phases together, so phase 1 can learn to put the boundary where phase 2 can
act. The normal pipeline is not differentiable, so phase 1's field is upsampled inside
the graph. `F.interpolate` would be **wrong** — it scales each axis by its own shape
ratio, which is not guaranteed isotropic. Instead the sampling grid is built from the
affines (constants, needing no gradient) and applied with `grid_sample`: verified
against `SpatialResample` at **3.6e-07**.

The band **weights** the loss rather than selecting voxels, since a hard threshold would
cut the gradient to phase 1. An SDF term holds phase 1 honest: without it, it is free to
distort the field into something only this phase 2 can read.

| | coarse | refined | HD95 |
|---|---|---|---|
| warm start | 0.9319 | 0.9347 | 3.84 |
| from scratch | 0.8842 | 0.8881 | 6.29 |

**It degraded phase 1.** The coarse baseline fell from 0.9417 to 0.9319 and 0.8842.
Phase 1 trained alone is better than phase 1 trained jointly.

### Adaptive slabs — tested and dropped

The idea: near the crown an axial slice cuts the band as a filled disc, through the
middle as a hollow ring, so the patch should change shape with where it sits.

Measured instead: the band's two tangential extents are within 4% of each other at
**100%** of sites at 16 and 32 voxels, and 95.2% at 64. Zero ring-like sites at any
scale. Disc-versus-ring is a property of cutting a shell with axial planes, not of the
shell — locally, every point of a shell looks like every other. Not built.

### Sparse convolution — built, measured, parked

No library works on this GPU: spconv, torchsparse and MinkowskiEngine all predate
Blackwell (sm_120). Written from plain torch ops — coordinate hashing, `searchsorted`
neighbour lookup, per-offset gather and matmul — and verified exact against dense
`Conv3d`. **1.2× slower than dense tiling** despite touching 6× fewer voxels: 27 unfused
gather-matmul launches per layer cost more than the voxel reduction saves. Kept in
`phase2/sparse_experiment/`.

---

## Five-fold validation of the best arm

The slab at 48, over a 300-epoch prior, five folds, one seed:

| fold | coarse | refined | Δ Dice | HD95 | ASD |
|---|---|---|---|---|---|
| 0 | 0.9418 | 0.9435 | +0.0017 | 3.47 | 1.122 |
| 1 | 0.9408 | 0.9437 | +0.0029 | 3.10 | 1.074 |
| 2 | 0.9416 | 0.9438 | +0.0022 | 3.38 | 1.077 |
| 3 | 0.9331 | 0.9383 | **+0.0052** | 3.90 | 1.210 |
| 4 | 0.9432 | 0.9442 | +0.0010 | 3.47 | 1.113 |
| **mean** | **0.9401** | **0.9427** | **+0.0026 ± 0.0015** | 3.46 | 1.119 |

The gain correlates with prior quality *within* the experiment: fold 3 has the worst
prior and the largest gain, fold 4 the best prior and the smallest. That is the
saturation relation below, observed across cases rather than across training budgets.

This corrects the +0.0039 that had been treated as the best result — that was one fold,
one seed. The honest figure for this arm is **+0.0026 ± 0.0015**.

---

## The control that changed the reading

Every arm adding about +0.004 admits two explanations: phase 2 cannot do more, or there
is nothing left at that quality for it to do. They are separated by making the prior
worse on purpose — training phase 1 for fewer epochs, which leaves architecture, data
and geometry untouched.

| prior | coarse Dice | Δ Dice | ASD coarse → refined |
|---|---|---|---|
| 15 epochs | 0.8546 | **+0.0384** | 2.99 → 2.12 |
| 40 epochs | 0.9017 | **+0.0278** | 1.97 → 1.40 |
| 100 epochs | 0.9338 | +0.0031 | 1.31 → 1.24 |
| 250 epochs | 0.9417 | +0.0021 | 1.17 → 1.12 |

**Phase 2 works.** Over a prior at 0.855 it recovers +0.0384, eighteen times what it
manages over the full one. What happens at 0.9417 is saturation, not incapacity.

The collapse is abrupt rather than gradual: +0.0278 at 0.9017 falls to +0.0031 at
0.9338. Below that the residual error is large and locally structured, and a refiner
finds it; above it what remains is boundary ambiguity, which no local model resolves.

This retracts an earlier conclusion that band refinement has an intrinsic ceiling here.
It has a *useful range*, and 0.9417 is above it.

---

## Things that did not work, recorded so they are not retried

**SWA.** Averaging the last 60 of 300 epochs' weights, with a constant rate during
averaging, against the same runs' best-epoch and last-epoch checkpoints on five folds:

| selection | Dice | HD95 | ASD |
|---|---|---|---|
| **best epoch** | **0.9401** | **3.51** | **1.174** |
| SWA | 0.9385 | 3.62 | 1.203 |
| last epoch | 0.9366 | 3.68 | 1.241 |

Best-epoch wins on every fold and every metric. Losing 5/5 is not what noise looks like.

**Longer training.** 300 epochs instead of 250 is worth about +0.002 Dice. Validation
loss falls 0.00005 per epoch over the last forty, so the curve has flattened.

**Gradient-difference loss.** v1's best phase 1 was L1 plus a gradient term at 0.1,
worth +0.0017 there. Here, five folds:

| | Dice | HD95 | ASD |
|---|---|---|---|
| L1 | 0.9401 ± 0.0036 | 3.51 | 1.174 |
| L1 + gd | 0.9411 ± 0.0056 | 3.50 | 1.154 |

+0.0010, and **not consistent**: it gains on four folds and loses 0.0030 on fold 3, the
hard one, with the between-fold spread rising from 0.0036 to 0.0056. ASD is the more
reliable signal (better on 4 of 5). It does not explain the gap to v1.

**Threshold.** Swept 0.10–0.90 on all five folds. 0.55 beats 0.50 by **+0.0001** on four
folds and ties on the fifth; the curve is flat from 0.45 to 0.60 and falls sharply
outside. Not worth changing.

---

## Why none of it moves further

Ruled out by measurement:

- **Capacity.** 9× the parameters changes validation MAE by 0.02 mm.
- **Architecture.** Six patch geometries, two encoder variants, FiLM, all within noise.
- **Representation.** Binary, residual SDF and polar radii all land in the same place.
- **Information.** The network memorises four cases to **0.186 mm**, against 1.008 mm
  for a constant. The answer *is* in the input.

What remains is generalisation between cases:

| training cases | train MAE |
|---|---|
| 4 | 0.375 mm |
| 12 | 0.644 mm |
| 36 | 0.723 mm |
| 72 | **1.302 mm** |

At 72 cases the model is worse than a constant predictor **on its own training data**.
That is not overfitting — it is a model failing to fit a growing training set, which
happens when cases demand mutually inconsistent mappings. Confirmed in closed form: a
linear predictor fitted on one case scores 0.954 mm there, 1.384 mm elsewhere, against a
constant's 1.045 mm. **What is learned on one case actively misleads on the next.**

Phase 1 does not have this problem: its validation tracks its training across all five
folds. It sees the whole volume and learns shape, which is consistent between cases.
Phase 2 sees local texture, which is not.

---

## Band coverage — the open lead

Measured on ground-truth **surface voxels**, not volume:

| band | surface covered | worst case | cases below 99% |
|---|---|---|---|
| 3.0 mm | 92.42% | 70.56% | 81/90 |
| **5.0 mm (current)** | **98.35%** | **83.52%** | **36/90** |
| 7.5 mm | 99.76% | 93.76% | 7/90 |
| 10.0 mm | 100.00% | 100.00% | 0/90 |

Distance from the prior's surface to the true one: median 0.95 mm, p95 3.43 mm, p99
4.82 mm, worst 9.93 mm.

**The 5 mm band leaves 1.65% of the true surface unreachable**, and in the worst case
16% of it. An earlier measurement put this at 0.065%, but that counted *volume*, which
is dominated by the easy interior; surface is the quantity the ceiling depends on. This
is the one untested change with a measured reason to expect a gain.

---

## Where this leaves the method

| | Dice | HD95 (mm) | ASD (mm) | params | inference |
|---|---|---|---|---|---|
| nnU-Net 250ep, with TTA | **0.9493 ± 0.0048** | 3.375 | **1.022** | 88.2 M | 5.63 s net / 6.23 s total |
| nnU-Net, no TTA | not evaluated | — | — | 88.2 M | 0.75 s net / 1.34 s total |
| v1 ShellRefine | 0.9457 ± 0.0045 | **3.202** | 1.063 | 2.80 M | 1.05 s net / 5.81 s total |
| **v2 (slab 48)** | 0.9427 ± 0.0024 | 3.46 | 1.119 | **2.80 M** | **1.16 s total** |

Same folds for nnU-Net and v1; v2's folds are its own, which is worth remembering given
that the nnU-Net's own folds range from 0.9407 to 0.9549.

Nearly all of the nnU-Net's inference cost is mirror TTA: 5.63 s with, 0.75 s without,
a 7.5× ratio consistent with its 8 passes. Its quality was only evaluated **with** TTA,
so "faster without it" cannot be claimed against a number measured with it.

### Inference optimisation

Phase 1 went from 580 ms to **97 ms** per case, with zero voxels changed (max difference
9.5e-07, which is float32 against float64):

- resample on the GPU instead of CPU float64
- the sampling grid built on the GPU — 203 ms of the 580 was that alone
- native shape read from the `.npy` header instead of decompressing 78 MB (50 → 0.5 ms)

Phase 2 went from 1270 ms to **1067 ms** by writing crops uncompressed. Loading a crop
costs 454 ms compressed against 142 ms uncompressed — more than the network itself,
which is 6 ms per tile.

---

## Caveats

- **Volume-level folds** with 13.3% of cases patient-leaked, worth +0.0011. Superseded.
- **Seeds.** Most arms have one; measured seed noise is ~0.005, larger than the gap
  between most arms in the tables.
- **Timings** are fp32, single run, no warmup, sometimes with another job on the GPU.
- **The 64×64×32 slab** ran with half the intended supervised voxels.
- **The polar arm** spends 2.5 of its 2.7 s per case in CPU resampling and rasterising;
  the network itself is 0.175 s. Both are GPU-portable.
