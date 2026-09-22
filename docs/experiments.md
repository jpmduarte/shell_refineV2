# Phase 2 refinement: what was tried, and what it did

Fetal head segmentation in 3D ultrasound, in two stages. Phase 1 regresses a signed
distance field on a downsampled volume and gives a boundary. Phase 2 is supposed to
sharpen that boundary at native resolution. This is a record of twelve attempts at
phase 2, the implementation choices each one rests on, and what each measured.

The short version: phase 1 alone scores 0.9417 Dice, and the best phase 2 adds
**+0.0039**. Every arm lands between +0.0000 and +0.0039, and the reason is diagnosed
at the end.

---

## The setup everything shares

**Phase 1.** A 1.4M-parameter 3D U-Net on the x8 volume (~3.2mm per voxel), padded to a
fixed 48³ cube, predicting a signed distance field truncated at 10mm and normalised to
[-1, 1]. Trained 250 epochs, L1 loss, lr 5e-4, five folds.

Two choices worth naming. The checkpoint is selected on **boundary Dice**, not MAE: MAE
is dominated by the saturated far field, where the field is ±1 and easy, so it can
improve while the boundary itself moves the wrong way. And the input is **padded** to
48³ rather than resized, because resizing would resample an already-isotropic cached
grid a second time.

**The prior phase 2 refines.** Each case's prior comes from the fold that held that case
out, so phase 2 never sees a prior produced by a model that trained on the same case.
Without this, phase 2 trains against priors cleaner than the ones it meets at inference
and its measured gain is inflated.

**The band.** `|sdf_prior| < 5mm`, about 2-3 million voxels per case. Phase 2 may only
rewrite inside it; everywhere else keeps phase 1's answer. Measured: the ground truth
disagrees with the prior outside the band in **0.065%** of voxels, so the band is not
what limits the result.

**Evaluation.** Native resolution, against the never-resampled label. `coarse` is phase
1 alone, `refined` is after phase 2. Also HD95 and ASD in mm, which are surface metrics
and see things Dice does not.

---

## The arms

### 1. Cartesian cubes — 64³, 96³, 128³

The baseline approach, inherited from v1. Sample cubic patches centred on band voxels,
train a U-Net to classify each voxel inside or outside.

*Implementation.* Patch counts are set so every arm sees the same ~2.1M supervised
voxels per case: 8 patches at 64³, 2 at 96³, 1 at 128³. Without that, a bigger patch
would win simply by seeing more data.

Patch centres are drawn with **Poisson-disk rejection** (minimum separation of half a
patch). Plain uniform draws put two of eight centres within a voxel of each other in
56% of epochs; rejection cuts that to 8% without changing the overall spread.

| | Δ Dice | HD95 |
|---|---|---|
| 64³ (3 seeds) | +0.0018 ± 0.0003 | 3.51 |
| 96³ | +0.0005 | 3.63 |
| 128³ | +0.0009 | 3.57 |

### 2. Band-masked loss

Score only band voxels rather than the whole patch. A 64³ cube around a ~25-voxel-thick
shell is mostly deep interior and exterior, which the prior already gets right.

*Implementation.* The band travels with the patch and weights the loss. The network
still **sees** the whole patch — it needs the surrounding context to decide — it is only
**scored** where refinement is allowed to act. v1 had tested the weak form of this
(down-weighting to a floor) and found nothing; this is the strong form.

Masked +0.0021, unmasked +0.0010. A real but small effect, and the masked arm was the
better of the two at every operating point on the flip-accuracy curve.

### 3. Slabs — patches shaped like the band

A cube is the wrong shape for a shell, which is thin across and extended along. A slab
of `long × long × thin` matches it: more tangential context for the same voxel count,
and a much higher share of it supervised (a 64³ cube is 44% band, a 128³ cube only 31%).

*Implementation.* The band's normal varies over the surface, so an axis-aligned slab
cannot match it everywhere. Rather than rotate the volume into the local frame — which
would interpolate, in a pipeline whose point is not to disturb the metric — each patch
picks **whichever of the three axes is nearest the local normal** and makes that one
thin. Worst case is a normal along the cube diagonal, ~54.7° off, where the effective
thickness is 25/cos(54.7°) ≈ 43 voxels: worse than aligned, still better than a cube.

| thickness | Δ Dice | HD95 |
|---|---|---|
| 128×128×16 | +0.0016 | 3.54 |
| 128×128×32 | +0.0034 | 3.43 |
| **128×128×48** | **+0.0039** | **3.40** |
| 128×128×64 | +0.0016 | 3.56 |
| 64×64×32 | +0.0006 | 3.61 |

A clear optimum at 48, which is ±9.6mm — and independently, the displacement from the
prior to the true surface has a p99 of 5.79mm and a worst case of 10.61mm. The best
thickness is the one that covers that tail.

The 64×64×32 arm matters: it keeps the thin section but drops the tangential extent, and
loses nearly all the gain. Both are needed.

*A caveat.* That arm was run with half the supervised voxels of the others (1.0M against
2.1M) — an arithmetic slip when the job was written. The conclusion is still supported
by 128³ losing while having the extent, but it is not the clean comparison it should be.

### 4. Residual SDF — regress a correction, not a class

Instead of classifying voxels, predict `Δ` and output `clamp(prior + Δ, -1, 1)`.

*Implementation.* The head is `0.5 · tanh`, bounding the correction at half the
truncation — which is the band half-width, so a larger correction could never be
applied. The final layer is **zero-initialised**, so training starts from "change
nothing" rather than from noise.

| | Δ Dice |
|---|---|
| cube64, 2 seeds | +0.0000, +0.0007 |
| cube128, 3 variants | −0.0001 to −0.0000 |

A clean negative result. It did make the model conservative — it flips 274k voxels
against the slab's 1.3M — but the flips it does make are right **51.6%** of the time,
which is chance. Restraint without judgement.

### 5. Polar — the band as a slab by construction

In spherical coordinates anchored on the prior's surface, the band *is* an axis-aligned
slab everywhere, with the radial axis along the normal by definition. The output is one
radius per direction, so it cannot have holes, islands or a doubled boundary.

*Four things measured before building anything:*

- **Star-convexity.** 0.02% of rays cross the surface more than once, by 0.19mm on
  average, against an ASD of ~1.1mm. The representation can express this anatomy.
- **Round-trip cost.** Resampling to (θ, φ, r) and back scores 0.9990 Dice at
  128×256×128, but only 0.9940 at 64×128×64. Since phase 2's whole contribution is
  ~+0.004, the coarse grid would have swallowed more than there was to win.
- **Shell coverage.** A ±5mm shell leaves 2.83% of directions unreachable; **±10mm**
  leaves 0.02%, matching the cartesian band's own miss rate.
- **Ceiling.** Rasterising the *true* radii gives **+0.0686** Dice. The representation
  is not the constraint.

*Implementation.* φ wraps and θ does not, so convolutions pad circularly on one axis and
by replication on the other; a seam at φ=0 would otherwise be a surface the network has
to learn around. The encoder strides the radial axis away and the decoder works on the
sphere, so the output has one dimension fewer than the input — 177k parameters against
1.4M elsewhere. Shells are built once per case and cached: 1.9s each, too slow to repeat
every epoch.

Result: **+0.0033** Dice, but **HD95 3.29mm and ASD 1.08mm — the best surface metrics of
any arm**. It predicts a surface, so it gets the surface right; Dice counts volume,
which is dominated by an interior every arm already has.

### 6. Capacity and encoder sweeps

Width 32 → 64 → 96 (177k → 1.58M parameters), and `keep_r=6`, which stops the radial
striding early and folds what is left into channels.

Both null. Training loss moved 5% across a 9× parameter range; validation MAE stayed
between 1.136 and 1.155mm. `keep_r=6` has **fewer** parameters than the baseline, so it
separates architecture from capacity — and neither mattered.

### 7. Global context

The polar patch covers a sixteenth of the sphere. A whole-case property — like the
prior's overall displacement — is invisible at that scale.

| patch | Δ Dice | HD95 |
|---|---|---|
| 32×64 (baseline) | +0.0033 | 3.29 |
| **64×128** | **+0.0039** | **3.27** |
| 128×256 (whole sphere) | +0.0029 | 3.31 |

More context helps, but the whole sphere is worse than a quarter of it.

### 8. FiLM — per-case conditioning

Each block gets a scale and shift computed from the patch's own channel statistics, so
the network can modulate itself per case instead of learning one mapping for all.

*Implementation.* Only mean and std of the input channels, which describe appearance
without leaking anything about the answer.

+0.0027 and +0.0025. No help.

### 9. Joint training

Train both phases together, so phase 1 can learn to put the boundary where phase 2 can
act, instead of optimising a field in isolation.

*Implementation, and the one real subtlety.* The normal pipeline is not differentiable:
predict, crop, threshold, cut patches. Here phase 1's field is upsampled inside the
graph. `F.interpolate` would have been the easy way and is **wrong** — it scales each
axis by its own shape ratio, which is not guaranteed isotropic, and isotropy is the
whole point of the cached grid. Instead the sampling grid is built from the affines
(constants, so they need no gradient) and applied with `grid_sample` inside the graph.
Verified against the pipeline's own `SpatialResample`: **max difference 3.6e-07**.

The band **weights** the loss rather than selecting voxels, since a hard threshold would
cut the gradient to phase 1. An SDF term holds phase 1 honest: without it, it is free to
distort the field into something only this particular phase 2 can read.

| | coarse | refined | HD95 |
|---|---|---|---|
| warm start | 0.9319 | 0.9347 | 3.84 |
| from scratch | 0.8842 | 0.8881 | 6.29 |

**It degraded phase 1.** The coarse baseline fell from 0.9417 to 0.9319 and 0.8842.
Phase 2's contribution was the same as always, but on a worse base. Phase 1 trained
alone is better than phase 1 trained jointly.

### 10. Adaptive slabs — tested and dropped

The idea: near the crown an axial slice cuts the band as a filled disc, through the
middle as a hollow ring, so the patch should change shape with where it sits.

Measured instead: the band's two tangential extents are within 4% of each other at
**100%** of sites at 16 and 32 voxels, and 95.2% at 64. Zero ring-like sites at any
scale. Disc-versus-ring is a property of cutting a shell with axial planes, not of the
shell — locally, every point of a shell looks like every other. Not built.

### 11. Sparse convolution — built, measured, parked

Convolve only on band voxels instead of dense tiles.

No library works on this GPU: spconv, torchsparse and MinkowskiEngine all predate
Blackwell (sm_120) and their newest wheels have no kernel for it. So it was written from
plain torch ops — coordinate hashing, `searchsorted` neighbour lookup, per-offset
gather and matmul — and verified exact against dense `Conv3d`.

**It is 1.2× slower than dense tiling**, despite touching 6× fewer voxels: 27 unfused
gather-matmul launches per layer cost more than the voxel reduction saves. A real
negative result about sparse convolution in this regime, kept in
`phase2/sparse_experiment/`.

---

## Everything against the baseline

Phase 1 alone: **Dice 0.9417, HD95 3.54mm, ASD 1.17mm, head circumference error 3.31mm.**

| arm | Dice | Δ | HD95 | ASD | HC err |
|---|---|---|---|---|---|
| polar, context 64×128 | 0.9457 | +0.0039 | **3.27** | **1.07** | — |
| slab 128×128×48 | **0.9457** | +0.0039 | 3.40 | 1.09 | 3.55 |
| slab 128×128×32 | 0.9452 | +0.0034 | 3.43 | 1.10 | 3.42 |
| polar (2 seeds) | 0.9451 | +0.0033 | 3.29 | 1.08 | 6.51 |
| polar keep_r=6 | 0.9450 | +0.0033 | 3.29 | 1.08 | — |
| polar FiLM | 0.9445 | +0.0027 | 3.35 | 1.09 | — |
| cube64 masked | 0.9438 | +0.0021 | 3.43 | 1.12 | **3.33** |
| cube64 (3 seeds) | 0.9435 | +0.0018 | 3.51 | 1.13 | 3.90 |
| slab ×16 / ×64 | 0.9433 | +0.0016 | 3.55 | 1.13 | — |
| cube128 | 0.9427 | +0.0009 | 3.57 | 1.15 | — |
| cube96 | 0.9423 | +0.0005 | 3.63 | 1.15 | — |
| residual SDF | 0.9418 | +0.0000 | 3.53 | 1.17 | 3.50 |
| joint, warm | 0.9347 | — | 3.84 | 1.29 | 5.02 |
| joint, cold | 0.8881 | — | 6.29 | 2.09 | 11.18 |

Head circumference is measured on the largest axial cross-section, with marching squares
rather than by counting boundary voxels, which would follow the voxel staircase and
overstate the length.

**No arm improves head circumference.** All make it worse, and the polar arm — best on
HD95 and ASD — is worst here, with a +6.42mm bias. HD95 and ASD are symmetric and
absolute; circumference accumulates a consistent bias instead of cancelling it. So the
hope that Dice was simply the wrong instrument does not survive contact with the
clinical measurement.

---

## Why none of it moves

Ruled out by measurement, not by argument:

- **Reachability.** The truth lies outside the band in 0.065% of voxels.
- **Capacity.** 9× the parameters changes validation MAE by 0.02mm.
- **Architecture.** Six patch geometries, two encoder variants, FiLM, all within noise.
- **Representation.** Binary, residual SDF and polar radii all land in the same place.
- **Information.** The network memorises four cases to **0.186mm**, against 1.008mm for
  a constant. The answer *is* in the input.

What remains is generalisation between cases, and it is measured directly:

| training cases | train MAE |
|---|---|
| 4 | 0.375mm |
| 12 | 0.644mm |
| 36 | 0.723mm |
| 72 | **1.302mm** |

At 72 cases the model is worse than a constant predictor **on its own training data**.
That is not overfitting, which would be a good training fit and a poor validation one.
It is a model failing to fit a growing training set, which happens when cases demand
mutually inconsistent mappings.

Confirmed in closed form: fit a linear predictor per case and score it elsewhere.

| | MAE |
|---|---|
| on the case it was fitted to | 0.954mm |
| on every other case | 1.384mm |
| predicting a constant | 1.045mm |

**A predictor fitted on one case does worse than a constant on another.** What is learned
on one case actively misleads on the next.

Part of it is a per-case bias: each prior is displaced by its own amount, −1.5mm to
+1.5mm. But removing it perfectly is worth only 1.840 → 1.648mm rms, about 10% — the
error is dominated by the spread *within* each case, not by the offset between them.

And phase 1 does not have this problem: its validation tracks its training across all
five folds. It sees the whole volume and learns shape, which is consistent between
cases. Phase 2 sees local texture, which is not.

---

## The control that changed the reading

Every arm adding about +0.004 over a 0.9417 prior admits two explanations: phase 2
cannot do more, or there is nothing left at that quality for it to do. They are
separated by making the prior worse on purpose — training phase 1 for fewer epochs,
which leaves architecture, data and geometry untouched.

| prior | coarse Dice | Δ Dice | ASD coarse → refined |
|---|---|---|---|
| 15 epochs | 0.8546 | **+0.0384** | 2.99 → 2.12 |
| 40 epochs | 0.9017 | **+0.0278** | 1.97 → 1.40 |
| 100 epochs | 0.9338 | +0.0024 | 1.31 → 1.26 |
| 250 epochs | 0.9417 | +0.0021 | 1.17 → 1.12 |

**Phase 2 works.** Over a prior at 0.855 it recovers +0.0384, eighteen times what it
manages over the full one, and takes almost a millimetre off ASD. What happens at 0.9417
is saturation, not incapacity.

The collapse is abrupt rather than gradual: +0.0278 at 0.9017 falls to +0.0024 at
0.9338. Something changes between those two qualities. Below it the residual error is
large and locally structured, and a refiner finds it; above it what remains is boundary
ambiguity, which is what the within-case spread of 1.537mm measures and which no local
model resolves.

This retracts the conclusion the diagnosis section originally drew — that band
refinement has an intrinsic ceiling here. It has a *useful range*, and 0.9417 is above
it. It also means improving phase 1 moves the pipeline further from where phase 2
helps, not closer.

## Where this leaves the method

Phase 1 delivers 0.9417 Dice on fold 0 (0.9380 ± 0.0240 across all five) in 0.116s.
Phase 2 adds at most +0.0039 over that prior for roughly 1s, and makes head
circumference worse in every arm. But it adds +0.0384 over a prior at 0.855, so the
limit is where the prior already is, not what the refiner can do.

## Two things that did not work, recorded so they are not retried

**SWA.** Boundary Dice still swings visibly from epoch to epoch past 200, which is an
argument that picking the best of 250 is partly luck. Averaging the weights of the last
60 epochs, over 300 epochs with a constant rate during averaging, was measured against
the same runs' best-epoch and last-epoch checkpoints on all five folds:

| selection | Dice | HD95 | ASD |
|---|---|---|---|
| **best epoch** | **0.9401** | **3.51** | **1.174** |
| SWA | 0.9385 | 3.62 | 1.203 |
| last epoch | 0.9366 | 3.68 | 1.241 |

SWA lands between the two, and best-epoch wins on every fold and every metric. Losing
5/5 is not what noise looks like, so the validation signal is real and worth selecting
on. Selection stays as it is.

**Longer training.** 300 epochs instead of 250 is worth about +0.002 Dice. The curves
said as much beforehand — validation loss falls 0.00005 per epoch over the last forty —
and the extra epochs were there for SWA to average over rather than for their own sake.

The 300-epoch checkpoints are kept but **not** adopted as the base: every phase 2 result
here rests on priors from the 250-epoch models, and swapping the base would make thirty
measured arms incomparable without rerunning them.

## Caveats

- **Fold 0 only.** Every phase 2 number is one fold. Phase 1 ran five.
- **Seeds.** Some arms have two or three, most have one. Measured seed noise is ~0.005 —
  larger than the gap between most arms in the table.
- **Timings** are fp32, single run, no warmup, sometimes with another job on the GPU.
  Fine for spotting regressions, not for publication.
- **The 64×64×32 slab** ran with half the intended supervised voxels.
- **The polar arm** spends 2.5 of its 2.7s per case in CPU resampling and rasterising;
  the network itself is 0.175s. Both are GPU-portable, and were left alone until the
  approach justified the work.
