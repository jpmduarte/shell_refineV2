# Final architecture: the target pipeline and how to get there

The aim for the journal paper: one pipeline that puts every head in a canonical clinical
frame (size, then pose) and reads the biometry there. It is built one block at a time, and
a block stays only if an ablation shows it helps.

## The target

```
[native volume V_raw] (x4)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ PASS 1: 3D U-Net trained on non-normalised heads          │
└──────────────────────────────────────────────────────────┘
       │
       ├─► coarse SDF ─► head volume V, centre c
       │                       │
       ▼                       ▼
[V_raw] ───────────────► resample: spacing ∝ (V / V0)^(1/3), centred on c
                               │
       ┌───────────────────────┘
       ▼
[normalised volume V_norm] (x4), Cellpose-style: every head at the size V0
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ PASS 2: one network, trained on normalised heads          │
│   STN: a small encoder predicts the rotation R (6D)       │
│        V_norm resampled with R  ─► clinical canonical space│
│   3D U-Net in the canonical space, three outputs:         │
│     A  SDF            continuous surface (the segmentation)│
│     B  polar head     radii in 64 directions: closes the  │
│                       shape across shadows and the sector │
│     C  landmarks      3D heatmaps ─► soft-argmax ─► points │
└──────────────────────────────────────────────────────────┘
       │
       ├─► SDF rotated back with R⁻¹ ─► mask in the native volume
       └─► landmarks + polar radii in the canonical space ─► biometry
           (BPD, OFD, HC in the standard planes; head volume)
```

Two design choices, both from what was measured:

- **The STN rotates the image, not the features.** Rotating features inside the U-Net
  would mean rotating every skip connection too. Rotating only the last features and then
  back would leave no layer working in the canonical space. Resampling the input
  (`grid_sample`, differentiable) at the start of pass 2 is what the oracle-pose test did:
  Dice 0.9571 against 0.9480 for size alone (fold 3). The STN learns that rotation inside
  the network instead of taking it from a separate step.
- **The STN is supervised; otherwise it learns whatever rotation helps the SDF, not the
  clinical one.** First by the head's own principal axes from the label (OFD, BPD, height),
  with a loss blind to each axis' sign, since that needs no new annotation. Then by the
  landmarks: the rotation that maps them to a template (Kabsch), and the landmark heatmaps
  matched in the canonical space. The landmarks add what the axes cannot: anterior or
  posterior, left or right.

## The steps, in order

| Step | Block | Status | Needs | Kept if |
|---|---|---|---|---|
| 1 | Pass 1, size normalisation, pass 2 (A) | done: Dice 0.9487 ens + TTA | — | — |
| 2 | **B: polar head** | running: seed 42, `runs/nnaug_x4normpolar_s42` | — | Dice ≥ size-only (3 seeds, Wilcoxon), or radii good enough for biometry |
| 3 | **C: landmarks** | next | landmark annotations on the 90 cases | point error small enough to define the pose |
| 4 | **STN pose in pass 2** | after 3 (axis-supervised version possible earlier) | the landmarks of step 3 | Dice approaches the oracle pose (0.9571) |
| 5 | **Biometry** | last | the clinical measurements of the exams, if available | agreement (Bland-Altman) within the clinical inter-observer range |

Each step is reported as an ablation against the previous one: size only, + polar,
+ landmarks, + pose.

## Open questions

- **Which landmarks** can be marked reliably at 19–24 weeks (thalami, cavum septi
  pellucidi, cerebellum, falx anterior/posterior, orbits?), and who annotates them.
- **The two groups of labels** (UID-named vs patient-named files) put the head's boundary
  ~0.5 mm apart (`size_normalization_approach/evaluation/pass1_error_results.md`); this
  lands directly in BPD and HC, so it has to be settled before step 5.
- **Cut heads** (80 of 90 leave the sector): the polar head and the landmarks are the way
  to complete the shape for biometry, but the labels stop at the sector's edge, so
  completion needs its own supervision (e.g. cutting the 7 whole heads artificially).
- **Validation outside our labels**: the 30 unlabelled volumes as an external test, and a
  blind re-annotation of ~10 cases for the annotation's own repeatability.
