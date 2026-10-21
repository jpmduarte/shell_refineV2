# Size-shift test: plain x4 against size-normalised x4

Does size normalisation make the method robust to head size? Each validation volume is
resampled so the head spans s times more voxels at the same 0.4 mm voxel, as if the fetus
were bigger or smaller, and both methods run exactly as deployed. Produced by
[size_shift_test.py](size_shift_test.py), once per fold and seed:

    python size_normalization/evaluation/size_shift_test.py --fold <0-4> --seed <42|43|44>

All 5 folds × 3 seeds: every case is predicted by the networks of the fold that held it
out. Dice at native resolution against the label resampled the same way (through its SDF),
no TTA. Per case, the mean over the 3 seeds; paired Wilcoxon over the 90 cases.

The normalised method knows the head's size only through pass 1. Its scale k = (V / V0)^(1/3)
is accepted within 0.5–2 (a head of about 16–1040 mL); otherwise, including when pass 1
finds no head, the method falls back to pass 1's own segmentation (the plain x4 mask).

| Head scale | Volume (median) | Plain x4 | Size-normalised x4 | Difference | p | Normalised better | Pass 1 rejected | Pass-1 volume error, median (signed mean) | k range |
|---|---|---|---|---|---|---|---|---|---|
| 0.60 (volume ×0.22) | 28 mL | 0.7138 | **0.7566** | **+0.0428** | 8e-13 | 68/90 | 94/270 (19 no head) | 26.2% (−34.8%) | 0.00–0.78 |
| 0.70 (×0.34) | 44 mL | 0.9049 | **0.9314** | **+0.0265** | 1e-13 | 77/90 | 5/270 (2 no head) | 8.5% (−10.0%) | 0.00–0.90 |
| 0.85 (×0.61) | 79 mL | 0.9401 | **0.9449** | +0.0048 | 2e-05 | 62/90 | 1/270 | 4.9% (−1.8%) | 0.46–1.09 |
| **1.00 (original)** | 128 mL | 0.9447 | **0.9462** | +0.0015 | 0.04 | 53/90 | 0 | 4.3% (−0.1%) | 0.65–1.27 |
| 1.15 (×1.52) | 195 mL | 0.9431 | **0.9462** | +0.0031 | 0.007 | 53/90 | 0 | 4.2% (−0.2%) | 0.78–1.44 |
| 1.30 (×2.20) | 282 mL | 0.9380 | **0.9460** | +0.0080 | 1e-06 | 64/90 | 0 | 5.3% (−1.0%) | 0.88–1.62 |
| 1.50 (×3.38) | 433 mL | 0.9254 | **0.9457** | **+0.0203** | 3e-12 | 76/90 | 0 | 6.6% (−0.9%) | 1.01–1.89 |

At the original size the two columns are exactly the main results' one-model figures
(0.9447 and 0.9462, [../../results.md](../../results.md)), which checks the test end to end. The
three seeds agree at every scale (normalised − plain, by seed: +0.042 to +0.044 at 0.6×,
0.000 to +0.003 at 1.0×, +0.018 to +0.024 at 1.5×).

## What it shows

- **The normalised method stays flat (0.945–0.946) from 0.85× to 1.5×**, where pass 1 is
  reliable: its network always sees the head at the reference size V0.
- **Plain x4 degrades both ways**: −0.0067 at 1.3×, −0.019 at 1.5×, −0.040 at 0.7×. Small
  heads hurt more, as they span few voxels at 1.6 mm.
- **At the original size the gain is small (+0.0015)**, which is why it looked modest in the
  cross-validation: the dataset has few heads far from the median. The value of the
  normalisation is its robustness to size; the further from the median, the larger it is.
- **Small heads expose pass 1.** Below 0.85× the plain network underestimates the volume
  (signed mean −10% at 0.7×, −35% at 0.6×) and sometimes finds no head at all. The
  normalised method still wins with the fallback, but it cannot be better than pass 1 lets
  it: on the cases whose scale was accepted it reaches 0.944 even at 0.6×.

## Without the fallback

Scoring a rejected scale as a failure (Dice 0) instead: 0.6156 at 0.6× (below plain, 94
of 270 rejected), 0.9276 at 0.7×, 0.9422 at 0.85×; unchanged from 1.0× up. The 0.5–2
window is about the GPU, not about plausibility: at 0.6× the true volume of many heads is
below 16 mL (k ≈ 0.4), and they are rejected although they could be normalised. A tiny but non-zero
estimate (k ≈ 0.05) asks for a grid that does not fit in 12 GB, which is what the window
prevents.

## Caveats

- **The shift is synthetic.** Resampling changes the head's size but also scales the
  speckle and texture (coarser and blurrier above 1×, finer with some aliasing below); a
  real bigger fetus has bigger anatomy with the same speckle in mm. The test shows
  robustness to this kind of shift, not to natural anatomical variation.
- The networks were trained on heads of 35–291 mL; 0.6× and 1.5× go beyond that range on
  purpose.
- The volume error here is on the volume itself (4.3% median at the original size). The
  1.4% quoted in results.md and docs/pipeline.md is the error of the linear scale k, not of
  the volume.
- The pipeline itself ([infer.py](../infer.py) `segment`) now has the same guard:
  outside 0.5 ≤ k ≤ 2 it returns pass 1's segmentation (see pass1_volume_results.md).
- The geometry assumes the cache's: RAS, isotropic 0.4 mm voxels. The volume scales as s³
  as intended (median 128 mL × s³ matches the measured medians at every scale).
