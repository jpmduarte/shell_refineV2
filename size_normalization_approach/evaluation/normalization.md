# Does size normalisation help, and where?

The size-normalised x4 method beats plain x4 by +0.0015 Dice in cross-validation (p = 0.055,
3 seeds, [../../results.md](../../results.md)) — a small average gain. Two tests ask where it
comes from: a synthetic one that rescales the heads, and one on the dataset's real spread
of head sizes. They agree on small heads and disagree on large ones, and the real-size test
is the one to trust where they differ.

## 1. Real head sizes (no resampling)

[size_by_volume.py](size_by_volume.py): the cross-validation results already on disk, per
case averaged over the 3 seeds (each case scored by the fold that held it out), grouped by
the label's head volume (35–291 mL, V0 = 130 mL). Paired Wilcoxon within each group.

    python size_normalization_approach/evaluation/size_by_volume.py              # terciles
    python size_normalization_approach/evaluation/size_by_volume.py --groups 5   # quintiles

**By head volume (terciles, 30 cases each)**

| Group | Volume | Plain x4 Dice | Normalised Dice | Difference | p | Normalised better | Plain HD95 | Normalised HD95 |
|---|---|---|---|---|---|---|---|---|
| small | 35–120 mL | 0.9417 | **0.9455** | **+0.0038** | 0.0024 | 21/30 | 3.18 | 2.93 |
| medium | 121–141 mL | 0.9480 | 0.9487 | +0.0006 | 0.61 | 16/30 | 3.04 | 2.94 |
| large | 141–291 mL | 0.9444 | 0.9443 | −0.0000 | 0.81 | 14/30 | 3.89 | 3.63 |

**By distance from V0, |log(V / V0)| (terciles)**

| Group | Volume | Plain x4 Dice | Normalised Dice | Difference | p | Normalised better | Plain HD95 | Normalised HD95 |
|---|---|---|---|---|---|---|---|---|
| near V0 | 121–141 mL | 0.9480 | 0.9487 | +0.0006 | 0.61 | 16/30 | 3.04 | 2.94 |
| middle | 107–160 mL | 0.9461 | 0.9458 | −0.0003 | 0.95 | 14/30 | 3.19 | 3.18 |
| far from V0 | 35–291 mL | 0.9400 | **0.9440** | **+0.0040** | 0.012 | 21/30 | 3.87 | 3.38 |

**Quintiles at the extremes (18 cases each):** smallest (35–108 mL) +0.0053 (p = 0.003,
14/18), HD95 3.34 → 3.02; largest (157–291 mL) +0.0003 (p = 0.97), HD95 4.45 → 3.97.

**Does Dice depend on size?** Spearman over the 90 cases:

| | with head volume | with distance from V0 |
|---|---|---|
| Plain Dice | +0.16 (p = 0.14) | −0.27 (p = 0.009) |
| Normalised Dice | +0.03 (p = 0.77) | −0.19 (p = 0.07) |
| Normalised − plain | −0.23 (p = 0.03) | +0.15 (p = 0.16) |

**Reading.**
- The gain is **on small heads**: +0.004 to +0.005 Dice and −0.3 mm HD95, significant even
  with 18–30 cases. It shrinks as the head grows (ρ = −0.23).
- **Large heads gain nothing in Dice**, though their HD95 drops by 0.3–0.5 mm (not tested
  for significance).
- Plain x4 gets worse the further a head is from the median size; the normalised method
  much less so. Normalisation removes most of the size dependence, not all of it.
- The +0.0015 average is small because most heads are near the median: the medium tercile
  spans only 121–141 mL.

## 2. Synthetic size shift

[size_shift_test.py](size_shift_test.py), full results in
[size_shift_results.md](size_shift_results.md): every validation volume resampled so its
head is 0.6× to 1.5× its linear size, 5 folds × 3 seeds. The normalised method stays at
~0.946 from 0.85× to 1.5×; plain x4 falls to 0.905 at 0.7× and 0.925 at 1.5×.

That test favours normalisation by construction. Rescaling also rescales the speckle, which
plain x4 has never seen; the normalised method resamples the head back to V0 and so undoes
the rescaling, texture included. A real large fetus has a large head with ordinary speckle,
and the normalised method then makes the speckle finer than in training. This is why the
synthetic test shows a gain for large heads (+0.008 at 1.3×, +0.020 at 1.5×) that the real
heads do not (−0.0000 for 141–291 mL). For small heads the two tests agree.

## Conclusion

Size normalisation makes the method **robust to small heads**, which plain x4 segments
worse (they span few voxels at 1.6 mm), and lowers HD95 on large ones. It does not improve
the Dice of large heads. Its average gain is small because the dataset is concentrated
around the median size; on data with more small heads (earlier gestational ages) it would
matter more.

## Caveats

- 30 (or 18) cases per group: the medium and large groups cannot show effects below ~0.002.
  Six groups are compared above; with a Bonferroni correction (p < 0.0083) the small-head
  tercile (p = 0.0024) and smallest quintile (p = 0.003) stay significant, the far-from-V0
  tercile (p = 0.012) does not.
- Head size is not the only thing that differs between groups (for instance how much of the
  head the scan sector cuts; 80 of the 90 heads are cut). That was not measured here.
- The normalised method uses pass 1's estimated volume. On the real heads its error is the
  same in every tercile (median 4.4%, 4.5%, 4.2% of the volume; 1.4% of the linear scale k),
  so it does not explain the difference between groups.
