# Can a better pass-1 volume improve the size normalisation?

The normalised network sees each head at the reference size V0, scaled with the head
volume that pass 1 (the plain x4 network) estimates. That estimate is off by 4.3% (median),
and by 8–13% (underestimates) on three of the ten worst cases, where the normalised method
did worse than plain x4. Two cheap fixes, both inference only, were tested with
[pass1_volume_test.py](pass1_volume_test.py):

    python size_normalization/evaluation/pass1_volume_test.py

1. **A better pass 1:** its 8 mirrorings averaged (TTA), or the 3 seeds' pass 1 with TTA
   averaged (an ensemble; all networks of the same fold, so no leak).
2. **Iterate:** run pass 2, measure the head volume on its own output, renormalise with
   that volume and run pass 2 again (once or twice).

Everything else fixed: pass 2 is one model without TTA, each seed on its own fold's
networks, all 90 cases × 3 seeds; per case the mean over the seeds, paired Wilcoxon against
the deployed estimate. The oracle feeds the label's volume: the best any estimate can do.

| Volume fed to pass 2 | Dice | vs deployed | p | HD95 | Volume error, median abs | signed mean | max abs |
|---|---|---|---|---|---|---|---|
| **pass 1 once (deployed)** | 0.9462 | — | — | 3.17 | 4.3% | −0.1% | 20.2% |
| pass 1 + TTA | 0.9463 | +0.0001 | 0.26 | 3.15 | 4.0% | −0.2% | 20.2% |
| pass 1, ensemble of 3 + TTA | 0.9463 | +0.0000 | 0.60 | 3.15 | 4.0% | −0.2% | 18.8% |
| iterate once | 0.9459 | −0.0003 | 0.29 | 3.18 | 4.3% | −0.7% | 20.3% |
| iterate twice | 0.9456 | −0.0006 | 0.10 | 3.20 | 4.3% | −0.7% | 21.3% |
| ensemble + TTA, then iterate once | 0.9458 | −0.0004 | 0.32 | 3.20 | 4.3% | −0.7% | 19.5% |
| oracle (true volume) | 0.9468 | +0.0006 | 0.08 | 3.17 | 0 | 0 | 0 |

The deployed row reproduces the cross-validation result (0.9462).

**The three underestimated cases** (Dice; mean volume error over the seeds):

| Case | Deployed | Ensemble + TTA | Iterate once | Oracle |
|---|---|---|---|---|
| `41_I0000005` | 0.8876 (−13.2%) | 0.8885 (−12.7%) | 0.8791 (−18.5%) | 0.8995 |
| `21_I0000008` | 0.9304 (−13.2%) | 0.9339 (−11.7%) | 0.9358 (−12.6%) | 0.9362 |
| `3_I0000006` | 0.9211 (−7.9%) | 0.9201 (−8.6%) | 0.9199 (−11.4%) | 0.9252 |

## What it shows

- **Neither fix helps.** TTA and the ensemble move the median volume error from 4.3% to 4.0%
  and the Dice not at all; iterating makes it slightly worse.
- **The volume error is systematic, not noise.** The volume pass 1 estimates and the one
  measured on pass 2's output err together (correlation 0.81 over the 270 case-seeds): both
  networks underestimate the same heads, presumably where part of the boundary is faint or
  ambiguous. Averaging removes only the random part; iterating feeds pass 2's own, larger
  underestimate back in (−13% → −18.5% on `41_I0000005`).
- **Little to gain anyway.** Even the true volume adds only +0.0006 (p = 0.08) on average.
  The cost of the estimate is concentrated in a few hard cases, and fixing it there needs a
  better segmentation of those cases, not a better volume trick.

## What changed in the pipeline

None of the variants is adopted. What the size-shift test showed is needed is a guard:
[infer.py](../infer.py) `segment` (the pipeline every script runs) uses pass 1's scale only
within `k_range` (0.5–2, a head of about 16–1040 mL); outside it, including a pass 1 that
finds no head, it returns pass 1's own segmentation instead of failing. It changes nothing
on the 90 cases (every k lies in 0.65–1.27). The three checks (a normal case unchanged, a
forced out-of-range case exactly pass 1's result, an empty volume without a crash) are
`tests/test_method.py` `Guard`.
