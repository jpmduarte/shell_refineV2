# Where pass 1's volume error comes from

Pass 1 (the plain x4 network) gives the head volume the size normalisation needs, and the
volume is what biometry will use later. Its error is 4.1% median, with almost no overall
bias (−0.1%). It is systematic per case: pass 2 errs the same way on the same heads
(Spearman 0.88). [pass1_error_analysis.py](pass1_error_analysis.py) takes every case's
error apart at native resolution against the label; [pass1_error_summary.py](pass1_error_summary.py)
makes the tables:

    python size_normalization/evaluation/pass1_error_analysis.py   # ~12 min, GPU
    python size_normalization/evaluation/pass1_error_summary.py

Each case uses the networks of the fold that held it out: all 3 seeds for the volumes,
seed 42 for the spatial breakdown.

| Volume | median abs error | signed mean | under by > 5% | over by > 5% |
|---|---|---|---|---|
| pass 1, x4 count (what k uses) | 4.1% | −0.1% | 20 | 17 |
| pass 1, native mask | 4.0% | −0.3% | 20 | 17 |
| pass 2, native mask | 4.5% | −0.8% | 21 | 14 |

## Mostly a thin offset of the contour

Per case, pass 1 misses 8.1 mL and adds 6.9 mL on average. 70% of the missed volume and
60% of the added volume lie within 1.2 mm (3 native voxels) of the true surface. This is
the contour drawn slightly inside or outside, not pieces of head missed. The uniform shift
of the predicted surface that would give the true volume is 0.31 mm median (90th
percentile 0.87 mm), less than one native voxel (0.4 mm) and a fifth of the x4 voxel
(1.6 mm). On a head of ~120 cm² of surface, 0.3 mm is ~3.6 mL, about 3%. That offset is most
of the error.

## The two groups of cases are offset in opposite directions

The cases fall into two groups by file name: 31 named by a DICOM UID
(`1.2.826.0.1.3680043...`) and 59 named `<patient>_I<series>`. Pass 1's error has
opposite signs in the two:

| Cases | n | pass-1 volume error, median | contour offset, median |
|---|---|---|---|
| UID names | 31 | **+4.3%** | −0.31 mm (prediction outside the label) |
| patient names | 59 | **−2.0%** | +0.17 mm (prediction inside the label) |

Mann-Whitney p = 6e-7. It is not the sector cut. Among the cut heads only, with the same
share of the surface at the sector's edge (18% and 20%), the gap stays (+4.1% vs −2.0%,
p = 3e-5). It also sits in the thin shell: the UID cases have twice the thin added volume
(6.1 vs 3.1 mL), and the patient-named cases twice the thin missed volume (6.9 vs 3.6 mL).

So the two groups' labels put the head's boundary about 0.5 mm apart relative to the image,
and one network trained on both learns the middle. Likely causes, which only the data's
provenance can settle:
- two annotators, or two annotation sessions with a different habit (the outer or the inner
  edge of the bright skull echo; the MITK interpolation settings);
- two acquisition or export batches, with different gain, depth or reconstruction, so that
  the same boundary looks different.

Removing each group's median offset would bring the median error from 4.1% to 3.2%, and
the cases off by more than 5% from 37 to 27.

## The rest: pieces missed where the head leaves the sector

The cases that are underestimated most are cut heads that miss thick pieces at the
sector's edge: `1_I0000029` and `0.1_I0000010` (290 mL, a third of the surface at the
edge) each miss 22 mL more than 1.2 mm from the label's surface. Across the 90 cases, the
more of the surface at the edge, the more negative the error (Spearman −0.33, p = 0.002).
Whether the missed boundary is darker than the found one, as with shadowing or dropout,
is not significant (−0.17, p = 0.12).

For a cut head, the "true volume" is the volume of the part inside the sector. Biometry
needs the whole head, which is a shape-completion problem (fit a model of the head to the
visible part) rather than a segmentation one.

## What would reduce the error

1. **Find out why the two groups differ** (who annotated which, which edge, which scanner
   or export), then annotate both the same way, or re-annotate one group. This is the
   largest single part and also limits the Dice.
2. **Measure the annotation's own repeatability** (re-annotate ~10 cases blind): a
   ±0.3 mm offset may already be at that limit.
3. **For biometry, complete the cut heads** from the visible part instead of measuring
   their segmented volume.
