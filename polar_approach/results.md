# Polar head v2: results

The v2 polar branch (`polar_model.py`): from the head's centre, the radius in 64
directions, read by an MLP from four levels of the U-Net, trained with an |log r| loss
(weight 0.1, ramped over 50 epochs). Two arms, seed 42, 5 folds each, both compared with
the size-normalised network without the branch (`nnaug_x4norm_s42`), 90 cases, paired
Wilcoxon:

- `x4normpolar2`: the polar loss only.
- `x4normpolar2cons`: plus the SDF–polar consistency |SDF(c + r·d)| (weight 0.1 from
  epoch 150), whose gradient reaches the SDF only once the mean |log r| error is below 0.05.

v1, a per-voxel 1×1×1 convolution inside the U-Net, is in [v1/results.md](v1/results.md).

## 1. Segmentation (the SDF, the objective)

| | Dice | HD95 (mm) | ASD (mm) |
|---|---|---|---|
| without the branch | **0.9466** | **3.14** | **1.049** |
| v2, polar loss | 0.9455 (−0.0011, p = 0.19) | 3.22 (+0.09, p = 0.29) | 1.072 (+0.023, p = 0.18) |
| v2, + consistency | 0.9458 (−0.0009, p = 0.26) | 3.20 (+0.07, p = 0.41) | 1.069 (+0.020, p = 0.23) |

The branch is better on 43 and 41 of the 90 cases. Consistency against the polar loss
alone: Dice +0.0002 (p = 0.08). None of the differences is significant; all point the
same way as v1 (−0.0014) and the earlier auxiliary heads.

The consistency mostly did not reach the SDF. Its gate opened only on fold 4, at epoch
270. On folds 0–3 the training |log r| error ended at 0.057–0.064 (1.06–1.16 voxels),
just above the 0.05 threshold. So on four folds out of five the consistency term trained
only the polar branch, towards the SDF's surface.

## 2. Radii for biometry

[radii_test.py](radii_test.py), from pass 1's centre, against the label's rays from the
same centre:

| | polar branch | rays traced on the SDF | polar better on |
|---|---|---|---|
| v2, polar loss | 1.58 mm (median 1.53) | **1.12 mm** (median 1.01) | 3/90 |
| v2, + consistency | 1.58 mm (median 1.47) | **1.11 mm** (median 1.03) | 5/90 |
| v1 (for reference) | 2.58 mm | 1.12 mm | 1/90 |

Wilcoxon p = 3e-16 for both v2 arms. By fold, the polar branch is at 1.50–1.70 mm and the
traced SDF at 1.05–1.19 mm.

## Conclusion

v2 fixes v1's training problems: no fold stalled, and the radius error fell from 2.58 to
1.58 mm. It still does not help. The SDF is not better with the branch, and the branch's
radii are about 40% further off than rays traced on the SDF the network already outputs.
The consistency term could not be tested properly, because its gate stayed closed on four
folds. Lowering the gate would let a 1.6 mm-error branch pull a 1.1 mm-error surface. For
biometry, diameters and radii should come from the SDF.
