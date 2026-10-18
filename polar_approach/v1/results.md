# The polar head: does it help?

These runs used the first version of the head, a 1x1x1 convolution inside the U-Net
(`model.py` `polar_dirs`, `train.py --polar-coef`), since moved out of the baseline. The
code is at commit `f2bc53c`; `radii_test.py` here needs it (`forward_with_polar`).

An auxiliary output trained with the size-normalised network (`polar.py`,
`train.py --polar-coef 0.5`): per voxel, the head's radius in 64 fixed directions,
supervised at 128 interior points per sample. The aim was global shape context for the SDF
and radii usable for biometry. Seed 42, 5 folds (`runs/nnaug_x4normpolar_s42`), compared
with the same recipe without the head (`runs/nnaug_x4norm_s42`), 90 cases, paired Wilcoxon.

## 1. Segmentation (mask from the SDF only)

| | with the polar head | without | difference | p | better with it |
|---|---|---|---|---|---|
| Dice | 0.9452 | 0.9466 | **−0.0014** | 0.04 | 40/90 |
| HD95 (mm) | 3.26 | 3.14 | +0.13 | 0.12 | 37/90 |
| ASD (mm) | 1.075 | 1.049 | +0.027 | 0.05 | 37/90 |

By fold: +0.0040 on fold 3, −0.0015 to −0.0044 on the other four.

The training log (`train_polar_vox`) shows how the head learns. Its radius error sits at
~8 voxels for the first 50–100 epochs, then falls to ~1.4 voxels by epoch 500 (folds 0–3).
On fold 4 it never left that plateau (6.8 voxels at epoch 500, close to predicting a mean
radius), and its weighted loss dominated the sum: that fold's SDF L1 ended at twice the
others' (0.0168 against 0.0087). Without fold 4 the change is still −0.0009.

## 2. Radii for biometry

[radii_test.py](radii_test.py): from pass 1's centre, the head's radius in the
64 directions read three ways, against the label's rays from the same centre:

| | mean radius error | median |
|---|---|---|
| the polar head | 2.58 mm | 1.76 mm |
| rays traced on pass 2's own SDF | **1.12 mm** | **1.04 mm** |

The traced SDF is better on 89 of 90 cases (p = 2e-16), and on every fold (polar
1.5–1.8 mm on folds 0–3, 5.9 mm on fold 4; SDF 1.0–1.2 mm). Pass 1's centre is 1.26 mm
(median) from the label's centroid, which moves the label's own radii by 0.60 mm.

## Conclusion

The head does not help, in either role. The SDF gets slightly worse, as with the earlier
auxiliary heads (contour head −0.0002, joint 1D point head +0.0003 n.s.). Its radii are
twice as far off as rays traced on the SDF that the network already outputs. Training
with it is also fragile: one fold in five never learned the radii. The fusion and
completion inference (SDF uncertain → polar) is not worth testing, since the polar
surface is the less accurate of the two. For biometry, rays and diameters can be read from
the SDF directly.
