# Segments the fetal head in a new 3D ultrasound volume: NIfTI in, NIfTI mask out.
#
#     python size_normalization/deploy/segment.py scan.nii.gz head_mask.nii.gz
#     python size_normalization/deploy/segment.py scan.nii.gz mask.nii.gz --tta --sdf sdf.nii.gz
#
# The size-normalised method (infer.py) with the trained networks: pass 1 (plain x4) for
# the head volume, then pass 2 on the grid that brings the head to the canonical size.
# Several --pass2 are averaged (the ensemble); --tta adds the 8 mirrorings.
#
# The volume is reoriented to RAS for the networks (as common/image_cache.py does for training) and
# the mask is written back in the input's own orientation, voxel for voxel on its grid,
# with its affine and header. Needs isotropic voxels (the dataset's are 0.4 mm); resample
# other scans first.
#
# The networks were trained on cross-validation folds, so each saw 72 of the 90 cases;
# the default is fold 0, seed 42.

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the repository root
from common.paths import NORM_CACHE, SIZE_NORM_RUNS  # noqa: E402

from size_normalization.infer import load_net, segment  # noqa: E402
from size_normalization.build_cache import reference_v0  # noqa: E402

RUNS = SIZE_NORM_RUNS


# A RAS-oriented array and affine -> the input's own orientation (flips and axis swaps only).
def to_input_orientation(arr: np.ndarray, input_affine: np.ndarray) -> np.ndarray:
    ras_to_input = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt("RAS"),
                                                   nib.orientations.io_orientation(input_affine))
    return nib.orientations.apply_orientation(arr, ras_to_input)


def main():
    p = argparse.ArgumentParser(description="Segment the fetal head in a NIfTI volume.")
    p.add_argument("image", help="input NIfTI (.nii or .nii.gz)")
    p.add_argument("mask", help="output mask NIfTI (1 inside the head, 0 outside)")
    p.add_argument("--sdf", default=None, help="also write the predicted SDF (mm / 10, negative inside)")
    p.add_argument("--pass1", default=os.path.join(RUNS, "nnaug_x4e500_s42", "fold0", "checkpoints", "phase1_last.pth"))
    p.add_argument("--pass2", nargs="+",
                   default=[os.path.join(RUNS, "nnaug_x4norm_s42", "fold0", "checkpoints", "phase1_last.pth")])
    p.add_argument("--tta", action="store_true", help="average the 8 mirrorings")
    p.add_argument("--v0-ml", type=float, default=None,
                   help="the canonical head volume pass 2 was trained at; default: from --reference")
    p.add_argument("--reference", default=NORM_CACHE,
                   help="pass 2's training cache, whose norm.json gives V0")
    a = p.parse_args()

    img = nib.load(a.image)
    ras = nib.as_closest_canonical(img)
    aff1 = np.asarray(ras.affine, dtype=np.float64)
    spacing = np.sqrt((aff1[:3, :3] ** 2).sum(0))
    if not np.allclose(aff1[:3, :3], np.diag(np.diag(aff1[:3, :3])), atol=1e-4 * spacing.max()):
        sys.exit(f"{a.image}: the voxel axes are oblique to the world axes; resample it first")
    if not np.allclose(spacing, spacing[0], rtol=1e-3):
        sys.exit(f"{a.image}: voxels of {spacing.round(3)} mm are not isotropic; resample it first")
    img1 = np.asarray(ras.get_fdata(dtype=np.float32))
    if img1.ndim == 4 and img1.shape[3] == 1:
        img1 = img1[..., 0]
    # Reorienting by flips leaves negative strides, which torch cannot take (and which make
    # every later pass slower, as in common/image_cache.py): a contiguous copy.
    img1 = np.ascontiguousarray(img1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass1, _ = load_net(a.pass1, dev)
    nets = [load_net(c, dev) for c in a.pass2]
    cfg = nets[0][1]
    v0_ml = a.v0_ml if a.v0_ml is not None else reference_v0(a.reference)

    field, info = segment(img1, aff1, pass1, [m for m, _ in nets], v0_ml, a.tta, dev,
                          factor2=cfg["factor"], cube2=cfg["cube"], mask2=cfg["loss"] == "mask")
    field = field.cpu().numpy().astype(np.float32)
    mask = (field < 0).astype(np.uint8)

    out = nib.Nifti1Image(to_input_orientation(mask, img.affine), img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, a.mask)
    if a.sdf:
        sdf = nib.Nifti1Image(to_input_orientation(field, img.affine), img.affine, img.header)
        sdf.set_data_dtype(np.float32)
        nib.save(sdf, a.sdf)

    head_ml = float(mask.sum()) * float(np.prod(spacing)) / 1000
    print(f"{a.image}: {img1.shape} at {spacing[0]:.2f} mm; pass 1 {info['volume_ml']:.1f} mL "
          f"(k = {info['k']:.3f}"
          + (", outside 0.5-2: pass 1's segmentation kept" if info["fallback"] else "")
          + f"); head {head_ml:.1f} mL -> {a.mask}")


if __name__ == "__main__":
    main()
