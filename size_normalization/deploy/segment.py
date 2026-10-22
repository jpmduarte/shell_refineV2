"""Segment the fetal head in a new 3D ultrasound volume with the size-normalised method: NIfTI in, NIfTI mask out.

The mask is written in the input's own orientation, voxel for voxel, with its affine and header.
Needs isotropic voxels (the dataset's are 0.4 mm). The default networks are fold 0, seed 42, which saw 72 of the 90 cases.
"""
import argparse
import sys

import nibabel as nib
import numpy as np
import torch

from common.geometry import MM3_PER_ML, SPATIAL
from datasets import add_dataset_argument, get as get_dataset
from size_normalization.build_cache import reference_v0
from size_normalization.infer import load_net, segment

DEFAULT_SEED, DEFAULT_FOLD = 42, 0
OBLIQUE_TOLERANCE = 1e-4
ISOTROPY_TOLERANCE = 1e-3

USAGE = """\
examples:
  python size_normalization/deploy/segment.py scan.nii.gz head_mask.nii.gz
  python size_normalization/deploy/segment.py scan.nii.gz mask.nii.gz --tta --sdf sdf.nii.gz"""


def to_input_orientation(arr: np.ndarray, input_affine: np.ndarray) -> np.ndarray:
    """A RAS-oriented array back to the input's orientation (flips and axis swaps only)."""
    ras_to_input = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt("RAS"),
                                                   nib.orientations.io_orientation(input_affine))
    return nib.orientations.apply_orientation(arr, ras_to_input)


def load_ras_volume(path):
    """(the NIfTI image, its volume in RAS as a contiguous float32 array, the RAS affine, the spacing per axis)."""
    img = nib.load(path)
    ras = nib.as_closest_canonical(img)
    aff1 = np.asarray(ras.affine, dtype=np.float64)
    voxel_axes = aff1[SPATIAL, SPATIAL]
    spacing = np.sqrt((voxel_axes ** 2).sum(0))
    if not np.allclose(voxel_axes, np.diag(np.diag(voxel_axes)), atol=OBLIQUE_TOLERANCE * spacing.max()):
        sys.exit(f"{path}: the voxel axes are oblique to the world axes; resample it first")
    if not np.allclose(spacing, spacing[0], rtol=ISOTROPY_TOLERANCE):
        sys.exit(f"{path}: voxels of {spacing.round(3)} mm are not isotropic; resample it first")
    img1 = np.asarray(ras.get_fdata(dtype=np.float32))
    if img1.ndim == 4 and img1.shape[3] == 1:
        img1 = img1[..., 0]
    # The flips leave negative strides, which torch cannot take.
    return img, np.ascontiguousarray(img1), aff1, spacing


def save_like(arr, dtype, reference, path):
    out = nib.Nifti1Image(to_input_orientation(arr, reference.affine), reference.affine, reference.header)
    out.set_data_dtype(dtype)
    nib.save(out, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="input NIfTI (.nii or .nii.gz)")
    parser.add_argument("mask", help="output mask NIfTI (1 inside the head, 0 outside)")
    parser.add_argument("--sdf", default=None, help="also write the predicted SDF (mm / 10, negative inside)")
    parser.add_argument("--pass1", default=None, help="default: the dataset's pass-1 run, seed 42, fold 0")
    parser.add_argument("--pass2", nargs="+", default=None,
                        help="several are averaged (the ensemble); default: the dataset's pass-2 run, seed 42, fold 0")
    parser.add_argument("--tta", action="store_true", help="average the 8 mirrorings")
    parser.add_argument("--v0-ml", type=float, default=None,
                        help="the canonical head volume pass 2 was trained at; default: from --reference")
    parser.add_argument("--reference", default=None,
                        help="pass 2's training cache, whose norm.json gives V0; default: the dataset's")
    add_dataset_argument(parser)
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    args.pass1 = args.pass1 or dataset.checkpoint(dataset.pass1_run, DEFAULT_SEED, DEFAULT_FOLD)
    args.pass2 = args.pass2 or [dataset.checkpoint(dataset.pass2_run, DEFAULT_SEED, DEFAULT_FOLD)]
    args.reference = args.reference or dataset.norm_cache

    img, img1, aff1, spacing = load_ras_volume(args.image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass1 = load_net(args.pass1, device)
    pass2 = [load_net(c, device) for c in args.pass2]
    v0_ml = args.v0_ml if args.v0_ml is not None else reference_v0(args.reference)

    field, estimate = segment(img1, aff1, pass1, pass2, v0_ml, args.tta, device, k_range=dataset.k_range)
    field = field.cpu().numpy().astype(np.float32)
    mask = (field < 0).astype(np.uint8)

    save_like(mask, np.uint8, img, args.mask)
    if args.sdf:
        save_like(field, np.float32, img, args.sdf)

    head_ml = float(mask.sum()) * float(np.prod(spacing)) / MM3_PER_ML
    low, high = dataset.k_range
    kept = f", outside {low}-{high:g}: pass 1's segmentation kept" if estimate.fallback else ""
    print(f"{args.image}: {img1.shape} at {spacing[0]:.2f} mm; pass 1 {estimate.volume_ml:.1f} mL "
          f"(k = {estimate.k:.3f}{kept}); head {head_ml:.1f} mL -> {args.mask}")


if __name__ == "__main__":
    main()
