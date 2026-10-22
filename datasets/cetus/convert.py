"""Convert CETUS from the zea HDF5 release (huggingface.co/datasets/zeahub/cetus-miccai-2014) to NIfTI pairs.

Writes <out>/<cv|test>/{images,labels}/<patientNN_ED|ES>.nii.gz: the B-mode volume in dB and the binary
left-ventricle mask. The official train and val patients (1-38) go to cv/, for cross-validation; the
official test patients (39-45) to test/, kept apart until the end. A volume too large for the 96^3
training cube at the x4 grid is resampled to LARGE_SPACING_MM, its mask through its signed distance.
"""
import argparse
import glob
import os

import h5py
import hdf5plugin  # noqa: F401  (registers the HDF5 compression filter the release uses)
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from common.geometry import CUBE, FACTOR
from common.sdf_targets import signed_distance_mm

TRACK = "tracks/track_0/data/"
M_TO_MM = 1000
LARGE_SPACING_MM = 0.64
SPLIT_OF = {"train": "cv", "val": "cv", "test": "test"}


def read_case(path):
    """(image in dB, boolean mask, isotropic spacing in mm, origin in mm)."""
    with h5py.File(path, "r") as h5:
        image = h5[TRACK + "image/values"][0].astype(np.float32)
        mask = h5[TRACK + "segmentation/values"][0, ..., 0].astype(bool)
        coords = h5[TRACK + "image/coordinates"]
        origin = coords[0, 0, 0].astype(np.float64) * M_TO_MM
        steps = np.array([coords[1, 0, 0], coords[0, 1, 0], coords[0, 0, 1]], dtype=np.float64) * M_TO_MM - origin
    spacing = np.linalg.norm(steps, axis=1)
    assert np.allclose(steps, np.diag(spacing), atol=1e-4) and np.ptp(spacing) < 1e-4, path   # isotropic, axis-aligned
    return image, mask, float(spacing[0]), origin


def fits_cube(shape):
    """Whether the volume fits the training cube once resampled to FACTOR x its spacing (as Spacingd sizes it)."""
    return max(round((n - 1) / FACTOR + 1) for n in shape) <= CUBE


def resample(image, mask, spacing_mm, new_spacing_mm):
    """The image trilinearly, the mask through its signed distance, so its surface stays smooth."""
    scale = spacing_mm / new_spacing_mm
    image = zoom(image, scale, order=1)
    sdf = zoom(signed_distance_mm(mask, (spacing_mm,) * 3), scale, order=1)
    assert sdf.shape == image.shape
    return image, sdf < 0


def save_nifti(array, spacing_mm, origin_mm, path):
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = origin_mm
    nib.save(nib.Nifti1Image(array, affine), path)


def convert(raw_dir, out_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*", "patient*", "*.hdf5")))
    print(f"{len(files)} volumes in {raw_dir} -> {out_dir}", flush=True)
    for path in files:
        split = SPLIT_OF[os.path.basename(os.path.dirname(os.path.dirname(path)))]
        case = os.path.splitext(os.path.basename(path))[0]
        image, mask, spacing, origin = read_case(path)
        note = ""
        if not fits_cube(image.shape):
            image, mask = resample(image, mask, spacing, LARGE_SPACING_MM)
            note = f"  resampled {spacing:.3f} -> {LARGE_SPACING_MM} mm"
            spacing = LARGE_SPACING_MM
        for kind, array in (("images", image), ("labels", mask.astype(np.uint8))):
            folder = os.path.join(out_dir, split, kind)
            os.makedirs(folder, exist_ok=True)
            save_nifti(array, spacing, origin, os.path.join(folder, f"{case}.nii.gz"))
        print(f"  {split:<4} {case:<14} {'x'.join(map(str, image.shape))} at {spacing:.3f} mm{note}", flush=True)


if __name__ == "__main__":
    from datasets.cetus import DATASET

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="the downloaded release: train/, val/, test/")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(DATASET.cache), "nifti"))
    args = parser.parse_args()
    convert(args.raw_dir, args.out_dir)
