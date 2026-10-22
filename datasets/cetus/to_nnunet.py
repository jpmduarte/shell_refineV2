"""Lay out the converted CETUS NIfTI as an nnU-Net raw dataset, with our folds as its splits_final.json.

The cross-validation volumes go to imagesTr/labelsTr, the test volumes to imagesTs, with their masks in
labelsTs_closed (not read by nnU-Net) for the final comparison. Case names are kept (patient07_ED), so
splits_final.json is datasets/cetus/splits.json case for case.
"""
import argparse
import json
import os
import shutil

from common.folds import fold_split, get_folds
from datasets.cetus import DATASET

DATASET_NAME = "Dataset502_CETUS"
CHANNEL = "_0000"


def copy_split(nifti_dir, out, images_name, labels_name):
    images, labels = os.path.join(out, images_name), os.path.join(out, labels_name)
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    cases = sorted(f[:-len(".nii.gz")] for f in os.listdir(os.path.join(nifti_dir, "images")))
    for case in cases:
        shutil.copyfile(os.path.join(nifti_dir, "images", f"{case}.nii.gz"), os.path.join(images, f"{case}{CHANNEL}.nii.gz"))
        shutil.copyfile(os.path.join(nifti_dir, "labels", f"{case}.nii.gz"), os.path.join(labels, f"{case}.nii.gz"))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nifti-dir", default=os.path.join(os.path.dirname(DATASET.cache), "nifti"))
    parser.add_argument("--out-dir", required=True, help="where to write nnUNet_raw/" + DATASET_NAME + " and splits_final.json")
    args = parser.parse_args()

    raw = os.path.join(args.out_dir, "nnUNet_raw", DATASET_NAME)
    training = copy_split(os.path.join(args.nifti_dir, "cv"), raw, "imagesTr", "labelsTr")
    copy_split(os.path.join(args.nifti_dir, "test"), raw, "imagesTs", "labelsTs_closed")
    with open(os.path.join(raw, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "ultrasound"}, "labels": {"background": 0, "left_ventricle": 1},
                   "numTraining": len(training), "file_ending": ".nii.gz"}, f, indent=1)

    folds = get_folds(DATASET)
    assert sorted(c for fold in folds for c in fold) == training
    splits = [dict(zip(("train", "val"), fold_split(folds, k))) for k in range(len(folds))]
    with open(os.path.join(args.out_dir, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=1)
    print(f"{len(training)} training cases, {len(splits)} folds -> {args.out_dir}")


if __name__ == "__main__":
    main()
