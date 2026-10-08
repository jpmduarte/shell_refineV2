# Builds a 20-case nnU-Net dataset to profile nnU-Net's training locally, using the plans
# of the run that was actually trained (results.md, Efficiency).
#
#     conda activate nnunet-bench
#     python scripts/nnunet_profile_dataset.py --out <dir>
#     set nnUNet_raw=<dir>/raw, nnUNet_preprocessed=<dir>/preprocessed, nnUNet_results=<dir>/results
#     nnUNetv2_preprocess -d 501 -c 3d_fullres -np 8
#     python scripts/profile_run.py --out prof.json -- nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_5epochs
#
# The cached 0.4 mm volumes are written as NIfTI in their stored (x, y, z) order; nnU-Net's
# reader (SimpleITK) returns them as (z, y, x), the order it was trained in (checked at
# inference: Dice 0.948 on fold 0). Copying the plans keeps patch 112x160x128 and batch 2.
# An nnU-Net epoch is a fixed 250 iterations, so 20 cases cost the same per epoch as 90.

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED = os.path.join(ROOT, "nnunet/runs/grouped_250ep/nnUNet_results/Dataset501_FetalHead/"
                             "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres")

p = argparse.ArgumentParser(description="Build a small nnU-Net dataset for profiling.")
p.add_argument("--out", required=True)
out = p.parse_args().out

raw = os.path.join(out, "raw", "Dataset501_FetalHead")
preprocessed = os.path.join(out, "preprocessed", "Dataset501_FetalHead")
for d in ("imagesTr", "labelsTr"):
    os.makedirs(os.path.join(raw, d), exist_ok=True)
os.makedirs(preprocessed, exist_ok=True)

# The first 20 cases, as NIfTI with a plain 0.4 mm affine.
ids = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "caches", "cache", "x1")))[:20]
affine = np.diag([0.4, 0.4, 0.4, 1.0])
for i, cid in enumerate(ids):
    img = np.load(os.path.join(ROOT, "caches", "cache", "x1", f"{cid}.npz"))["img"].astype(np.float32).squeeze()
    lab = np.load(os.path.join(ROOT, "caches", "cache", "label", f"{cid}.npz"))["label"].squeeze().astype(np.uint8)
    nib.save(nib.Nifti1Image(img, affine), os.path.join(raw, "imagesTr", f"case{i:03d}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(lab, affine), os.path.join(raw, "labelsTr", f"case{i:03d}.nii.gz"))
    print(i, cid[:30], img.shape, flush=True)

# The trained run's dataset description (with the new case count), plans and fingerprint.
dataset = json.load(open(os.path.join(TRAINED, "dataset.json")))
dataset["numTraining"] = len(ids)
for d in (raw, preprocessed):
    json.dump(dataset, open(os.path.join(d, "dataset.json"), "w"), indent=1)
shutil.copy(os.path.join(TRAINED, "plans.json"), os.path.join(preprocessed, "nnUNetPlans.json"))
shutil.copy(os.path.join(TRAINED, "dataset_fingerprint.json"),
            os.path.join(preprocessed, "dataset_fingerprint.json"))
print("->", out)
