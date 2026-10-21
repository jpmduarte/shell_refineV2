# Phase 2: trains the boundary refiner on one fold, on 64^3 cubes from the band crops.
#
#     python boundary_refiner/train_cube.py --fold 0 --seed 42 \
#         --crops-dir boundary_refiner/crops/x4_s42 \
#         --run-dir boundary_refiner/runs/x4_s42_cube/fold0
#
# The refiner takes the image and phase 1's prior and outputs, per voxel, the probability
# of being inside the head. It is scored with Dice + BCE inside the band only (train.py);
# the BCE is weighted towards the prior's surface (--sw-sigma). 100 epochs, Adam 3e-4
# with cosine decay, one case (12 cubes) per step. No weight decay or dropout: on the
# earlier refiner they reduced overfitting but gained no Dice.
#
# Network size (refiner_model.py): --depth (pooling levels), --kernel (3 or 5),
# --base-ch (channels at the top level). They are saved in <run-dir>/config.json, which
# predict.py reads to rebuild the same network.
#
# Resuming: every --resume-every epochs the full training state is saved to
# <run-dir>/checkpoints/resume.pth (network, optimiser, learning-rate schedule, epoch, and
# the random generators). Run the same command again after an interruption (a power cut)
# and training continues from there. The data loader's workers are re-seeded every epoch
# from the loader's generator, which is part of the saved state, so a resumed run follows
# exactly the same path as an uninterrupted one. resume.pth is deleted once training ends.
#
# Reported checkpoint: phase2_last.pth, the last epoch (nothing is chosen on the
# validation fold).

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import SPLITS_JSON, REFINER_DIR  # noqa: E402

from boundary_refiner.dataset import CUBE, PATCHES_PER_CASE, CubeDataset, collate_cases  # noqa: E402
from common.profiler import Profiler, model_summary  # noqa: E402
from boundary_refiner.refiner_model import RefinerUNet  # noqa: E402
from common.folds import fold_split, get_folds  # noqa: E402
from boundary_refiner.train import train_epoch, val_epoch  # noqa: E402


# Each data-loader worker gets its own random generator, seeded from the worker's seed.
def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed % 2**32)


# Everything needed to continue training exactly where it stopped. Written to a temporary
# file and renamed, so an interruption during the save cannot leave a broken checkpoint.
def save_resume(path, epoch, model, optimizer, scheduler, loader_gen, train_set):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_gen": loader_gen.get_state() if loader_gen is not None else None,
        "dataset_rng": train_set._rng.bit_generator.state,   # used when --workers 0
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    p = argparse.ArgumentParser(description="Train the boundary refiner on cubes, one fold.")
    p.add_argument("--fold",             type=int,   required=True)
    p.add_argument("--n-folds",          type=int,   default=5)
    p.add_argument("--crops-dir",        type=str,   required=True)
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--cube",             type=int,   default=CUBE, help="cube side, a multiple of 8")
    p.add_argument("--patches-per-case", type=int,   default=PATCHES_PER_CASE)
    p.add_argument("--batch-size",       type=int,   default=1, help="cases per step")
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--depth",            type=int,   default=3, help="pooling levels")
    p.add_argument("--kernel",           type=int,   default=3, help="convolution kernel size")
    p.add_argument("--base-ch",          type=int,   default=16, help="channels at the top level")
    p.add_argument("--sw-sigma",         type=float, default=2.0,
                   help="mm; BCE weighted by exp(-|prior|/sigma) inside the band; 0 = binary band")
    p.add_argument("--workers",          type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--resume-every",     type=int,   default=10,
                   help="save a resume checkpoint every this many epochs; 0 = never")
    p.add_argument("--splits-json",      type=str,   default=SPLITS_JSON)
    p.add_argument("--run-dir",          type=str,   default=None)
    a = p.parse_args()

    if a.cube % 2 ** a.depth:
        p.error(f"--cube must be divisible by {2 ** a.depth} ({a.depth} pooling levels)")
    if a.kernel % 2 == 0:
        p.error("--kernel must be odd")
    run_dir = a.run_dir or os.path.join(REFINER_DIR, "runs", f"cube_s{a.seed}", f"fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)

    # What predict.py needs to rebuild this network and cut the same cubes.
    config = {"depth": a.depth, "kernel": a.kernel, "base_ch": a.base_ch, "cube": a.cube}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # A seed makes the run reproducible: weights, cube draws, augmentation, kernels.
    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {a.fold}  Run: {run_dir}")
    print(f"Cubes: {a.cube}^3 x {a.patches_per_case} per case  "
          f"({a.patches_per_case * a.cube ** 3:,} voxels per case)  crops: {a.crops_dir}")

    # Only cases that have a crop; the folds are the shared patient-grouped ones.
    available = sorted(f[:-4] for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    folds = get_folds(available, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    train_ids = [c for c in train_ids if c in set(available)]
    val_ids   = [c for c in val_ids if c in set(available)]
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    loader_gen  = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    worker_init = seed_worker if a.seed is not None else None
    common = dict(cube=a.cube, patches_per_case=a.patches_per_case, seed=a.seed)
    train_set = CubeDataset(a.crops_dir, train_ids, augment=True, **common)
    train_loader = DataLoader(
        train_set, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True,
        collate_fn=collate_cases, generator=loader_gen, worker_init_fn=worker_init)
    val_loader = DataLoader(
        CubeDataset(a.crops_dir, val_ids, augment=False, **common),
        batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True,
        collate_fn=collate_cases, generator=loader_gen, worker_init_fn=worker_init)

    model = RefinerUNet(base_channels=a.base_ch, depth=a.depth, kernel=a.kernel).to(device)
    print(f"Network: depth {a.depth}, kernel {a.kernel}, base channels {a.base_ch}")
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    print(f"Model: {model_summary(model)}")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(checkpoint_dir, "resume.pth")
    log_path = os.path.join(run_dir, "train_log.csv")

    # Continue an interrupted run if it left a resume checkpoint; otherwise start fresh.
    start_epoch = 1
    if os.path.exists(resume_path):
        # Loaded on the CPU: generator states must be CPU byte tensors (load_state_dict
        # moves the network's and optimiser's tensors to their device by itself). The
        # states are cloned because the loaded tensors share storage, which the
        # generators refuse.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"].clone())
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.clone() for s in state["cuda_rng"]])
        if loader_gen is not None and state["loader_gen"] is not None:
            loader_gen.set_state(state["loader_gen"].clone())
        train_set._rng.bit_generator.state = state["dataset_rng"]
        start_epoch = state["epoch"] + 1
        # Keep the log's rows up to the checkpoint; later ones are about to be redone.
        with open(log_path) as f:
            rows = f.readlines()
        with open(log_path, "w") as f:
            f.writelines(rows[:1] + [r for r in rows[1:] if int(r.split(",")[0]) <= state["epoch"]])
        print(f"Resuming after epoch {state['epoch']}")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_band_dice\n")

    prof = Profiler("refiner_train")
    with prof:
        for epoch in range(start_epoch, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, optimizer, device,
                                         band_masked=True, sw_sigma=a.sw_sigma)
                s["items"] = len(train_ids) * a.patches_per_case

            # Logged for the curves only; nothing is chosen from it.
            with prof.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, device, band_masked=True)

            scheduler.step()

            print(f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_band_dice={val_dice:.4f}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

            if a.resume_every and epoch % a.resume_every == 0 and epoch < a.epochs:
                save_resume(resume_path, epoch, model, optimizer, scheduler, loader_gen, train_set)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase2_last.pth"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"\nCheckpoint : {checkpoint_dir}")


if __name__ == "__main__":
    main()
