# Trains the network on one fold.
#
#     python size_normalization_approach/train.py --fold 0 --seed 42 --run-dir size_normalization_approach/runs/x4norm_s42/fold0
#         the size-normalised network (cache_norm4, the default)
#     python size_normalization_approach/train.py --fold 0 --seed 42 --cache-dir caches/cache --run-dir size_normalization_approach/runs/x4_s42/fold0
#         the plain x4 network: the first pass, which estimates the head volume
#
# The reported checkpoint is phase1_last.pth, the last epoch: nothing is chosen on the
# validation fold, since that is the fold that gets reported.
#
# Writes <run-dir>/train_log.csv (one row per epoch) and <run-dir>/checkpoints/phase1_last.pth.

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import case_ids  # noqa: E402
from dataset import CUBE, FACTOR, SDFDataset  # noqa: E402
from loss import surface_weighted_l1  # noqa: E402
from metrics import boundary_dice_from_sdf, dice_from_sdf, mae  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds, make_folds  # noqa: E402


# Each data-loader worker is a copy of the dataset, including its random generator. Without
# this, every worker would draw the same "random" augmentations; here each gets its own
# generator, seeded from the worker's seed (which PyTorch derives from the run's seed).
def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed % 2**32)


# One pass over the training cases. Returns the mean loss per case.
def train_epoch(model, loader, optimizer, device, sw_coef: float, sw_sigma: float) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # L1 everywhere, plus the surface-weighted L1 (loss.py).
        loss = F.l1_loss(pred, y)
        if sw_coef > 0:
            loss = loss + sw_coef * surface_weighted_l1(pred, y, sw_sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# Scores on the validation cases, for the training curves only:
# (L1 loss, MAE, Dice, boundary Dice), each averaged over cases.
@torch.no_grad()
def val_epoch(model, loader, device, band_mm: float) -> tuple[float, float, float, float]:
    model.eval()
    total_loss   = 0.0
    mae_scores   = []
    dice_scores  = []
    bdice_scores = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += F.l1_loss(pred, y).item() * x.size(0)

        # Per case within the batch. Dice is None when undefined (both masks empty).
        for p, t in zip(pred, y):
            mae_scores.append(mae(p, t))
            score = dice_from_sdf(p, t)
            if score is not None:
                dice_scores.append(score)
            bscore = boundary_dice_from_sdf(p, t, band_mm=band_mm)
            if bscore is not None:
                bdice_scores.append(bscore)

    mean_mae   = sum(mae_scores)   / len(mae_scores)   if mae_scores   else 0.0
    mean_dice  = sum(dice_scores)  / len(dice_scores)  if dice_scores  else 0.0
    mean_bdice = sum(bdice_scores) / len(bdice_scores) if bdice_scores else 0.0
    return total_loss / len(loader.dataset), mean_mae, mean_dice, mean_bdice


def main():
    p = argparse.ArgumentParser(description="Train the network on one fold.")
    p.add_argument("--fold",        type=int,   required=True, help="0-4, which fold is validation")
    p.add_argument("--n-folds",     type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--workers",     type=int,   default=2,
                   help="data-loader processes; 6 measured slower on this machine")
    p.add_argument("--sw-coef",     type=float, default=0.5,
                   help="weight of the surface-weighted L1 added to the plain L1; 0 = off")
    p.add_argument("--sw-sigma",    type=float, default=1.5,
                   help="mm over which that weight decays away from the surface")
    p.add_argument("--factor",      type=int,   default=FACTOR,
                   help="the grid, as a multiple of the native spacing; needs <cache>/x<factor>")
    p.add_argument("--cube",        type=int,   default=None,
                   help="training cube per axis; default 96 x 4 / factor")
    p.add_argument("--base-ch",     type=int,   default=16,
                   help="channels at the top level, doubling per level (32: 32-64-128, ~4x parameters)")
    p.add_argument("--band-mm",     type=float, default=5.0,
                   help="half-width in mm of the band for the logged boundary Dice")
    p.add_argument("--inner-val",   type=int,   default=0,
                   help="N > 0: hold out 1/N of the training cases (patient-grouped, the same for "
                        "every seed) as an inner validation set, validate on it instead of the "
                        "fold, and also save the epoch with its best Dice as phase1_best.pth")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--cache-dir",   type=str,   default=os.path.join(PARENT, "caches", "cache_norm4"))
    p.add_argument("--splits-json", type=str,   default=os.path.join(PARENT, "splits.json"))
    p.add_argument("--run-dir",     type=str,   default=None,
                   help="default: size_normalization_approach/runs/fold<k>")
    a = p.parse_args()

    run_dir = a.run_dir or os.path.join(HERE, "runs", f"fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)
    cube = a.cube or CUBE * FACTOR // a.factor

    # A seed makes the run reproducible: same initial weights, same batches, same
    # augmentation, and deterministic GPU kernels.
    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {a.fold}  Run: {run_dir}")
    print(f"Loss: L1 + {a.sw_coef} x surface-weighted L1 @ {a.sw_sigma}mm  "
          f"cache: {a.cache_dir}  x{a.factor} padded to {cube}^3")

    # Which cases train and which validate.
    ids   = case_ids(a.cache_dir)
    folds = get_folds(ids, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    config = {"factor": a.factor, "cube": cube, "loss": "sdf", "base_ch": a.base_ch}
    if a.inner_val:
        # The inner validation set comes out of the training cases, by patient, with a fixed
        # seed: every seed of a fold holds out the same cases, and the fold itself stays
        # unseen, so the best epoch is chosen without looking at what gets reported.
        inner = make_folds(train_ids, n_folds=a.inner_val, seed=0)[0]
        train_ids = [c for c in train_ids if c not in set(inner)]
        val_ids = inner
        config["inner_val"] = inner
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}"
          + ("  (inner validation)" if a.inner_val else ""))

    # What predict.py needs to rebuild the input and read the output.
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # With a seed, the loader's shuffling and its workers' generators are seeded too.
    loader_gen  = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    # Always, seeded or not: without it every worker starts from the same generator state,
    # every epoch, and the augmentation repeats in lock-step.
    worker_init = seed_worker

    train_loader = DataLoader(
        SDFDataset(a.cache_dir, train_ids, augment=True, seed=a.seed, factor=a.factor, cube=cube),
        batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)
    val_loader = DataLoader(
        SDFDataset(a.cache_dir, val_ids, augment=False, seed=a.seed, factor=a.factor, cube=cube),
        batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
        pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)

    # Adam, with the learning rate decaying along a cosine to 0 over the run.
    model = UNet3D(in_channels=1, base_channels=a.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_dice,val_boundary_dice\n")

    prof = Profiler("train")
    print(f"Model: {model_summary(model)}")

    best_dice, best_epoch = -1.0, 0
    with prof:
        for epoch in range(1, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, optimizer, device,
                                         a.sw_coef, a.sw_sigma)
                s["items"] = len(train_loader.dataset)

            # Logged for the curves only; nothing is chosen from it, except with an inner
            # validation set, which keeps the epoch with its best Dice.
            with prof.span("val_epoch"):
                val_loss, val_mae, val_dice, val_bdice = val_epoch(
                    model, val_loader, device, band_mm=a.band_mm)
            if a.inner_val and val_dice > best_dice:
                best_dice, best_epoch = val_dice, epoch
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1_best.pth"))

            scheduler.step()

            print(f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}  val_dice={val_dice:.4f}"
                  f"  val_boundary_dice={val_bdice:.4f}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_mae:.4f},{val_dice:.4f},"
                        f"{val_bdice:.4f}\n")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1_last.pth"))
    if a.inner_val:
        with open(os.path.join(run_dir, "best_epoch.json"), "w") as f:
            json.dump({"epoch": best_epoch, "inner_val_dice": best_dice}, f)
        print(f"\nBest inner-validation epoch: {best_epoch} (Dice {best_dice:.4f})")
    print(f"\nCheckpoint : {checkpoint_dir}")
    print(f"Log        : {log_path}")


if __name__ == "__main__":
    main()
