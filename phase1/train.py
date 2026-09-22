"""Phase 1 training, one fold at a time. Run from anywhere; paths resolve relative to
this file and its parent (shared cache/splits.json), not the working directory.

    python phase1/train.py --fold 0
    python phase1/train.py --fold 0 --loss l1 --gd-coef 0.1
    python phase1/train.py --fold 0 --run-dir phase1/runs/my_run/fold0
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from cache import case_ids  # noqa: E402
from dataset import Phase1Dataset  # noqa: E402
from loss import gradient_difference_loss, get_loss  # noqa: E402
from metrics import boundary_dice_from_sdf, dice_from_sdf, mae  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def seed_worker(worker_id):
    # Dataset keeps its own np.random.Generator (self._rng), not the legacy global
    # numpy RNG — a fork copies that Generator's state byte-for-byte into every worker,
    # so without this every worker would draw the identical "random" sequence.
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed % 2**32)


def train_epoch(model, loader, loss_fn, optimizer, device, gd_coef: float) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        if gd_coef > 0:
            loss = loss + gd_coef * gradient_difference_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, band_mm: float) -> tuple[float, float, float, float]:
    model.eval()
    total_loss   = 0.0
    mae_scores   = []
    dice_scores  = []
    bdice_scores = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item() * x.size(0)

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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold",        type=int,   required=True, help="0-4, which fold is validation")
    p.add_argument("--n-folds",     type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=250)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--workers",     type=int,   default=2)
    p.add_argument("--base-ch",     type=int,   default=16)
    p.add_argument("--loss",        type=str,   default="l1")
    p.add_argument("--gd-coef",     type=float, default=0.0,
                   help="0.0 (default) = off. Weight of gradient_difference_loss added on top of --loss")
    p.add_argument("--band-mm",     type=float, default=5.0,
                   help="half-width in mm of the boundary band used for checkpoint selection")
    p.add_argument("--swa",         action="store_true",
                   help="average the weights of the last epochs instead of picking one. "
                        "Boundary Dice still swings visibly from epoch to epoch after 200, "
                        "so the best of 250 is partly whichever epoch flattered 18 "
                        "validation cases; a weight average also tends to land somewhere "
                        "flatter than any single iterate")
    p.add_argument("--swa-start",   type=int,   default=0,
                   help="epoch to start averaging from; 0 means 80%% of the budget")
    p.add_argument("--swa-lr",      type=float, default=1e-4,
                   help="constant rate during averaging, so the iterates keep moving")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--cache-dir",   type=str,   default=os.path.join(PARENT, "cache"))
    p.add_argument("--splits-json", type=str,   default=os.path.join(PARENT, "splits.json"))
    p.add_argument("--run-dir",     type=str,   default=None,
                   help="default: runs/fold<k> under phase1/")
    a = p.parse_args()

    run_dir = a.run_dir or os.path.join(HERE, "runs", f"fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = get_loss(a.loss)
    print(f"Device: {device}  Loss: {a.loss}  gd_coef: {a.gd_coef}  Fold: {a.fold}  Run: {run_dir}")

    ids   = case_ids(a.cache_dir)
    folds = get_folds(ids, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    loader_gen  = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    worker_init = seed_worker if a.seed is not None else None

    train_loader = DataLoader(
        Phase1Dataset(a.cache_dir, train_ids, augment=True, seed=a.seed),
        batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)
    val_loader = DataLoader(
        Phase1Dataset(a.cache_dir, val_ids, augment=False, seed=a.seed),
        batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
        pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)

    model     = UNet3D(in_channels=1, base_channels=a.base_ch, out_activation="tanh").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)

    # Cosine annealing until SWA starts, then a constant moderate rate. Letting cosine
    # run to zero would make the last epochs' weights nearly identical, and averaging
    # identical weights achieves nothing — SWA works because the iterates differ and
    # their mean lands somewhere flatter than any of them.
    swa_start = a.swa_start if a.swa_start > 0 else int(a.epochs * 0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=swa_start if a.swa else a.epochs)
    swa_model = AveragedModel(model) if a.swa else None
    swa_sched = SWALR(optimizer, swa_lr=a.swa_lr, anneal_epochs=5) if a.swa else None

    checkpoint_dir  = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase1_best.pth")
    log_path        = os.path.join(run_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_dice,val_boundary_dice\n")

    best_boundary_dice = -1.0
    prof = Profiler("phase1_train")
    print(f"Model: {model_summary(model)}")

    with prof:
        for epoch in range(1, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, a.gd_coef)
                s["items"] = len(train_loader.dataset)

            with prof.span("val_epoch"):
                val_loss, val_mae, val_dice, val_bdice = val_epoch(
                    model, val_loader, loss_fn, device, band_mm=a.band_mm)

            if a.swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_sched.step()
            else:
                scheduler.step()

            improved = val_bdice > best_boundary_dice
            if improved:
                best_boundary_dice = val_bdice
                torch.save(model.state_dict(), checkpoint_path)

            line = (f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                   f"  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}  val_dice={val_dice:.4f}"
                   f"  val_boundary_dice={val_bdice:.4f}"
                   + ("  *" if improved else ""))
            print(line, flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_mae:.4f},{val_dice:.4f},{val_bdice:.4f}\n")

    # Three candidates, so the choice between them is made on numbers rather than on
    # an argument about which selection rule is sounder.
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1_last.pth"))

    if a.swa:
        # BatchNorm's running statistics are not weights and were never averaged, so
        # they belong to no particular iterate. One pass over the training data with the
        # averaged weights recomputes them.
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(),
                   os.path.join(checkpoint_dir, "phase1_swa.pth"))
        swa_loss, swa_mae, swa_dice, swa_bdice = val_epoch(
            swa_model.module, val_loader, loss_fn, device, band_mm=a.band_mm)
        print(f"\nSWA (from epoch {swa_start}): val_mae={swa_mae:.4f} "
              f"val_dice={swa_dice:.4f} val_boundary_dice={swa_bdice:.4f}")

    print(f"\nBest val boundary Dice : {best_boundary_dice:.4f}")
    print(f"Checkpoints  : {checkpoint_dir}")
    print(f"Log          : {log_path}")


if __name__ == "__main__":
    main()
