"""Train the network on one fold; the reported checkpoint is the last epoch, so nothing is chosen on the reported fold."""
import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.augmentation import SCALE_P, SCALE_RANGE
from common.folds import SEED as FOLDS_SEED, fold_split, get_folds, make_folds
from common.image_cache import case_ids
from common.loss import SURFACE_SIGMA_MM, surface_weighted_l1
from common.metrics import BOUNDARY_BAND_MM, boundary_dice_from_sdf, dice_from_sdf, mae
from common.model import UNet3D
from common.paths import NORM_CACHE, SIZE_NORM_RUNS, SPLITS_JSON
from common.profiler import Profiler, model_summary
from common.seeding import loader_generator, make_deterministic, seed_worker
from size_normalization.dataset import SDFDataset

INNER_VAL_SEED = 0   # the same held-out cases for every seed of a fold
LOG_HEADER = "epoch,train_loss,val_loss,val_mae,val_dice,val_boundary_dice,train_l1,train_sw\n"

USAGE = """\
examples:
  the size-normalised network (pass 2, on caches/cache_norm4):
    python size_normalization/train.py --fold 0 --seed 42 --run-dir size_normalization/runs/x4norm_s42/fold0
  the plain x4 network (pass 1, which estimates the head volume):
    python size_normalization/train.py --fold 0 --seed 42 --cache-dir caches/cache \\
        --run-dir size_normalization/runs/x4_s42/fold0

writes <run-dir>/config.json, train_log.csv (one row per epoch) and checkpoints/phase1_last.pth"""


@dataclass
class TrainScores:
    loss: float
    l1: float
    sw: float   # the surface-weighted L1, unweighted by its coefficient


@dataclass
class ValScores:
    loss: float
    mae: float
    dice: float
    boundary_dice: float


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def train_epoch(model, loader, optimizer, device, sw_coef: float, sw_sigma: float) -> TrainScores:
    """One pass over the training cases; every score is a mean per case."""
    model.train()
    total_loss = total_l1 = total_sw = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        batch = x.size(0)

        l1 = F.l1_loss(pred, y)
        loss = l1
        if sw_coef > 0:
            sw = surface_weighted_l1(pred, y, sw_sigma)
            loss = loss + sw_coef * sw
            total_sw += sw.item() * batch
        total_l1 += l1.item() * batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch

    n = len(loader.dataset)
    return TrainScores(total_loss / n, total_l1 / n, total_sw / n)


@torch.no_grad()
def val_epoch(model, loader, device, band_mm: float) -> ValScores:
    """For the training curves only; Dice is skipped for a case where it is undefined."""
    model.eval()
    total_loss = 0.0
    mae_scores, dice_scores, boundary_scores = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += F.l1_loss(pred, y).item() * x.size(0)

        for case_pred, case_target in zip(pred, y):
            mae_scores.append(mae(case_pred, case_target))
            score = dice_from_sdf(case_pred, case_target)
            if score is not None:
                dice_scores.append(score)
            boundary_score = boundary_dice_from_sdf(case_pred, case_target, band_mm=band_mm)
            if boundary_score is not None:
                boundary_scores.append(boundary_score)

    return ValScores(total_loss / len(loader.dataset), mean_or_zero(mae_scores),
                     mean_or_zero(dice_scores), mean_or_zero(boundary_scores))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, help="0-4, which fold is validation")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2,
                        help="data-loader processes; 6 measured slower on this machine")
    parser.add_argument("--sw-coef", type=float, default=0.5,
                        help="weight of the surface-weighted L1 added to the plain L1; 0 = off")
    parser.add_argument("--sw-sigma", type=float, default=SURFACE_SIGMA_MM,
                        help="mm over which that weight decays away from the surface")
    parser.add_argument("--base-ch", type=int, default=16,
                        help="channels at the top level, doubling per level (32: 32-64-128, ~4x parameters)")
    parser.add_argument("--band-mm", type=float, default=BOUNDARY_BAND_MM,
                        help="half-width in mm of the band for the logged boundary Dice")
    parser.add_argument("--scale-p", type=float, default=SCALE_P,
                        help="probability of the augmentation's isotropic scaling (nnU-Net: 0.2)")
    parser.add_argument("--scale-range", type=float, nargs=2, default=list(SCALE_RANGE),
                        help="its factor range; > 1 shrinks the head (nnU-Net: 0.7 1.4). The "
                             "strong-scaling baseline: --scale-p 0.5 --scale-range 0.6 1.7")
    parser.add_argument("--inner-val", type=int, default=0,
                        help="N > 0: hold out 1/N of the training cases (patient-grouped, the same for "
                             "every seed) as an inner validation set, validate on it instead of the "
                             "fold, and also save the epoch with its best Dice as phase1_best.pth")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=NORM_CACHE)
    parser.add_argument("--splits-json", type=str, default=SPLITS_JSON)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="default: size_normalization/runs/fold<k>")
    return parser.parse_args()


def split_cases(args) -> tuple[list[str], list[str], list[str] | None]:
    """(training, validation, the inner validation set or None)."""
    folds = get_folds(case_ids(args.cache_dir), n_folds=args.n_folds, seed=FOLDS_SEED,
                      splits_json=args.splits_json)
    train_ids, val_ids = fold_split(folds, args.fold)
    if not args.inner_val:
        return train_ids, val_ids, None
    # Out of the training cases, by patient: the best epoch is chosen without seeing the reported fold.
    inner = make_folds(train_ids, n_folds=args.inner_val, seed=INNER_VAL_SEED)[0]
    train_ids = [c for c in train_ids if c not in set(inner)]
    return train_ids, inner, inner


def make_loader(dataset, args, shuffle, generator):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=True, generator=generator, worker_init_fn=seed_worker)


def log_row(epoch, train: TrainScores, val: ValScores) -> str:
    return (f"{epoch},{train.loss:.4f},{val.loss:.4f},{val.mae:.4f},{val.dice:.4f},"
            f"{val.boundary_dice:.4f},{train.l1:.4f},{train.sw:.4f}\n")


def main():
    args = parse_args()
    run_dir = args.run_dir or os.path.join(SIZE_NORM_RUNS, f"fold{args.fold}")
    os.makedirs(run_dir, exist_ok=True)

    if args.seed is not None:
        make_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {args.fold}  Run: {run_dir}")
    print(f"Loss: L1 + {args.sw_coef} x surface-weighted L1 @ {args.sw_sigma}mm  cache: {args.cache_dir}")

    train_ids, val_ids, inner = split_cases(args)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}" + ("  (inner validation)" if inner else ""))

    config = {"base_ch": args.base_ch, "scale_p": args.scale_p, "scale_range": args.scale_range}
    if inner:
        config["inner_val"] = inner
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f)

    generator = loader_generator(args.seed)
    train_loader = make_loader(SDFDataset(args.cache_dir, train_ids, augment=True, seed=args.seed,
                                          scale_p=args.scale_p, scale_range=args.scale_range),
                               args, shuffle=True, generator=generator)
    val_loader = make_loader(SDFDataset(args.cache_dir, val_ids, augment=False, seed=args.seed),
                             args, shuffle=False, generator=generator)

    model = UNet3D(in_channels=1, base_channels=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write(LOG_HEADER)

    profiler = Profiler("train")
    print(f"Model: {model_summary(model)}")

    best_dice, best_epoch = -1.0, 0
    with profiler:
        for epoch in range(1, args.epochs + 1):
            with profiler.span("train_epoch") as span:
                train = train_epoch(model, train_loader, optimizer, device, args.sw_coef, args.sw_sigma)
                span["items"] = len(train_loader.dataset)

            with profiler.span("val_epoch"):
                val = val_epoch(model, val_loader, device, band_mm=args.band_mm)
            if inner and val.dice > best_dice:
                best_dice, best_epoch = val.dice, epoch
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1_best.pth"))

            scheduler.step()

            print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={train.loss:.4f}"
                  f"  val_loss={val.loss:.4f}  val_mae={val.mae:.4f}  val_dice={val.dice:.4f}"
                  f"  val_boundary_dice={val.boundary_dice:.4f}", flush=True)
            with open(log_path, "a") as f:
                f.write(log_row(epoch, train, val))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1_last.pth"))
    if inner:
        with open(os.path.join(run_dir, "best_epoch.json"), "w") as f:
            json.dump({"epoch": best_epoch, "inner_val_dice": best_dice}, f)
        print(f"\nBest inner-validation epoch: {best_epoch} (Dice {best_dice:.4f})")
    print(f"\nCheckpoint : {checkpoint_dir}")
    print(f"Log        : {log_path}")


if __name__ == "__main__":
    main()
