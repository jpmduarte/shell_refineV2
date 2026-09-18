"""Phase 2 training on normal-aligned slabs instead of cubes.

Reuses train.py's loss and epoch loops unchanged — they only ever see (x, y, band)
tensors and never assume a cubic shape — so the only thing that differs here is how
patches are cut. See slab.py for why the band is better matched by a slab.

The default (128x128x32, 4 per case) is chosen to match a 64^3 x8 cube run on BOTH
compute and supervision: ~2.10M total voxels and ~0.91M band voxels per step in each.
That leaves patch shape as the only variable, which the isotropic patch-size sweep
could not do: growing a cube changes the band fraction too.

    python phase2/train_slab.py --fold 0 --crops-dir phase2/crops/all
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from dataset import collate_patches  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from slab import LONG, SlabDataset, THIN  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402
from train import train_epoch, val_epoch  # noqa: E402


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._seed_rng = np.random.default_rng(info.seed % 2**32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold",             type=int,   required=True)
    p.add_argument("--n-folds",          type=int,   default=5)
    p.add_argument("--crops-dir",        type=str,   required=True)
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=1)
    p.add_argument("--long",             type=int,   default=LONG)
    p.add_argument("--thin",             type=int,   default=THIN)
    p.add_argument("--patches-per-case", type=int,   default=4)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--workers",          type=int,   default=2)
    p.add_argument("--base-ch",          type=int,   default=16)
    p.add_argument("--no-band-masked-loss", action="store_true")
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--splits-json",      type=str,   default=os.path.join(REPO_ROOT, "splits.json"))
    p.add_argument("--run-dir",          type=str,   default=None)
    a = p.parse_args()

    if a.long % 8 or a.thin % 8:
        p.error("--long and --thin must be divisible by 8 (three pooling levels)")

    run_dir = a.run_dir or os.path.join(HERE, "runs", f"slab_fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    band_masked = not a.no_band_masked_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {a.fold}  Run: {run_dir}")
    print(f"Slab: {a.long}x{a.long}x{a.thin} x{a.patches_per_case}/case"
          f"  ({a.patches_per_case * a.long * a.long * a.thin:,} voxels/step)"
          f"  band_masked={band_masked}")

    available = sorted(f[:-4] for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    folds = get_folds(available, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    train_ids = [c for c in train_ids if c in set(available)]
    val_ids   = [c for c in val_ids   if c in set(available)]
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    loader_gen  = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    worker_init = seed_worker if a.seed is not None else None
    common = dict(long=a.long, thin=a.thin, patches_per_case=a.patches_per_case, seed=a.seed)

    train_loader = DataLoader(
        SlabDataset(a.crops_dir, train_ids, augment=True, **common),
        batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        pin_memory=True, collate_fn=collate_patches,
        generator=loader_gen, worker_init_fn=worker_init)
    # See train.py: fixed validation patches. Slabs need it more, since resampling
    # changes which orientations get scored, not just where.
    val_loader = DataLoader(
        SlabDataset(a.crops_dir, val_ids, augment=False, deterministic=True, **common),
        batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
        pin_memory=True, collate_fn=collate_patches,
        generator=loader_gen, worker_init_fn=worker_init)

    model     = UNet3D(in_channels=2, base_channels=a.base_ch, out_activation="sigmoid").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    print(f"Model: {model_summary(model)}")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase2_best.pth")
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_band_dice\n")

    best_dice = -1.0
    prof = Profiler("phase2_slab_train")
    with prof:
        for epoch in range(1, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, optimizer, device, band_masked)
                s["items"] = len(train_loader.dataset) * a.patches_per_case

            with prof.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, device, band_masked)

            scheduler.step()

            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                torch.save(model.state_dict(), checkpoint_path)

            print(f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_band_dice={val_dice:.4f}"
                  + ("  *" if improved else ""), flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    print(f"\nBest val band Dice : {best_dice:.4f}")
    print(f"Checkpoint         : {checkpoint_path}")


if __name__ == "__main__":
    main()
