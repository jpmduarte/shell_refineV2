"""Phase 2 training: refine the band with a dense 3D U-Net on 64^3 patches.

Patches are drawn at random band-centred positions each epoch (see dataset.py).
Inference tiles deterministically instead — that is predict.py's job.

    python phase2/train.py --fold 0 --crops-dir phase2/crops/all
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

from dataset import collate_patches, PATCH_SIZE, PATCHES_PER_CASE, Phase2Dataset  # noqa: E402
from metrics import dice  # noqa: E402
from model import UNet3D  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def dice_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
             eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice + BCE on the sigmoid output (v1's best phase 2 loss).

    With `weight` (the band), every voxel outside it contributes exactly zero. A 64^3
    cube around a ~25-voxel-thick shell is mostly deep interior and exterior, which the
    prior already gets right — scoring them dilutes the gradient with voxels the
    refiner is not allowed to change anyway. v1 tested the weak form of this
    (down-weighting non-band voxels to a floor) and measured no benefit; masking them
    out entirely is the untested strong form.
    """
    bce_map = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")

    if weight is None:
        inter = (pred * target).sum()
        soft_dice = 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)
        return soft_dice + bce_map.mean()

    inter = (pred * target * weight).sum()
    denom = (pred * weight).sum() + (target * weight).sum()
    soft_dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    bce = (bce_map * weight).sum() / weight.sum().clamp_min(1.0)
    return soft_dice + bce


def seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset._seed_rng = np.random.default_rng(info.seed % 2**32)


def train_epoch(model, loader, optimizer, device, band_masked: bool) -> float:
    model.train()
    total, seen = 0.0, 0
    for x, y, b in loader:
        x, y, b = x.to(device), y.to(device), b.to(device)
        pred = model(x)
        loss = dice_bce(pred, y, weight=b if band_masked else None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        seen  += x.size(0)
    return total / seen


@torch.no_grad()
def val_epoch(model, loader, device, band_masked: bool) -> tuple[float, float]:
    model.eval()
    total, seen = 0.0, 0
    dice_scores = []
    for x, y, b in loader:
        x, y, b = x.to(device), y.to(device), b.to(device)
        pred = model(x)
        total += dice_bce(pred, y, weight=b if band_masked else None).item() * x.size(0)
        seen  += x.size(0)
        # Dice restricted to the band, matching what the refiner is scored on and what
        # evaluate.py reports as "band" — a whole-patch Dice would be dominated by the
        # easy interior the prior already had right.
        for p, t, bb in zip(pred, y, b):
            sel = bb > 0.5
            if sel.any():
                score = dice(p[sel], t[sel])
                if score is not None:
                    dice_scores.append(score)
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return total / seen, mean_dice


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold",             type=int,   required=True)
    p.add_argument("--n-folds",          type=int,   default=5)
    p.add_argument("--crops-dir",        type=str,   required=True,
                   help="make_crops.py output covering every case in the fold")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=1,
                   help="cases per batch; each contributes --patches-per-case patches")
    p.add_argument("--patches-per-case", type=int,   default=PATCHES_PER_CASE)
    p.add_argument("--patch-size",       type=int,   default=PATCH_SIZE)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--workers",          type=int,   default=2)
    p.add_argument("--base-ch",          type=int,   default=16)
    p.add_argument("--no-band-masked-loss", action="store_true",
                   help="score every voxel in the patch, not just band voxels (ablation)")
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--splits-json",      type=str,   default=os.path.join(REPO_ROOT, "splits.json"))
    p.add_argument("--run-dir",          type=str,   default=None)
    a = p.parse_args()

    run_dir = a.run_dir or os.path.join(HERE, "runs", f"fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    band_masked = not a.no_band_masked_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {a.fold}  Crops: {a.crops_dir}  Run: {run_dir}")
    print(f"Loss: dice_bce  band_masked={band_masked}")

    available = sorted(f[:-4] for f in os.listdir(a.crops_dir) if f.endswith(".npz"))
    folds = get_folds(available, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    train_ids = [c for c in train_ids if c in set(available)]
    val_ids   = [c for c in val_ids   if c in set(available)]
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  "
          f"({a.patches_per_case} patches/case, {a.patch_size}^3)")

    loader_gen  = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    worker_init = seed_worker if a.seed is not None else None

    common = dict(patch_size=a.patch_size, patches_per_case=a.patches_per_case, seed=a.seed)
    train_loader = DataLoader(
        Phase2Dataset(a.crops_dir, train_ids, augment=True, **common),
        batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        pin_memory=True, collate_fn=collate_patches,
        generator=loader_gen, worker_init_fn=worker_init)
    # deterministic: the same patches every epoch, so val moves only when the model
    # does. Without it the metric carries the noise of resampling and "best epoch"
    # partly selects the kindest draw.
    val_loader = DataLoader(
        Phase2Dataset(a.crops_dir, val_ids, augment=False, deterministic=True, **common),
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
        f.write("epoch,train_loss,val_loss,val_patch_dice\n")

    best_dice = -1.0
    prof = Profiler("phase2_train")
    with prof:
        for epoch in range(1, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, optimizer, device, band_masked)
                s["items"] = len(train_loader.dataset) * a.patches_per_case

            with prof.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, device, band_masked)

            scheduler.step()

            # Patch Dice, not whole-case Dice: cheap per-epoch proxy for model
            # selection. evaluate.py reports the number that actually matters,
            # stitched over the full volume against phase 1's baseline.
            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                torch.save(model.state_dict(), checkpoint_path)

            print(f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_patch_dice={val_dice:.4f}"
                  + ("  *" if improved else ""), flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    print(f"\nBest val patch Dice : {best_dice:.4f}")
    print(f"Checkpoint          : {checkpoint_path}")


if __name__ == "__main__":
    main()
