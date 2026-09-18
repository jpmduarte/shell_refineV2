"""Phase 2 training on the band, one fold at a time. Behavior check: does the sparse
network learn the real segmentation task, using the true native SDF as the band prior
(swap in phase 1's prediction via predict.py's saved output once available).

    python phase2/train.py --fold 0
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from cache import case_ids  # noqa: E402
from dataset import BAND_MM, collate, Phase2Dataset  # noqa: E402
from model import SparseRefiner  # noqa: E402
from profiler import Profiler, model_summary  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402


def dice_from_logits(logit: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logit) > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2 * inter + eps) / (union + eps)


def train_epoch(model, loader, optimizer, device, loss_fn) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for coords, features, target in loader:
        coords, features, target = coords.to(device), features.to(device), target.to(device)
        logit = model(coords, features)
        loss = loss_fn(logit, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * coords.shape[0]
        n += coords.shape[0]
    return total_loss / n


@torch.no_grad()
def val_epoch(model, loader, device, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    dice_scores = []
    for coords, features, target in loader:
        coords, features, target = coords.to(device), features.to(device), target.to(device)
        logit = model(coords, features)
        loss = loss_fn(logit, target)
        total_loss += loss.item() * coords.shape[0]
        n += coords.shape[0]
        dice_scores.append(dice_from_logits(logit, target))
    return total_loss / n, sum(dice_scores) / len(dice_scores)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold",        type=int,   required=True)
    p.add_argument("--n-folds",     type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--width",       type=int,   default=32)
    p.add_argument("--depth",       type=int,   default=4)
    p.add_argument("--band-mm",     type=float, default=BAND_MM)
    p.add_argument("--cache-dir",   type=str,   default=os.path.join(REPO_ROOT, "cache"))
    p.add_argument("--splits-json", type=str,   default=os.path.join(REPO_ROOT, "splits.json"))
    p.add_argument("--run-dir",     type=str,   default=None)
    a = p.parse_args()

    run_dir = a.run_dir or os.path.join(HERE, "runs", f"fold{a.fold}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {a.fold}  band_mm: {a.band_mm}  Run: {run_dir}")

    ids   = case_ids(a.cache_dir)
    folds = get_folds(ids, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    train_loader = DataLoader(Phase2Dataset(a.cache_dir, train_ids, band_mm=a.band_mm),
                              batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Phase2Dataset(a.cache_dir, val_ids, band_mm=a.band_mm),
                            batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    model     = SparseRefiner(in_channels=2, width=a.width, depth=a.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn   = torch.nn.BCEWithLogitsLoss()

    print(f"Model: {model_summary(model)}")

    checkpoint_path = os.path.join(run_dir, "phase2_best.pth")
    log_path        = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice\n")

    best_dice = -1.0
    prof = Profiler("phase2_train")
    with prof:
        for epoch in range(1, a.epochs + 1):
            with prof.span("train_epoch") as s:
                train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
                s["items"] = len(train_loader.dataset)

            with prof.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, device, loss_fn)

            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                torch.save(model.state_dict(), checkpoint_path)

            line = (f"Epoch {epoch:03d}/{a.epochs}  train_loss={train_loss:.4f}"
                   f"  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}"
                   + ("  *" if improved else ""))
            print(line, flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    print(f"\nBest val Dice: {best_dice:.4f}")
    print(f"Checkpoint   : {checkpoint_path}")


if __name__ == "__main__":
    main()
