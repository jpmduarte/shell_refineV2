# Trains the size-normalised network with the v2 polar branch (polar_model.py) on one fold.
#
#     python polar_approach/train.py --fold 0 --seed 42 --run-dir size_normalization_approach/runs/x4normpolar2_s42/fold0
#     python polar_approach/train.py ... --cons-coef 0.1        # + the SDF-polar consistency
#
# The baseline recipe (size_normalization_approach/train.py: data, augmentation, L1 +
# 0.5 x surface-weighted L1, Adam 5e-4 cosine, 500 epochs, batch 4) plus:
#
#   polar loss    |log r_pred - log r_true| over the 64 directions from the head's centre
#                 (scale-free), weight --polar-coef, ramped up from 0 over --polar-warmup
#                 epochs so it cannot dominate before the SDF has useful features
#   consistency   from epoch --cons-start: |SDF(c + r_pred d)|, the SDF at the surface
#                 points the polar branch predicts, weight --cons-coef. Its gradient reaches
#                 the SDF only once the polar branch is good (last epoch's mean log error
#                 below --cons-gate); before that the SDF is detached, so the consistency
#                 only teaches the polar branch where the SDF puts the surface.
#
# Saves checkpoints/phase1_last.pth (the U-Net alone: infer.py and predict.py read it as
# any baseline network) and checkpoints/polar_last.pth (the polar branch). train_log.csv
# adds each term, the polar radius error in voxels, and whether the gate was open.

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNA = os.path.join(ROOT, "size_normalization_approach")
sys.path[:0] = [ROOT, SNA, HERE]

import polar  # noqa: E402
from cache import case_ids  # noqa: E402
from dataset import SDFDataset  # noqa: E402
from geometry import CUBE  # noqa: E402
from loss import surface_weighted_l1  # noqa: E402
from polar_dataset import PolarDataset  # noqa: E402
from polar_model import PolarNet  # noqa: E402
from splits import fold_split, get_folds  # noqa: E402
from train import seed_worker, val_epoch  # noqa: E402  (the baseline's)


# The SDF (B, 1, cube^3) at voxel points (B, D, 3).
def sdf_at(sdf, pts):
    cube = sdf.shape[-1]
    grid = ((2 * pts + 1) / cube - 1).flip(-1)[:, :, None, None, :]   # (B, D, 1, 1, 3) x y z
    return F.grid_sample(sdf, grid, align_corners=False, padding_mode="border")[:, 0, :, 0, 0]


def train_epoch(model, loader, optimizer, device, a, epoch, dirs, gate_open):
    model.train()
    sums = {"loss": 0.0, "l1": 0.0, "sw": 0.0, "polar_log": 0.0, "polar_vox": 0.0, "cons": 0.0}
    ramp = min(1.0, epoch / a.polar_warmup) if a.polar_warmup > 0 else 1.0
    use_cons = a.cons_coef > 0 and epoch >= a.cons_start
    for x, y, c, r, w in loader:
        x, y, c, r, w = (t.to(device) for t in (x, y, c, r, w))
        sdf, log_r = model(x, c)

        l1 = F.l1_loss(sdf, y)
        sw = surface_weighted_l1(sdf, y, a.sw_sigma)
        wsum = w.sum().clamp(min=1.0)
        polar_log = (w * (log_r - torch.log(r)).abs().mean(dim=1)).sum() / wsum
        loss = l1 + a.sw_coef * sw + a.polar_coef * ramp * polar_log

        if use_cons:
            # The surface points the polar branch predicts, (B, D, 3) voxels.
            pts = c[:, None, :] + torch.exp(log_r)[:, :, None] * dirs[None]
            field = sdf if gate_open else sdf.detach()
            cons = (w * sdf_at(field, pts).abs().mean(dim=1)).sum() / wsum
            loss = loss + a.cons_coef * cons
            sums["cons"] += cons.item() * x.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = x.size(0)
        sums["loss"] += loss.item() * n
        sums["l1"] += l1.item() * n
        sums["sw"] += sw.item() * n
        sums["polar_log"] += polar_log.item() * n
        sums["polar_vox"] += ((w * (torch.exp(log_r) - r).abs().mean(dim=1)).sum() / wsum).item() * n
    return {k: v / len(loader.dataset) for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser(description="Train the size-normalised network with the v2 polar branch.")
    p.add_argument("--fold",         type=int,   required=True)
    p.add_argument("--n-folds",      type=int,   default=5)
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--workers",      type=int,   default=2)
    p.add_argument("--sw-coef",      type=float, default=0.5)
    p.add_argument("--sw-sigma",     type=float, default=1.5)
    p.add_argument("--polar-coef",   type=float, default=0.1)
    p.add_argument("--polar-warmup", type=int,   default=50, help="epochs over which the polar weight ramps from 0")
    p.add_argument("--cons-coef",    type=float, default=0.0, help="SDF-polar consistency weight; 0 = off")
    p.add_argument("--cons-start",   type=int,   default=150)
    p.add_argument("--cons-gate",    type=float, default=0.05,
                   help="mean |log r| error (~5%%) below which the consistency also moves the SDF")
    p.add_argument("--band-mm",      type=float, default=5.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--cache-dir",    type=str,   default=os.path.join(ROOT, "caches", "cache_norm4"))
    p.add_argument("--splits-json",  type=str,   default=os.path.join(ROOT, "splits.json"))
    p.add_argument("--run-dir",      type=str,   required=True)
    a = p.parse_args()

    os.makedirs(a.run_dir, exist_ok=True)
    with open(os.path.join(a.run_dir, "config.json"), "w") as f:
        json.dump({"factor": 4, "cube": CUBE, "loss": "sdf", "base_ch": 16, "polar": "v2",
                   **{k: getattr(a, k) for k in ("polar_coef", "polar_warmup", "cons_coef",
                                                 "cons_start", "cons_gate")}}, f)

    if a.seed is not None:
        torch.manual_seed(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    ids = case_ids(a.cache_dir)
    folds = get_folds(ids, n_folds=a.n_folds, seed=42, splits_json=a.splits_json)
    train_ids, val_ids = fold_split(folds, a.fold)
    print(f"Fold {a.fold}  Train {len(train_ids)}  Val {len(val_ids)}  polar {a.polar_coef} "
          f"(warm-up {a.polar_warmup})  consistency {a.cons_coef} from {a.cons_start}, gate {a.cons_gate}")

    gen = torch.Generator().manual_seed(a.seed) if a.seed is not None else None
    train_loader = DataLoader(PolarDataset(a.cache_dir, train_ids, augment=True, seed=a.seed),
                              batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                              pin_memory=True, generator=gen, worker_init_fn=seed_worker)
    val_loader = DataLoader(SDFDataset(a.cache_dir, val_ids, augment=False, seed=a.seed),
                            batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                            pin_memory=True, generator=gen, worker_init_fn=seed_worker)

    model = PolarNet().to(device)
    dirs = torch.from_numpy(polar.directions()).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    ck = os.path.join(a.run_dir, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    log_path = os.path.join(a.run_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_dice,val_boundary_dice,train_l1,train_sw,"
                "train_polar_log,train_polar_vox,train_cons,gate_open\n")

    gate_open = False
    for epoch in range(1, a.epochs + 1):
        t = train_epoch(model, train_loader, optimizer, device, a, epoch, dirs, gate_open)
        val_loss, val_mae, val_dice, val_bdice = val_epoch(model.net, val_loader, device, band_mm=a.band_mm)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{a.epochs}  loss={t['loss']:.4f}  val_dice={val_dice:.4f}  "
              f"polar {t['polar_vox']:.2f} vox (log {t['polar_log']:.3f})  cons {t['cons']:.4f}"
              f"{'  gate open' if gate_open else ''}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{t['loss']:.4f},{val_loss:.4f},{val_mae:.4f},{val_dice:.4f},{val_bdice:.4f},"
                    f"{t['l1']:.4f},{t['sw']:.4f},{t['polar_log']:.4f},{t['polar_vox']:.3f},"
                    f"{t['cons']:.4f},{int(gate_open)}\n")
        # The gate for the next epoch: is the polar branch good enough to move the SDF?
        gate_open = t["polar_log"] < a.cons_gate

    torch.save(model.net.state_dict(), os.path.join(ck, "phase1_last.pth"))
    torch.save(model.mlp.state_dict(), os.path.join(ck, "polar_last.pth"))
    print(f"\nCheckpoints: {ck}")


if __name__ == "__main__":
    main()
