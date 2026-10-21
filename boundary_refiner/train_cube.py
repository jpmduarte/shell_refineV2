"""Train the boundary refiner on one fold, on cubes from the band crops; the reported checkpoint is the last epoch.

Every --resume-every epochs the full training state (network, optimiser, schedule, epoch, random
generators) goes to <run-dir>/checkpoints/resume.pth; rerunning the same command continues from it
exactly as an uninterrupted run would. The network size is saved in <run-dir>/config.json for predict.py.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from boundary_refiner.dataset import CUBE, PATCHES_PER_CASE, CubeDataset, collate_cases
from boundary_refiner.refiner_model import RefinerUNet
from boundary_refiner.train import train_epoch, val_epoch
from common.folds import SEED as FOLDS_SEED, fold_split, get_folds
from common.paths import REFINER_DIR, SPLITS_JSON
from common.profiler import Profiler, model_summary
from common.seeding import loader_generator, make_deterministic, seed_worker

LOG_HEADER = "epoch,train_loss,val_loss,val_band_dice\n"

USAGE = """\
example:
  python boundary_refiner/train_cube.py --fold 0 --seed 42 --crops-dir boundary_refiner/crops/x4_s42 \\
      --run-dir boundary_refiner/runs/x4_s42_cube/fold0"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--crops-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cube", type=int, default=CUBE, help="cube side, a multiple of 8")
    parser.add_argument("--patches-per-case", type=int, default=PATCHES_PER_CASE)
    parser.add_argument("--batch-size", type=int, default=1, help="cases per step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--depth", type=int, default=3, help="pooling levels")
    parser.add_argument("--kernel", type=int, default=3, help="convolution kernel size")
    parser.add_argument("--base-ch", type=int, default=16, help="channels at the top level")
    parser.add_argument("--sw-sigma", type=float, default=2.0,
                        help="mm; BCE weighted by exp(-|prior|/sigma) inside the band; 0 = binary band")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-every", type=int, default=10,
                        help="save a resume checkpoint every this many epochs; 0 = never")
    parser.add_argument("--splits-json", type=str, default=SPLITS_JSON)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    if args.cube % 2 ** args.depth:
        parser.error(f"--cube must be divisible by {2 ** args.depth} ({args.depth} pooling levels)")
    if args.kernel % 2 == 0:
        parser.error("--kernel must be odd")
    return args


def available_cases(crops_dir) -> list[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(crops_dir) if f.endswith(".npz"))


def split_cases(args) -> tuple[list[str], list[str]]:
    """The shared patient-grouped folds, restricted to the cases that have a crop."""
    available = available_cases(args.crops_dir)
    folds = get_folds(available, n_folds=args.n_folds, seed=FOLDS_SEED, splits_json=args.splits_json)
    train_ids, val_ids = fold_split(folds, args.fold)
    return [c for c in train_ids if c in set(available)], [c for c in val_ids if c in set(available)]


def save_resume(path, epoch, model, optimizer, scheduler, generator, train_set):
    """Written to a temporary file and renamed, so an interruption cannot leave a broken checkpoint."""
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_gen": generator.get_state() if generator is not None else None,
        "dataset_rng": train_set._rng.bit_generator.state,   # used when --workers 0
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def resume(path, model, optimizer, scheduler, generator, train_set) -> int:
    """Restore a resume checkpoint; returns the epoch it was saved after."""
    # On the CPU, since generator states must be CPU byte tensors; cloned, since loaded tensors share storage.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"].clone())
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.clone() for s in state["cuda_rng"]])
    if generator is not None and state["loader_gen"] is not None:
        generator.set_state(state["loader_gen"].clone())
    train_set._rng.bit_generator.state = state["dataset_rng"]
    return state["epoch"]


def truncate_log(log_path, last_epoch):
    """Keep the header and the rows up to last_epoch; later ones are about to be redone."""
    with open(log_path) as f:
        header, *rows = f.readlines()
    with open(log_path, "w") as f:
        f.writelines([header] + [r for r in rows if int(r.split(",")[0]) <= last_epoch])


def main():
    args = parse_args()
    run_dir = args.run_dir or os.path.join(REFINER_DIR, "runs", f"cube_s{args.seed}", f"fold{args.fold}")
    os.makedirs(run_dir, exist_ok=True)

    config = {"depth": args.depth, "kernel": args.kernel, "base_ch": args.base_ch, "cube": args.cube}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    if args.seed is not None:
        make_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Fold: {args.fold}  Run: {run_dir}")
    print(f"Cubes: {args.cube}^3 x {args.patches_per_case} per case  "
          f"({args.patches_per_case * args.cube ** 3:,} voxels per case)  crops: {args.crops_dir}")

    train_ids, val_ids = split_cases(args)
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

    generator = loader_generator(args.seed)
    worker_init = seed_worker if args.seed is not None else None
    dataset_options = dict(cube=args.cube, patches_per_case=args.patches_per_case, seed=args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                          collate_fn=collate_cases, generator=generator, worker_init_fn=worker_init)
    train_set = CubeDataset(args.crops_dir, train_ids, augment=True, **dataset_options)
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(CubeDataset(args.crops_dir, val_ids, augment=False, **dataset_options),
                            shuffle=False, **loader_options)

    model = RefinerUNet(base_channels=args.base_ch, depth=args.depth, kernel=args.kernel).to(device)
    print(f"Network: depth {args.depth}, kernel {args.kernel}, base channels {args.base_ch}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Model: {model_summary(model)}")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(checkpoint_dir, "resume.pth")
    log_path = os.path.join(run_dir, "train_log.csv")

    start_epoch = 1
    if os.path.exists(resume_path):
        last_epoch = resume(resume_path, model, optimizer, scheduler, generator, train_set)
        start_epoch = last_epoch + 1
        truncate_log(log_path, last_epoch)
        print(f"Resuming after epoch {last_epoch}")
    else:
        with open(log_path, "w") as f:
            f.write(LOG_HEADER)

    profiler = Profiler("refiner_train")
    with profiler:
        for epoch in range(start_epoch, args.epochs + 1):
            with profiler.span("train_epoch") as span:
                train_loss = train_epoch(model, train_loader, optimizer, device, sw_sigma=args.sw_sigma)
                span["items"] = len(train_ids) * args.patches_per_case

            with profiler.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, device)

            scheduler.step()

            print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_band_dice={val_dice:.4f}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

            if args.resume_every and epoch % args.resume_every == 0 and epoch < args.epochs:
                save_resume(resume_path, epoch, model, optimizer, scheduler, generator, train_set)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase2_last.pth"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"\nCheckpoint : {checkpoint_dir}")


if __name__ == "__main__":
    main()
