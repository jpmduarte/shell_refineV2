"""Measure the GPU memory a training step really needs, for the job queue to schedule on.

Runs two forward and backward steps at the given shape (Adam allocates its state on the first, so
one step understates the peak) and prints the line the queue parses, PEAK_MB <n>: memory reserved,
not allocated, since the reserve is what PyTorch holds away from other processes.
"""
import argparse

import torch
import torch.nn.functional as F

from common.geometry import CUBE
from common.model import UNet3D

MB = 1024 ** 2
STEPS = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=int, nargs=4, default=[4, CUBE, CUBE, CUBE],
                        metavar=("N", "D", "H", "W"), help="batch size and spatial size (default: training's)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("PEAK_MB 0")
        return

    n, d, h, w = args.shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet3D().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(n, 1, d, h, w, device="cuda")
    y = torch.rand(n, 1, d, h, w, device="cuda") * 2 - 1   # in [-1, 1], like an SDF

    for _ in range(STEPS):
        loss = F.l1_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    print(f"PEAK_MB {int(torch.cuda.max_memory_reserved() / MB)}")


if __name__ == "__main__":
    main()
