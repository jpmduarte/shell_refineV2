# Measures the GPU memory a training step really needs, for the job queue to schedule on.
#
#     python scripts/probe_vram.py                       # batch 4 at 96^3, the training shape
#     python scripts/probe_vram.py --shape 8 96 96 96
#
# Runs two forward and backward steps at the given shape and prints one line the queue
# parses:  PEAK_MB <n>
# It reports memory reserved, not allocated: the reserve is what PyTorch actually holds
# away from other processes.

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.model import UNet3D  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Measure a training step's peak GPU memory.")
    p.add_argument("--shape", type=int, nargs=4, default=[4, 96, 96, 96],
                   metavar=("N", "D", "H", "W"), help="batch size and spatial size")
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("PEAK_MB 0")
        return

    n, d, h, w = a.shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet3D().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Random input and a random target in [-1, 1], like an SDF.
    x = torch.randn(n, 1, d, h, w, device="cuda")
    y = torch.rand(n, 1, d, h, w, device="cuda") * 2 - 1

    # Two steps, not one: Adam allocates its state on the first step, so one step
    # understates the steady-state peak.
    for _ in range(2):
        loss = F.l1_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    print(f"PEAK_MB {int(torch.cuda.max_memory_reserved() / (1024 ** 2))}")


if __name__ == "__main__":
    main()
