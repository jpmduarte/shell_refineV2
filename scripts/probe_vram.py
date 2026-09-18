"""Measure what a phase 2 configuration actually costs in VRAM.

The queue schedules on this rather than on a number somebody guessed in a job header.
One forward and one backward at the real shape, peak reserved reported — reserved, not
allocated, because the caching allocator is what actually holds memory away from other
processes.

    python scripts/probe_vram.py --shape 8 64 64 64      # cube 64^3 x8
    python scripts/probe_vram.py --shape 4 128 128 32    # slab 128x128x32 x4

Prints one line the caller parses:  PEAK_MB <n>
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import UNet3D  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", type=int, nargs=4, required=True,
                   metavar=("N", "D", "H", "W"), help="batch and spatial extent")
    p.add_argument("--in-channels", type=int, default=2)
    p.add_argument("--base-ch", type=int, default=16)
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("PEAK_MB 0")
        return

    n, d, h, w = a.shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet3D(in_channels=a.in_channels, base_channels=a.base_ch,
                   out_activation="sigmoid").cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(n, a.in_channels, d, h, w, device="cuda")
    y = torch.rand(n, 1, d, h, w, device="cuda")

    # Two steps, not one: Adam allocates its moment buffers on the first step, so a
    # single iteration understates the steady-state peak.
    for _ in range(2):
        loss = torch.nn.functional.binary_cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    print(f"PEAK_MB {int(torch.cuda.max_memory_reserved() / (1024 ** 2))}")


if __name__ == "__main__":
    main()
