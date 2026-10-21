"""Run any command and sample what it costs the machine; writes the peaks, medians, wall time and trace to --out.

Sampled every --every seconds:
    GPU memory  the card's use minus its use before launch (Windows gives no per-process figure,
                so run with the GPU otherwise idle); PyTorch's reserve and the CUDA context included
    RAM         resident memory of the process and all its children
    CPU         the process tree's CPU use, in cores
    GPU util    nvidia-smi's utilisation, in %
"""
import argparse
import json
import subprocess
import time

import numpy as np
import psutil

MB = 2 ** 20
COLUMNS = ("time", "gpu_mb", "ram_mb", "cpu_cores", "gpu_util")

USAGE = """\
example:
  python scripts/profile_run.py --out prof.json -- python size_normalization/train.py ..."""


def gpu():
    """(the card's memory in use in MB, its utilisation in %), from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, util = out.strip().splitlines()[0].split(",")
    return float(used), float(util)


def process_tree(process):
    """A process and all its descendants (empty once it has exited)."""
    try:
        return [process] + process.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def sample_tree(root, cpu_previous) -> tuple[float, float]:
    """(resident bytes, CPU seconds since the previous sample) of the tree; a new process starts at 0."""
    rss, cpu = 0.0, 0.0
    for process in process_tree(root):
        try:
            rss += process.memory_info().rss
            cpu_now = sum(process.cpu_times()[:2])   # user + system
            cpu += cpu_now - cpu_previous.get(process.pid, cpu_now)
            cpu_previous[process.pid] = cpu_now
        except psutil.NoSuchProcess:
            pass   # it exited between listing and reading
    return rss, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="the command, after --")
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd

    gpu_baseline, _ = gpu()
    start = time.time()
    proc = subprocess.Popen(cmd)
    root = psutil.Process(proc.pid)

    cpu_previous = {}
    rows = []   # one tuple of COLUMNS per sample
    while proc.poll() is None:
        time.sleep(args.every)
        rss, cpu = sample_tree(root, cpu_previous)
        used, util = gpu()
        rows.append((time.time() - start, used - gpu_baseline, rss / MB, cpu / args.every, util))

    wall = time.time() - start
    samples = np.array(rows) if rows else np.zeros((1, len(COLUMNS)))
    result = {"cmd": " ".join(cmd), "exit": proc.returncode, "wall_s": wall,
              "gpu_baseline_mb": gpu_baseline, "samples": len(rows)}
    for column, name in enumerate(COLUMNS):
        if name == "time":
            continue
        result[name + "_peak"] = float(samples[:, column].max())
        result[name + "_median"] = float(np.median(samples[:, column]))
    result["trace"] = [list(map(float, x)) for x in rows]

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in result.items() if k != "trace"})


if __name__ == "__main__":
    main()
