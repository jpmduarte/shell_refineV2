# Runs any command and records what it costs the machine while it runs.
#
#     python scripts/profile_run.py --out prof.json -- python size_normalization/train.py ...
#
# Sampled every --every seconds:
#     GPU memory   the card's total use minus its use before launch. Windows' driver gives
#                  no per-process figure, so run with the GPU otherwise idle. This is what
#                  the job occupies (PyTorch's reserve and the CUDA context included), not
#                  only what its tensors hold.
#     RAM          resident memory of the process and all its children (data loader
#                  workers included).
#     CPU          the process tree's CPU use, in cores (one fully busy core = 1.0).
#     GPU util     nvidia-smi's utilisation, in %.
#
# Writes the peak and median of each, the wall time, and the full trace to --out.

import argparse
import json
import subprocess
import time

import numpy as np
import psutil


# The card's memory in use (MB) and its utilisation (%), from nvidia-smi.
def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, util = out.strip().splitlines()[0].split(",")
    return float(used), float(util)


# A process and all its descendants (empty once it has exited).
def tree(p):
    try:
        return [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def main():
    p = argparse.ArgumentParser(description="Run a command and sample its GPU, RAM and CPU use.")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="the command, after --")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

    gpu_baseline, _ = gpu()
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)

    cpu_prev = {}   # pid -> its CPU seconds at the previous sample
    rows = []       # (time, GPU MB, RAM MB, CPU cores, GPU %) per sample
    while proc.poll() is None:
        time.sleep(a.every)
        rss, cpu = 0.0, 0.0
        for q in tree(ps):
            try:
                rss += q.memory_info().rss
                # CPU seconds used since the previous sample; a new process starts at 0.
                c = sum(q.cpu_times()[:2])
                cpu += c - cpu_prev.get(q.pid, c)
                cpu_prev[q.pid] = c
            except psutil.NoSuchProcess:
                pass                       # it exited between listing and reading
        used, util = gpu()
        rows.append((time.time() - t0, used - gpu_baseline, rss / 2**20, cpu / a.every, util))

    wall = time.time() - t0
    r = np.array(rows) if rows else np.zeros((1, 5))
    res = {"cmd": " ".join(cmd), "exit": proc.returncode, "wall_s": wall,
           "gpu_baseline_mb": gpu_baseline, "samples": len(rows)}
    for i, k in ((1, "gpu_mb"), (2, "ram_mb"), (3, "cpu_cores"), (4, "gpu_util")):
        res[k + "_peak"] = float(r[:, i].max())
        res[k + "_median"] = float(np.median(r[:, i]))
    res["trace"] = [list(map(float, x)) for x in rows]

    json.dump(res, open(a.out, "w"), indent=1)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "trace"})


if __name__ == "__main__":
    main()
