"""Run queued jobs, starting one only when the GPU has room for it.

A job is a file in queue/pending/ holding shell commands, preceded by a header:

    # probe: python scripts/probe_vram.py --shape 8 64 64 64
    # vram: 3000
    # desc: cube 64^3 x8, train -> predict -> evaluate

Jobs start in filename order, so numeric prefixes set priority: 010_..., 020_....
Several run at once whenever one more fits in what is free; otherwise it waits. This
matters because a single training here leaves the GPU idle ~70% of the time — it is
bound by decompressing crops on the CPU, not by compute — so packing several together
costs little and finishes the set sooner.

`# probe` is how a job says what it costs: a command that runs one real forward and
backward at the job's shape and prints `PEAK_MB <n>`. Scheduling on a measurement
beats scheduling on an estimate — the guesses for these four configurations were out
by up to 300 MiB in both directions. `# vram` is the fallback when there is no probe,
or when the probe cannot run yet.

Accounting deliberately does not attribute VRAM per process: nvidia-smi reports that
as N/A on Windows. The runner reads total free memory, keeps a margin clear, and waits
a settling period after each launch so a new allocation is visible before the next
decision.

Jobs can be added while it runs — it rescans between decisions, so dropping a file into
queue/pending/ schedules it without touching the running process.

A failing job moves to queue/failed/ and the others carry on. A job's exit status is
that of its last command, so a pipeline must chain its steps with `; if (-not $?) {
exit 1 }` or the later steps will paper over an earlier failure — the exact way a
killed training still produced an evaluation of a stale checkpoint here.

    python scripts/queue_runner.py
    python scripts/queue_runner.py --watch --max-jobs 3
"""

import argparse
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

HERE  = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(os.path.dirname(HERE), "queue")
PENDING, RUNNING = os.path.join(QUEUE, "pending"), os.path.join(QUEUE, "running")
DONE, FAILED     = os.path.join(QUEUE, "done"),    os.path.join(QUEUE, "failed")
LOGS             = os.path.join(QUEUE, "logs")

DEFAULT_VRAM_MB = 3000
SETTLE_SECONDS  = 75
MARGIN_MB       = 1536
# A probe allocates real memory, so it needs room of its own. Below this, defer the
# probe and fall back to the declared figure rather than risk an OOM in a running job.
PROBE_HEADROOM_MB = 4500


def gpu_total_used() -> tuple[int, int]:
    """(total, used) MiB. (0, 0) if nvidia-smi is unavailable, which disables gating."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        total, used = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
        return total, used
    except Exception:
        return 0, 0


def parse_header(text: str) -> dict:
    head = {"vram": DEFAULT_VRAM_MB, "desc": "", "probe": ""}
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(vram|desc|probe)\s*:\s*(.+?)\s*$", line, re.I)
        if m:
            key, value = m.group(1).lower(), m.group(2)
            head[key] = int(value) if key == "vram" else value
        elif line.strip() and not line.lstrip().startswith("#"):
            break
    return head


_probe_cache: dict[str, int] = {}


def measure(probe: str, declared: int, free: int) -> int:
    """Peak MiB for this configuration, measured once and remembered."""
    if not probe:
        return declared
    if probe in _probe_cache:
        return _probe_cache[probe]
    if free < PROBE_HEADROOM_MB:
        return declared

    try:
        out = subprocess.check_output(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", probe],
            text=True, stderr=subprocess.DEVNULL, timeout=600)
        m = re.search(r"PEAK_MB\s+(\d+)", out)
        if not m:
            return declared
        peak = int(m.group(1))
    except Exception:
        return declared

    _probe_cache[probe] = peak
    print(f"[{datetime.now():%H:%M:%S}] probe  {peak} MiB measured "
          f"(header said {declared})", flush=True)
    return peak


class Job:
    def __init__(self, name: str):
        self.name = name
        self.proc = None
        self.log = None
        self.vram = DEFAULT_VRAM_MB
        self.desc = ""
        self.started = None


def launch(name: str, need: int) -> Job:
    job = Job(name)
    active = os.path.join(RUNNING, name)
    shutil.move(os.path.join(PENDING, name), active)

    # utf-8-sig: PowerShell's Set-Content writes a BOM, and a leading U+FEFF makes
    # `powershell -Command` read the first command name as "﻿Write-Output".
    with open(active, encoding="utf-8-sig") as f:
        commands = f.read()
    job.vram, job.desc = need, parse_header(commands)["desc"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS, f"{stamp}_{name}.log")
    job.log = open(log_path, "w", encoding="utf-8")
    job.log.write(f"# {name}\n# {job.desc}\n# vram {job.vram} MiB\n"
                  f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
    job.log.flush()

    job.proc = subprocess.Popen(
        ["powershell", "-ExecutionPolicy", "Bypass", "-Command", commands],
        stdout=job.log, stderr=subprocess.STDOUT, text=True)
    job.started = time.time()
    print(f"[{datetime.now():%H:%M:%S}] start  {name}  ({job.vram} MiB)"
          f"{'  — ' + job.desc if job.desc else ''}", flush=True)
    print(f"                       log: {log_path}", flush=True)
    return job


def reap(running: list[Job]) -> list[Job]:
    still = []
    for job in running:
        code = job.proc.poll()
        if code is None:
            still.append(job)
            continue
        job.log.close()
        ok = code == 0
        shutil.move(os.path.join(RUNNING, job.name),
                    os.path.join(DONE if ok else FAILED, job.name))
        mins = (time.time() - job.started) / 60
        print(f"[{datetime.now():%H:%M:%S}] {'done ' if ok else 'FAIL '} {job.name}"
              f"  (exit {code}, {mins:.0f} min)", flush=True)
    return still


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", action="store_true",
                   help="stay alive when the queue empties, waiting for new jobs")
    p.add_argument("--max-jobs", type=int, default=0,
                   help="concurrent jobs cap; 0 means only free VRAM limits it")
    p.add_argument("--poll", type=int, default=20, help="seconds between decisions")
    p.add_argument("--margin", type=int, default=MARGIN_MB,
                   help="MiB kept free on top of what jobs declare")
    a = p.parse_args()

    for d in (PENDING, RUNNING, DONE, FAILED, LOGS):
        os.makedirs(d, exist_ok=True)

    # Anything in running/ means a previous runner died mid-job. Requeue rather than
    # silently drop it.
    for name in sorted(os.listdir(RUNNING)):
        print(f"requeueing interrupted job: {name}")
        shutil.move(os.path.join(RUNNING, name), os.path.join(PENDING, name))

    total, _ = gpu_total_used()
    cap = a.max_jobs if a.max_jobs else "unlimited"
    print(f"GPU total {total} MiB   concurrency {cap}   margin {a.margin} MiB\n")

    running: list[Job] = []
    last_launch = 0.0
    last_hold = None
    while True:
        running = reap(running)
        pending = sorted(f for f in os.listdir(PENDING) if not f.startswith("."))

        if not pending and not running:
            if not a.watch:
                break
            time.sleep(a.poll)
            continue

        settling = time.time() - last_launch < SETTLE_SECONDS
        under_cap = (not a.max_jobs) or len(running) < a.max_jobs
        if pending and under_cap and not settling:
            name = pending[0]
            with open(os.path.join(PENDING, name), encoding="utf-8-sig") as f:
                head = parse_header(f.read())

            total, used = gpu_total_used()
            free = (total - used) if total else 10 ** 9
            need = measure(head["probe"], head["vram"], free)
            # Re-read: the probe itself allocated and released while we were measuring.
            total, used = gpu_total_used()
            free = (total - used) if total else 10 ** 9

            if free >= need + a.margin:
                running.append(launch(name, need))
                last_launch = time.time()
            else:
                # Only when the situation changes: a job can wait out a long training,
                # and repeating the same line every poll buries everything else.
                state = (name, len(running))
                if state != last_hold:
                    print(f"[{datetime.now():%H:%M:%S}] hold   {name} needs {need} MiB, "
                          f"{free} MiB free, {a.margin} MiB margin "
                          f"({len(running)} running)", flush=True)
                    last_hold = state

        time.sleep(a.poll)

    print("\nqueue empty")


if __name__ == "__main__":
    main()
