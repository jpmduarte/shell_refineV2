"""Run queued jobs one at a time, so experiments cannot contend for the GPU.

A job is a file in queue/pending/ whose contents are shell commands. They run in
filename order, so numeric prefixes set the order: 010_train.txt, 020_eval.txt.

Jobs can be added while the runner is going — it rescans between jobs, so dropping a
file into queue/pending/ schedules it without touching the running process.

A failing job is moved to queue/failed/ and the runner continues. Blocking the queue on
one failure is how a chain of dependent scripts wastes a night; recording it and moving
on is not.

A job's exit status is that of its LAST command, as in any shell. A job whose training
step fails but whose final command happens to succeed is reported ok — so put the step
whose success actually matters last, or end the job with an explicit check.

    python scripts/queue_runner.py            # run until the queue is empty
    python scripts/queue_runner.py --watch    # keep waiting for new jobs
"""

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime

HERE  = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(os.path.dirname(HERE), "queue")
PENDING, RUNNING = os.path.join(QUEUE, "pending"), os.path.join(QUEUE, "running")
DONE, FAILED     = os.path.join(QUEUE, "done"),    os.path.join(QUEUE, "failed")
LOGS             = os.path.join(QUEUE, "logs")


def next_job() -> str | None:
    if not os.path.isdir(PENDING):
        return None
    jobs = sorted(f for f in os.listdir(PENDING) if not f.startswith("."))
    return jobs[0] if jobs else None


def run_job(name: str) -> bool:
    src = os.path.join(PENDING, name)
    active = os.path.join(RUNNING, name)
    shutil.move(src, active)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS, f"{stamp}_{name}.log")
    # utf-8-sig, not utf-8: PowerShell 5.1's Set-Content writes a BOM, and passing a
    # leading U+FEFF to `powershell -Command` makes it read the first command name as
    # "﻿Write-Output" and fail.
    with open(active, encoding="utf-8-sig") as f:
        commands = f.read()

    print(f"[{datetime.now():%H:%M:%S}] running {name}   -> {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {name}\n# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        log.flush()
        proc = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", commands],
                              stdout=log, stderr=subprocess.STDOUT, text=True)

    ok = proc.returncode == 0
    dest = DONE if ok else FAILED
    shutil.move(active, os.path.join(dest, name))
    print(f"[{datetime.now():%H:%M:%S}] {name} {'ok' if ok else 'FAILED'} "
          f"(exit {proc.returncode}, {time.time() - t0:.0f}s)", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", action="store_true",
                   help="stay alive when the queue empties, waiting for new jobs")
    p.add_argument("--poll", type=int, default=30, help="seconds between rescans in --watch")
    a = p.parse_args()

    for d in (PENDING, RUNNING, DONE, FAILED, LOGS):
        os.makedirs(d, exist_ok=True)

    # A job left in running/ means the runner died mid-job. Put it back at the front
    # rather than silently losing it.
    for name in sorted(os.listdir(RUNNING)):
        print(f"requeueing interrupted job: {name}")
        shutil.move(os.path.join(RUNNING, name), os.path.join(PENDING, name))

    ran = failed = 0
    while True:
        name = next_job()
        if name is None:
            if not a.watch:
                break
            time.sleep(a.poll)
            continue
        ok = run_job(name)
        ran += 1
        failed += 0 if ok else 1

    print(f"\nqueue empty: {ran} job(s), {failed} failed")
    if failed:
        print(f"see {FAILED} and {LOGS}")


if __name__ == "__main__":
    main()
