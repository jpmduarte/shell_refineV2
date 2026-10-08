"""Record outcomes for jobs started by a runner that predates outcome records.

The runner writes queue/.outcomes/<job>.json when a job ends, so a later runner can
tell a job that finished from one that died. The runner currently live started before
that existed, so its jobs will finish leaving no record, and whoever restarts next will
file them as failures — which is exactly what happened to thirty-one completed jobs.

Restarting the runner would fix it but would also discard jobs mid-training. This
watches the jobs it already started instead, and writes the record on their behalf.
Verdict comes from the same evidence a human would use: an evaluation summary on disk,
or a log ending in an error.

    python scripts/outcome_watcher.py            # watch until running/ empties
    python scripts/outcome_watcher.py --once     # one pass, no waiting
"""

import argparse
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(os.path.dirname(HERE), "queue")
RUNNING = os.path.join(QUEUE, "running")
LOGS = os.path.join(QUEUE, "logs")
OUTCOMES = os.path.join(QUEUE, ".outcomes")


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                      text=True, stderr=subprocess.DEVNULL)
        return str(pid) in out
    except Exception:
        return False


def run_dirs(job_path: str) -> list[str]:
    text = open(job_path, encoding="utf-8-sig").read()
    dirs = re.findall(r'\$rd\s*=\s*"([^"]+)"', text)
    dirs += re.findall(r'--run-dir\s+"?([^\s"`]+)"?', text)
    return [d for d in dirs if not d.startswith("$")]


def finished_well(job: str) -> bool:
    """A completed job leaves an evaluation summary; a broken one leaves an error."""
    for d in run_dirs(os.path.join(RUNNING, job)):
        if os.path.exists(os.path.join(d, "eval_val", "summary.json")):
            return True
    logs = sorted(f for f in os.listdir(LOGS) if f.endswith(job + ".log"))
    if logs:
        with open(os.path.join(LOGS, logs[-1]), encoding="utf-8", errors="replace") as f:
            tail = f.read()[-600:]
        if "Traceback" in tail or "FAILED" in tail:
            return False
    return False


def sweep() -> int:
    """Write a record for every job whose process is gone. Returns how many remain."""
    os.makedirs(OUTCOMES, exist_ok=True)
    remaining = 0

    for name in sorted(f for f in os.listdir(RUNNING) if f.endswith(".txt")):
        state_path = os.path.join(RUNNING, name + ".state")
        if not os.path.exists(state_path):
            continue
        try:
            with open(state_path) as f:
                pid = json.load(f).get("pid")
        except (json.JSONDecodeError, OSError):
            continue

        if pid and pid_alive(pid):
            remaining += 1
            continue

        if os.path.exists(os.path.join(OUTCOMES, name + ".json")):
            continue

        ok = finished_well(name)
        with open(os.path.join(OUTCOMES, name + ".json"), "w") as f:
            json.dump({"status": "done" if ok else "failed",
                       "exit": 0 if ok else 1,
                       "at": time.time(),
                       "by": "outcome_watcher"}, f)
        print(f"[{time.strftime('%H:%M:%S')}] recorded {name}: "
              f"{'done' if ok else 'failed'}", flush=True)

    return remaining


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--poll", type=int, default=60, help="seconds between passes")
    a = p.parse_args()

    while True:
        remaining = sweep()
        if a.once:
            print(f"{remaining} job(s) still running")
            return
        if remaining == 0:
            print("running/ is empty; nothing left to watch")
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
