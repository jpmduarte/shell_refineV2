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

Stopping the runner does not stop what it launched: those jobs keep going, and keep
holding the GPU. So each one records its PID beside it in running/, and a runner
starting up checks them — adopting the ones still alive, and moving the rest to
failed/. Without that, a restart launched live jobs a second time and the duplicates
fought over the same memory. Adopted jobs are polled rather than waited on, since they
are not this process's children, and their memory still counts because the budget is
read from nvidia-smi rather than tallied per job.

An interrupted job goes to failed/ rather than back to pending/. Whatever killed it —
an OOM, a missing file, a kill — will usually happen again, and relaunching silently
hides both the interruption and its cause. Putting it back is a decision to make after
looking at the log.

Both memories gate a launch. VRAM is the obvious one; system RAM is the one that bit —
four slab folds fit easily in 12 GB of VRAM and still took 31 of 32 GB of RAM, because
every dataloader worker decompresses its own native-resolution crop. A job declares
both with # vram: and # ram:, and RAM keeps a wider margin: overcommitting the GPU
fails one job, overcommitting RAM puts the whole machine into swap.

--max-jobs is rarely the right tool. The two memory checks already decide, measured
rather than guessed, and capping on top of them leaves the card idle: five phase 1
folds together use 4.3 GB of 12.2 GB.

    python scripts/queue_runner.py
    python scripts/queue_runner.py --watch
"""

import argparse
import atexit
import json
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
LOCK             = os.path.join(QUEUE, ".runner.lock")

DEFAULT_VRAM_MB = 3000
# A phase 2 training with two dataloader workers measured about 7 GB of RAM: the main
# process plus a worker each holding a decompressed native-resolution crop.
DEFAULT_RAM_MB  = 7000
SETTLE_SECONDS  = 75
MARGIN_MB       = 1536
# Leave more system RAM clear than VRAM: overcommitting the GPU fails a job, while
# overcommitting RAM takes the whole machine down to swap.
RAM_MARGIN_MB   = 4096
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


def host_free_mb() -> int:
    """Free system RAM in MiB, or a large number if it cannot be read.

    Gating on VRAM alone is not enough: four slab folds fit easily in 12 GB of VRAM and
    still took 31 of 32 GB of RAM between them, because each dataloader worker
    decompresses its own copy of a native-resolution crop — seven arrays of about 45 MB
    per case, times three processes per fold.
    """
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        return int(int(out.strip()) / 1024)
    except Exception:
        return 10 ** 9


def parse_header(text: str) -> dict:
    head = {"vram": DEFAULT_VRAM_MB, "ram": DEFAULT_RAM_MB, "desc": "", "probe": ""}
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(vram|ram|desc|probe)\s*:\s*(.+?)\s*$", line, re.I)
        if m:
            key, value = m.group(1).lower(), m.group(2)
            head[key] = int(value) if key in ("vram", "ram") else value
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
        self.log_path = None
        self.vram = DEFAULT_VRAM_MB
        self.desc = ""
        self.started = None
        self.adopted = False


class Adopted:
    """A job from a previous runner that is still alive.

    Polled by PID rather than waited on: it is not this process's child, so there is no
    exit status to collect. That also means its VRAM has to keep counting against the
    budget, or this runner would launch on top of memory that is already spoken for.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if self.returncode is None and not pid_alive(self.pid):
            self.returncode = 0
        return self.returncode


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            text=True, stderr=subprocess.DEVNULL)
        return str(pid) in out
    except Exception:
        return False


def acquire_lock(force: bool = False) -> None:
    """Refuse to start if another runner is already here.

    Two runners do not merely duplicate effort: each sees the other's jobs in running/
    but has no idea the other exists, so both scan pending/ and both launch the same
    file. That happened - two trainings of the same fold, competing for the GPU, which
    then filled and left the remaining folds held.

    Adoption by PID does not cover this. It reconciles a runner with jobs from a
    runner that is gone; here both are alive.
    """
    if os.path.exists(LOCK):
        try:
            with open(LOCK) as f:
                other = json.load(f)
        except (json.JSONDecodeError, OSError):
            other = {}
        pid = other.get("pid")
        if pid and pid_alive(pid) and pid != os.getpid():
            if not force:
                since = other.get("started", "")
                raise SystemExit(
                    "another runner is already running (pid "
                    f"{pid}, started {since}). Queue files are shared, so a "
                    "second one would launch the same jobs again. Stop it "
                    "first, or pass --force if you are certain it is gone "
                    "and the lock is stale.")
            print(f"WARNING: --force, ignoring the lock held by pid {pid}")
        else:
            print(f"clearing stale lock from pid {pid}")

    with open(LOCK, "w") as f:
        json.dump({"pid": os.getpid(),
                   "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)
    atexit.register(release_lock)


def release_lock() -> None:
    """Only ours: a runner that crashed leaves its lock, and the next one clears it
    after checking the PID rather than assuming."""
    try:
        with open(LOCK) as f:
            if json.load(f).get("pid") == os.getpid():
                os.remove(LOCK)
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        pass


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
    job.log_path = log_path

    # Written so a later runner can tell a job that is still going from one whose
    # runner died. Without it, restarting requeues everything in running/ blindly:
    # live jobs get launched a second time while still holding the GPU, and the
    # duplicates fight over it.
    with open(os.path.join(RUNNING, name + ".state"), "w") as f:
        json.dump({"pid": job.proc.pid, "vram": job.vram, "desc": job.desc,
                   "log": log_path, "started": job.started}, f)

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
        if job.log:
            job.log.close()
        ok = code == 0
        shutil.move(os.path.join(RUNNING, job.name),
                    os.path.join(DONE if ok else FAILED, job.name))
        state_path = os.path.join(RUNNING, job.name + ".state")
        if os.path.exists(state_path):
            os.remove(state_path)
        mins = (time.time() - job.started) / 60
        # An adopted job's exit status is not visible: it is not this process's child,
        # so only its disappearance can be observed, not how it ended.
        status = "ended" if job.adopted else ("done " if ok else "FAIL ")
        detail = f"{mins:.0f} min" if job.adopted else f"exit {code}, {mins:.0f} min"
        print(f"[{datetime.now():%H:%M:%S}] {status} {job.name}  ({detail})", flush=True)
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
    p.add_argument("--ram-margin", type=int, default=RAM_MARGIN_MB,
                   help="MiB of system RAM kept free; overcommitting it swaps the whole machine")
    p.add_argument("--force", action="store_true",
                   help="start even if another runner holds the lock; only when you know "
                        "it is dead and the lock is stale")
    a = p.parse_args()

    for d in (PENDING, RUNNING, DONE, FAILED, LOGS):
        os.makedirs(d, exist_ok=True)
    acquire_lock(a.force)

    # Anything in running/ belonged to a previous runner. Some of those jobs may still
    # be going: stopping a runner does not stop what it launched. Requeueing blindly
    # would start them a second time while the originals still hold the GPU, so each is
    # checked by the PID it recorded and either adopted or requeued.
    running: list[Job] = []
    for name in sorted(os.listdir(RUNNING)):
        if name.endswith(".state"):
            continue
        state_path = os.path.join(RUNNING, name + ".state")
        state = {}
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # Loud, because the consequence is silently relaunching a job that is
                # still running: the fallback below looks identical to "no state file".
                print(f"WARNING: {name}.state is unreadable ({e}); treating as stale. "
                      f"Check nothing is still running before it relaunches.")

        pid = state.get("pid")
        if pid and pid_alive(pid):
            job = Job(name)
            job.proc = Adopted(pid)
            job.vram = state.get("vram", DEFAULT_VRAM_MB)
            job.desc = state.get("desc", "")
            job.log_path = state.get("log")
            job.started = state.get("started", time.time())
            job.adopted = True
            running.append(job)
            mins = (time.time() - job.started) / 60
            print(f"adopted running job: {name} (pid {pid}, {job.vram} MiB, "
                  f"{mins:.0f} min in)")
            if job.log_path:
                print(f"                     log: {job.log_path}")
        else:
            # To failed/, not back to pending/. A job whose process is gone was either
            # killed or died, and relaunching it silently hides both: the fact that it
            # was interrupted, and whatever killed it — an OOM or a missing file will
            # simply happen again. Requeueing is a decision for whoever looks at it.
            print(f"job interrupted, moved to failed/: {name}")
            shutil.move(os.path.join(RUNNING, name), os.path.join(FAILED, name))
            if os.path.exists(state_path):
                os.remove(state_path)

    total, _ = gpu_total_used()
    cap = a.max_jobs if a.max_jobs else "unlimited"
    print(f"GPU total {total} MiB   concurrency {cap}   margin {a.margin} MiB\n")

    last_launch = 0.0
    last_hold = None
    while True:
        running = reap(running)
        pending = sorted(f for f in os.listdir(PENDING)
                         if not f.startswith(".") and not f.endswith(".state"))

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
            ram_free = host_free_mb()
            ram_need = head["ram"]

            gpu_ok = free >= need + a.margin
            ram_ok = ram_free >= ram_need + a.ram_margin

            if gpu_ok and ram_ok:
                running.append(launch(name, need))
                last_launch = time.time()
            else:
                # Only when the situation changes: a job can wait out a long training,
                # and repeating the same line every poll buries everything else.
                state = (name, len(running), gpu_ok, ram_ok)
                if state != last_hold:
                    why = ("GPU" if not gpu_ok else "") + ("+" if not gpu_ok and not ram_ok else "") \
                          + ("RAM" if not ram_ok else "")
                    print(f"[{datetime.now():%H:%M:%S}] hold   {name} on {why}: "
                          f"needs {need} MiB VRAM ({free} free), "
                          f"{ram_need} MiB RAM ({ram_free} free) "
                          f"({len(running)} running)", flush=True)
                    last_hold = state

        time.sleep(a.poll)

    print("\nqueue empty")


if __name__ == "__main__":
    main()
