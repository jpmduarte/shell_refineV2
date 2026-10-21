"""Run queued jobs, starting each only when the GPU and the system RAM have room for it.

A job is a file in queue/pending/ of PowerShell commands, preceded by a header:

    # probe: python scripts/probe_vram.py --shape 8 64 64 64   (prints PEAK_MB <n>: measured cost)
    # vram: 3000                                               (fallback when there is no probe)
    # ram: 7000
    # needs: size_normalization/runs/fold*/preds_val/*.npz >= 90
    # desc: cube 64^3 x8, train -> predict -> evaluate

Jobs start in filename order (numeric prefixes set priority), several at once while they fit;
files dropped into pending/ are picked up while it runs. A job's exit status is its last
command's, so steps must be chained with `; if (-not $?) { exit 1 }`.

Stopping the runner does not stop its jobs: each records its PID in running/, and the next runner
adopts the live ones and files the rest by the outcome they recorded in queue/.outcomes/.
An interrupted job goes to failed/, not back to pending/: look at its log before relaunching.
"""
import argparse
import atexit
import glob
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QUEUE = os.path.join(ROOT, "queue")
PENDING, RUNNING = os.path.join(QUEUE, "pending"), os.path.join(QUEUE, "running")
DONE, FAILED = os.path.join(QUEUE, "done"), os.path.join(QUEUE, "failed")
LOGS = os.path.join(QUEUE, "logs")
LOCK = os.path.join(QUEUE, ".runner.lock")
OUTCOMES = os.path.join(QUEUE, ".outcomes")

VERSION = 2   # bump when the recorded state changes; version 1 wrote no outcome records

DEFAULT_VRAM_MB = 3000
DEFAULT_RAM_MB = 7000      # a refiner training with two loader workers measured about 7 GB
SETTLE_SECONDS = 75        # after a launch, so its allocation is visible before the next decision
MARGIN_MB = 1536
RAM_MARGIN_MB = 4096       # wider: overcommitting RAM swaps the whole machine, not one job
PROBE_HEADROOM_MB = 4500   # a probe allocates real memory; below this, use the declared figure
PROBE_TIMEOUT_S = 600
UNLIMITED_MB = 10 ** 9     # when a memory reading is unavailable, which disables that gate
LOG_TAIL_CHARS = 8000
FAILURE_MARKERS = ("Traceback (most recent call last)", "FAILED", "WRONG COUNT")
HEADER_LINE = re.compile(r"^\s*#\s*(vram|ram|desc|probe|needs)\s*:\s*(.+?)\s*$", re.I)
NEEDS_SPEC = re.compile(r"^(.*?)(?:\s*>=\s*(\d+))?$")


def now() -> str:
    return f"{datetime.now():%H:%M:%S}"


def gpu_total_used() -> tuple[int, int]:
    """(total, used) MiB; (0, 0) if nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        total, used = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
        return total, used
    except Exception:
        return 0, 0


def gpu_free_mb() -> int:
    total, used = gpu_total_used()
    return (total - used) if total else UNLIMITED_MB


def host_free_mb() -> int:
    """Free system RAM in MiB; each loader worker decompresses its own native-resolution crop."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        return int(int(out.strip()) / 1024)
    except Exception:
        return UNLIMITED_MB


@dataclass
class Header:
    vram: int = DEFAULT_VRAM_MB
    ram: int = DEFAULT_RAM_MB
    desc: str = ""
    probe: str = ""
    needs: str = ""


def parse_header(text: str) -> Header:
    """The `# key: value` lines before the first command."""
    header = Header()
    for line in text.splitlines():
        match = HEADER_LINE.match(line)
        if match:
            key, value = match.group(1).lower(), match.group(2)
            setattr(header, key, int(value) if key in ("vram", "ram") else value)
        elif line.strip() and not line.lstrip().startswith("#"):
            break
    return header


def read_job(path: str) -> str:
    """utf-8-sig: Set-Content writes a BOM, which PowerShell would read as part of the first command."""
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


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
            text=True, stderr=subprocess.DEVNULL, timeout=PROBE_TIMEOUT_S)
        match = re.search(r"PEAK_MB\s+(\d+)", out)
        if not match:
            return declared
        peak = int(match.group(1))
    except Exception:
        return declared

    _probe_cache[probe] = peak
    print(f"[{now()}] probe  {peak} MiB measured (header said {declared})", flush=True)
    return peak


def needs_met(spec: str) -> tuple[bool, str]:
    """A glob, optionally `>= n`: that many files must exist, so partial work does not count as ready."""
    if not spec:
        return True, ""
    match = NEEDS_SPEC.match(spec.strip())
    pattern, want = match.group(1).strip(), int(match.group(2) or 1)
    have = len(glob.glob(os.path.join(ROOT, pattern)))
    return have >= want, f"{have}/{want}"


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


def log_says_failed(log_path: str | None) -> bool:
    """A job announces its own failures and a crash leaves a traceback, even when the exit status is lost."""
    if not log_path or not os.path.exists(log_path):
        return False
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-LOG_TAIL_CHARS:]
    except OSError:
        return False
    return any(marker in tail for marker in FAILURE_MARKERS)


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                      text=True, stderr=subprocess.DEVNULL)
        return str(pid) in out
    except Exception:
        return False


class Adopted:
    """A live job from a previous runner: polled by PID (it is not our child), its verdict read from its log."""

    def __init__(self, pid: int, log_path: str | None = None):
        self.pid = pid
        self.log_path = log_path
        self.returncode = None

    def poll(self):
        if self.returncode is None and not pid_alive(self.pid):
            self.returncode = 1 if log_says_failed(self.log_path) else 0
        return self.returncode


def outcome_path(name: str) -> str:
    return os.path.join(OUTCOMES, name + ".json")


def mark_outcome(name: str, status: str, code: int) -> None:
    """Outside running/, so a later runner can tell a finished job from a crashed one."""
    os.makedirs(OUTCOMES, exist_ok=True)
    with open(outcome_path(name), "w") as f:
        json.dump({"status": status, "exit": code, "at": time.time()}, f)


def read_outcome(name: str) -> dict | None:
    try:
        with open(outcome_path(name)) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return None


def acquire_lock(force: bool = False) -> None:
    """Refuse to start beside another live runner: both would launch the same pending files."""
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
        json.dump({"pid": os.getpid(), "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)
    atexit.register(release_lock)


def release_lock() -> None:
    """Only our own lock; a crashed runner's is cleared by the next one after checking its PID."""
    try:
        with open(LOCK) as f:
            if json.load(f).get("pid") == os.getpid():
                os.remove(LOCK)
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        pass


def state_path(name: str) -> str:
    return os.path.join(RUNNING, name + ".state")


def launch(name: str, need: int) -> Job:
    job = Job(name)
    active = os.path.join(RUNNING, name)
    shutil.move(os.path.join(PENDING, name), active)
    commands = read_job(active)
    job.vram, job.desc = need, parse_header(commands).desc

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

    # So a later runner adopts this job instead of launching it a second time.
    with open(state_path(name), "w") as f:
        json.dump({"pid": job.proc.pid, "vram": job.vram, "desc": job.desc,
                   "log": log_path, "started": job.started, "runner": VERSION}, f)

    print(f"[{now()}] start  {name}  ({job.vram} MiB)"
          f"{'  — ' + job.desc if job.desc else ''}", flush=True)
    print(f"                       log: {log_path}", flush=True)
    return job


def file_job(name: str, destination: str) -> None:
    shutil.move(os.path.join(RUNNING, name), os.path.join(destination, name))
    if os.path.exists(state_path(name)):
        os.remove(state_path(name))


def reap(running: list[Job]) -> list[Job]:
    """File the jobs that ended; returns those still running."""
    still = []
    for job in running:
        code = job.proc.poll()
        if code is None:
            still.append(job)
            continue
        if job.log:
            job.log.close()
        ok = code == 0
        file_job(job.name, DONE if ok else FAILED)
        mark_outcome(job.name, "done" if ok else "failed", code)
        mins = (time.time() - job.started) / 60
        status = "done " if ok else "FAIL "
        detail = f"{mins:.0f} min, from log" if job.adopted else f"exit {code}, {mins:.0f} min"
        print(f"[{now()}] {status} {job.name}  ({detail})", flush=True)
    return still


def read_state(name: str) -> dict:
    if not os.path.exists(state_path(name)):
        return {}
    try:
        with open(state_path(name)) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Loud: treating it as stale relaunches the job if it is in fact still running.
        print(f"WARNING: {name}.state is unreadable ({e}); treating as stale. "
              f"Check nothing is still running before it relaunches.")
        return {}


def adopt(name: str, state: dict) -> Job:
    job = Job(name)
    job.proc = Adopted(state["pid"], state.get("log"))
    job.vram = state.get("vram", DEFAULT_VRAM_MB)
    job.desc = state.get("desc", "")
    job.log_path = state.get("log")
    job.started = state.get("started", time.time())
    job.adopted = True
    mins = (time.time() - job.started) / 60
    print(f"adopted running job: {name} (pid {state['pid']}, {job.vram} MiB, {mins:.0f} min in)")
    if state.get("runner", 1) < VERSION:
        print(f"  NOTE: started by runner v{state.get('runner', 1)}, which does "
              f"not write outcome records.\n"
              f"  If this process ends while no runner is watching, its result "
              f"will look like a crash.\n"
              f"  Check its log before trusting the verdict.")
    if job.log_path:
        print(f"                     log: {job.log_path}")
    return job


def recover_previous_jobs() -> list[Job]:
    """Adopt a previous runner's live jobs; file the dead ones by the outcome they recorded."""
    running = []
    for name in sorted(os.listdir(RUNNING)):
        if name.endswith(".state"):
            continue
        state = read_state(name)
        pid = state.get("pid")
        if pid and pid_alive(pid):
            running.append(adopt(name, state))
            continue
        outcome = read_outcome(name)
        if outcome and outcome.get("status") == "done":
            print(f"completed while no runner was watching: {name}")
            file_job(name, DONE)
        else:
            print(f"job interrupted, moved to failed/: {name}")
            file_job(name, FAILED)
    return running


def pending_jobs() -> list[str]:
    return sorted(f for f in os.listdir(PENDING) if not f.startswith(".") and not f.endswith(".state"))


def first_ready(pending: list[str]):
    """(the first job whose needs are met, its header) or None, and the blocked ones before it."""
    blocked = []
    for candidate in pending:
        header = parse_header(read_job(os.path.join(PENDING, candidate)))
        ok, progress = needs_met(header.needs)
        if ok:
            return (candidate, header), blocked
        blocked.append((candidate, header.needs, progress))
    return None, blocked


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true",
                        help="stay alive when the queue empties, waiting for new jobs")
    parser.add_argument("--max-jobs", type=int, default=0,
                        help="concurrent jobs cap; 0 means only free memory limits it (rarely worth setting)")
    parser.add_argument("--poll", type=int, default=20, help="seconds between decisions")
    parser.add_argument("--margin", type=int, default=MARGIN_MB, help="MiB kept free on top of what jobs declare")
    parser.add_argument("--ram-margin", type=int, default=RAM_MARGIN_MB,
                        help="MiB of system RAM kept free; overcommitting it swaps the whole machine")
    parser.add_argument("--force", action="store_true",
                        help="start even if another runner holds the lock; only when you know "
                             "it is dead and the lock is stale")
    return parser.parse_args()


class Scheduler:
    """The launch decisions; holds are printed only when the situation changes, not every poll."""

    def __init__(self, args, running):
        self.args = args
        self.running = running
        self.last_launch = 0.0
        self.last_hold = None

    def report_blocked(self, blocked):
        state = ("waiting", len(blocked), blocked[0][2])
        if state == self.last_hold:
            return
        first, need, progress = blocked[0]
        more = f" (+{len(blocked) - 1} more)" if len(blocked) > 1 else ""
        print(f"[{now()}] wait   {first} needs {need} ({progress}){more}", flush=True)
        self.last_hold = state

    def try_launch(self, name, header: Header):
        need = measure(header.probe, header.vram, gpu_free_mb())
        free = gpu_free_mb()   # again: the probe allocated and released while measuring
        ram_free = host_free_mb()
        gpu_ok = free >= need + self.args.margin
        ram_ok = ram_free >= header.ram + self.args.ram_margin

        if gpu_ok and ram_ok:
            self.running.append(launch(name, need))
            self.last_launch = time.time()
            return
        state = (name, len(self.running), gpu_ok, ram_ok)
        if state != self.last_hold:
            why = "+".join(part for part, ok in (("GPU", gpu_ok), ("RAM", ram_ok)) if not ok)
            print(f"[{now()}] hold   {name} on {why}: "
                  f"needs {need} MiB VRAM ({free} free), "
                  f"{header.ram} MiB RAM ({ram_free} free) "
                  f"({len(self.running)} running)", flush=True)
            self.last_hold = state

    def step(self) -> bool:
        """One decision; False once there is nothing left to do."""
        self.running = reap(self.running)
        pending = pending_jobs()
        if not pending and not self.running:
            return self.args.watch

        settling = time.time() - self.last_launch < SETTLE_SECONDS
        under_cap = (not self.args.max_jobs) or len(self.running) < self.args.max_jobs
        ready, blocked = first_ready(pending)
        if ready is None and blocked:
            self.report_blocked(blocked)
        if ready and under_cap and not settling:
            self.try_launch(*ready)
        return True


def main():
    args = parse_args()
    for d in (PENDING, RUNNING, DONE, FAILED, LOGS, OUTCOMES):
        os.makedirs(d, exist_ok=True)
    acquire_lock(args.force)

    scheduler = Scheduler(args, recover_previous_jobs())
    total, _ = gpu_total_used()
    cap = args.max_jobs if args.max_jobs else "unlimited"
    print(f"GPU total {total} MiB   concurrency {cap}   margin {args.margin} MiB\n")

    while scheduler.step():
        time.sleep(args.poll)

    print("\nqueue empty")


if __name__ == "__main__":
    main()
