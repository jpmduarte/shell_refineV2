"""Minimal per-span profiler: wall time, peak VRAM, RAM, CPU. No background sampling
thread, no NVML — reads torch's own counters and psutil at span boundaries only.

    prof = Profiler("phase1_train")
    with prof:
        for epoch in range(epochs):
            with prof.span("train_epoch") as s:
                ...
                s["items"] = n_cases
    # prints a summary table; prof.summary() has the same data as a dict
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import torch

try:
    import psutil
except ImportError:
    psutil = None

MB = 1024 ** 2


def _cuda_active() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def model_summary(model: torch.nn.Module) -> dict:
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "params_mb": round(params * 4 / MB, 2)}


class Profiler:

    def __init__(self, stage: str):
        self.stage  = stage
        self._spans = defaultdict(list)
        self._t0    = None
        self._proc  = psutil.Process() if psutil else None
        if self._proc:
            self._proc.cpu_percent()  # prime; first call always returns 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        if _cuda_active():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        self.print_summary()
        return False

    @contextmanager
    def span(self, name: str, **meta):
        """Time a block; yields a dict a caller can add fields to (e.g. s["items"] = n)."""
        if _cuda_active():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield meta
        finally:
            if _cuda_active():
                torch.cuda.synchronize()
            duration = time.perf_counter() - t0
            record = {"duration_s": duration, **meta}

            if _cuda_active():
                record["peak_vram_mb"] = torch.cuda.max_memory_allocated() / MB
            if self._proc:
                record["ram_mb"]  = self._proc.memory_info().rss / MB
                record["cpu_pct"] = self._proc.cpu_percent()

            self._spans[name].append(record)

    def summary(self) -> dict:
        spans = {}
        for name, records in self._spans.items():
            durations = [r["duration_s"] for r in records]
            entry = {
                "count":   len(records),
                "total_s": round(sum(durations), 3),
                "mean_s":  round(sum(durations) / len(durations), 4),
            }
            vram = [r["peak_vram_mb"] for r in records if "peak_vram_mb" in r]
            if vram:
                entry["peak_vram_mb"] = round(max(vram), 1)
            ram = [r["ram_mb"] for r in records if "ram_mb" in r]
            if ram:
                entry["peak_ram_mb"] = round(max(ram), 1)
            cpu = [r["cpu_pct"] for r in records if r.get("cpu_pct")]
            if cpu:
                entry["mean_cpu_pct"] = round(sum(cpu) / len(cpu), 1)
            items = sum(r.get("items", 0) or 0 for r in records)
            if items:
                entry["items_per_s"] = round(items / max(sum(durations), 1e-9), 2)
            spans[name] = entry

        return {
            "stage": self.stage,
            "wall_s": round(time.perf_counter() - self._t0, 2) if self._t0 else 0.0,
            "spans": spans,
        }

    def print_summary(self):
        data = self.summary()
        print(f"\n{'-' * 72}")
        print(f"  profile — {data['stage']}   wall {data['wall_s']:.1f}s")
        if data["spans"]:
            print(f"  {'span':<16} {'n':>5} {'mean':>9} {'total':>9} {'VRAM':>8} {'RAM':>8} {'CPU':>6}")
            for name, s in data["spans"].items():
                vram = f"{s['peak_vram_mb']:.0f}MB" if "peak_vram_mb" in s else "-"
                ram  = f"{s['peak_ram_mb']:.0f}MB"  if "peak_ram_mb"  in s else "-"
                cpu  = f"{s['mean_cpu_pct']:.0f}%"  if "mean_cpu_pct" in s else "-"
                print(f"  {name:<16} {s['count']:>5} {s['mean_s']:>8.3f}s"
                      f" {s['total_s']:>8.1f}s {vram:>8} {ram:>8} {cpu:>6}")
                if "items_per_s" in s:
                    print(f"    {'':<14} throughput {s['items_per_s']:.2f} items/s")
        print(f"{'-' * 72}\n", flush=True)
