# Measures the cost of turning a label into the training target, the SDF: the exact
# distance transform on the native label, truncation, and resampling to the x4 grid. This
# runs once per case when the cache is built (common/sdf_targets.py), never at inference, where the
# network outputs the SDF itself.
#
#     python scripts/benchmark_sdf.py --out docs/profiling/bench_sdf.json
#
# Per case, stage by stage, the same calls as make_sdf.build: reading the label from disk is
# outside the timing. CPU only (scipy), one thread for the distance transform, as in the
# cache builder.

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from common.paths import NATIVE_CACHE  # noqa: E402

from monai.data import MetaTensor  # noqa: E402

from common.image_cache import case_ids, load_label  # noqa: E402
from common.geometry import downsample  # noqa: E402
from common.sdf_targets import normalise, signed_distance_mm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Time label -> SDF target, per case.")
    p.add_argument("--cache-dir", default=NATIVE_CACHE)
    p.add_argument("--limit", type=int, default=0, help="only the first N cases")
    p.add_argument("--out", default=None, help="write the results as JSON here")
    a = p.parse_args()

    rows = []
    for i, cid in enumerate(case_ids(a.cache_dir)[:a.limit or None], start=1):
        label, affine = load_label(a.cache_dir, cid)
        label = np.asarray(label[0] if label.ndim == 4 else label)
        spacing = np.abs(np.diag(affine[:3, :3]))
        t = {}

        t0 = time.perf_counter()
        sdf_mm = signed_distance_mm(label, spacing)
        t["distance transform (native)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sdf = normalise(sdf_mm, 10.0)
        t["truncate + normalise"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        item = {"sdf": MetaTensor(torch.from_numpy(sdf)[None].float(), affine=torch.from_numpy(affine))}
        downsample(item, 4, key="sdf")
        t["resample to x4"] = time.perf_counter() - t0

        rows.append({k: v * 1e3 for k, v in t.items()})
        print(f"[{i}] {cid[:40]:<40} {label.shape}  " +
              ", ".join(f"{k} {v * 1e3:.0f} ms" for k, v in t.items()), flush=True)

    total = np.array([sum(r.values()) for r in rows])
    res = {"cases": len(rows),
           "ms_median": float(np.median(total)), "ms_mean": float(total.mean()),
           "ms_p5": float(np.percentile(total, 5)), "ms_p95": float(np.percentile(total, 95)),
           "all_cases_s": float(total.sum() / 1e3),
           "stages_ms_median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]}}
    print(f"\nlabel -> SDF: {res['ms_median']:.0f} ms/case median ({res['ms_mean']:.0f} mean, "
          f"p5-p95 {res['ms_p5']:.0f}-{res['ms_p95']:.0f}); all {len(rows)} cases {res['all_cases_s']:.0f} s")
    print("   stages (median ms): " + ", ".join(f"{k} {v:.0f}" for k, v in res["stages_ms_median"].items()))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
