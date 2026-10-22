"""Time label -> SDF target per case: the exact distance transform, truncation, resampling to x4.

It runs once per case when the cache is built (common/sdf_targets.py), never at inference. The
same calls as sdf_targets.build, CPU only; reading the label is outside the timing.
"""
import argparse
import json
import time

import numpy as np

from common.geometry import FACTOR, downsample, voxel_spacing
from common.image_cache import case_ids, load_label
from common.sdf_targets import TRUNC_MM, as_resamplable, normalise, signed_distance_mm
from datasets import add_dataset_argument, get as get_dataset

MS_PER_S = 1e3

USAGE = """\
example:
  python scripts/benchmark_sdf.py --out docs/profiling/bench_sdf.json"""


def time_case(label, affine) -> dict:
    """Seconds per stage."""
    seconds = {}
    start = time.perf_counter()
    sdf_mm = signed_distance_mm(label, voxel_spacing(affine))
    seconds["distance transform (native)"] = time.perf_counter() - start

    start = time.perf_counter()
    sdf = normalise(sdf_mm, TRUNC_MM)
    seconds["truncate + normalise"] = time.perf_counter() - start

    start = time.perf_counter()
    downsample(as_resamplable(sdf, affine), FACTOR, key="sdf")
    seconds["resample to x4"] = time.perf_counter() - start
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_argument(parser)
    parser.add_argument("--cache-dir", default=None, help="default: the dataset's cache")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases")
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or get_dataset(args.dataset).cache

    rows = []
    for i, cid in enumerate(case_ids(args.cache_dir)[:args.limit or None], start=1):
        label, affine = load_label(args.cache_dir, cid)
        label = np.asarray(label)
        seconds = time_case(label, affine)
        rows.append({k: v * MS_PER_S for k, v in seconds.items()})
        print(f"[{i}] {cid[:40]:<40} {label.shape}  " +
              ", ".join(f"{k} {v * MS_PER_S:.0f} ms" for k, v in seconds.items()), flush=True)

    total = np.array([sum(r.values()) for r in rows])
    result = {"cases": len(rows),
              "ms_median": float(np.median(total)), "ms_mean": float(total.mean()),
              "ms_p5": float(np.percentile(total, 5)), "ms_p95": float(np.percentile(total, 95)),
              "all_cases_s": float(total.sum() / MS_PER_S),
              "stages_ms_median": {k: float(np.median([r[k] for r in rows])) for k in rows[0]}}
    print(f"\nlabel -> SDF: {result['ms_median']:.0f} ms/case median ({result['ms_mean']:.0f} mean, "
          f"p5-p95 {result['ms_p5']:.0f}-{result['ms_p95']:.0f}); all {len(rows)} cases {result['all_cases_s']:.0f} s")
    print("   stages (median ms): " + ", ".join(f"{k} {v:.0f}" for k, v in result["stages_ms_median"].items()))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
