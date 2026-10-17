"""T10 — precompute and cache coverage scores for every dataset.

One full kNN scan costs ~66 s per dataset at n_ref=100k. Every later training
run reuses the cache, so this is paid once.

    python experiments/build_coverage_cache.py
    python experiments/build_coverage_cache.py --tier medium-replay
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from orl.coverage.knn import DEFAULT_K, DEFAULT_N_REF, cache_path, get_coverage
from orl.paths import COVERAGE_CACHE, all_env_ids

sys.stdout.reconfigure(line_buffering=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--tier", help="only this tier")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    ids = all_env_ids()
    if a.tier:
        ids = [i for i in ids if f"_{a.tier.replace('-', '_')}-v2" in i]
    print(f"{len(ids)} datasets  n_ref={a.n_ref}  k={a.k}  "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    COVERAGE_CACHE.mkdir(parents=True, exist_ok=True)

    for env_id in ids:
        p = cache_path(env_id, a.n_ref, a.k)
        if p.exists() and not a.force:
            print(f"  {env_id:<34} cached")
            continue
        t0 = time.perf_counter()
        s = get_coverage(env_id, n_ref=a.n_ref, k=a.k, force=a.force)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        print(f"  {env_id:<34} {len(s):>9,} scores  {time.perf_counter() - t0:6.1f}s  "
              f"peak_vram={peak:.2f}GB  median={np.median(s):.3f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    files = list(COVERAGE_CACHE.glob("*.npy"))
    print(f"\ncache: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB "
          f"in {COVERAGE_CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

