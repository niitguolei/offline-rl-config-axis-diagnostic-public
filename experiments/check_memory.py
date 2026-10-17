"""T2 acceptance check: peak memory when the largest dataset is loaded.

The largest D4RL locomotion file is halfcheetah_medium_expert-v2 (2M
transitions). The S0 plan sets a budget of 1.5 GB for the delta.

    python experiments/check_memory.py
"""
from __future__ import annotations

import gc
import os
import sys

import psutil

from orl.data.d4rl_hdf5 import D4RLDataset

BUDGET_GB = 1.5
LARGEST = "halfcheetah_medium_expert-v2"


def main() -> int:
    proc = psutil.Process(os.getpid())
    rss = lambda: proc.memory_info().rss / 1024**3  # noqa: E731

    base = rss()
    print(f"baseline RSS        {base:.3f} GB")

    with D4RLDataset(LARGEST, mode="ram") as ds:
        for k in ("observations", "next_observations", "actions",
                  "rewards", "terminals", "timeouts"):
            ds.get(k)
        gc.collect()
        peak = rss()
        delta = peak - base
        nbytes = ds.n * (ds.obs_dim * 2 + ds.act_dim) * 4 + ds.n * 3

        print(f"after full RAM load {peak:.3f} GB   (delta {delta:.3f} GB)")
        print(f"transitions {ds.n:,}  obs {ds.obs_dim}  act {ds.act_dim}  "
              f"episodes {ds.episodes.n_episodes:,}")
        print(f"theoretical arrays {nbytes / 1e6:.0f} MB")
        print()
        ok = delta < BUDGET_GB
        print(f"T2 criterion (delta < {BUDGET_GB} GB): {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

