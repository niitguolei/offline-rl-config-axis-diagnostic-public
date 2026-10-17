"""Worker for bench_concurrency.py: do a fixed number of IQL steps, timed."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from orl.algos.iql import IQL, IQLConfig
from orl.data.d4rl_hdf5 import D4RLDataset

KEYS = ("observations", "actions", "rewards", "next_observations", "terminals")


def main() -> int:
    steps = int(sys.argv[1])
    tag = sys.argv[2]
    ds = D4RLDataset("halfcheetah_medium-v2", mode="ram")
    for k in KEYS:
        ds.get(k)
    agent = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device="cuda"))
    arrs = [ds.get(k) for k in KEYS]
    rng = np.random.default_rng(0)

    def one():
        i = rng.integers(0, ds.n, 256)
        batch = {k: a[i] for k, a in zip(KEYS, arrs)}
        agent.update(batch)

    for _ in range(50):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    print(f"WORKER_DONE {tag} {time.perf_counter() - t0:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
