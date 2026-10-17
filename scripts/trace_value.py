"""Trace V and Q over training to find when the value function turns over.

State so far (all with layer_norm, normalised obs, float next_observations):
    2k steps   V=26.4  Q=27.5   (healthy, approaching the +309 fixed point)
  150k steps   V= 0.19 Q=-427    (collapsed and negative)

So the failure is long-horizon. This prints the trajectory cheaply.

    python scripts/trace_value.py --steps 60000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from orl.algos.iql import IQL, IQLConfig
from orl.data.d4rl_hdf5 import D4RLDataset

sys.stdout.reconfigure(line_buffering=True)

KEYS = ("observations", "actions", "rewards", "next_observations", "terminals")
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--every", type=int, default=2_000)
    ap.add_argument("--no-layernorm", action="store_true")
    ap.add_argument("--clip", type=float, default=None)
    a = ap.parse_args()

    dev = "cuda"
    ds = D4RLDataset("halfcheetah_medium_replay-v2", mode="ram", normalize=True)
    o, ac, r, no, te = [ds.get(k) for k in KEYS]
    probe = np.sort(np.random.default_rng(7).choice(ds.n, 4096, replace=False))
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    po, pa = T(o[probe]), T(ac[probe])
    ag = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, gamma=0.99,
                       layer_norm=not a.no_layernorm,
                       clip_grad_norm=a.clip, device=dev))
    print(f"layer_norm={not a.no_layernorm}  clip={a.clip}")
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    print(f"{'step':>9}{'q_loss':>13}{'V':>11}{'Q':>11}{'V_std':>10}")
    for i in range(1, a.steps + 1):
        j = rng.integers(0, ds.n, 256)
        batch = dict(zip(KEYS, (o[j], ac[j], r[j], no[j], te[j])))
        out = ag.update(batch)
        if i % a.every == 0 or i == a.steps:
            with torch.no_grad():
                V = ag.v(po).squeeze(-1)
                Q = ag.q1(ag._sa(po, pa)).squeeze(-1)
            print(f"{i:>9,}{out['q_loss']:>13.3f}{V.mean():>11.2f}{Q.mean():>11.2f}{V.std():>10.2f}", flush=True)
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min; fixed point r/(1-gamma) = {r.mean() / 0.01:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
