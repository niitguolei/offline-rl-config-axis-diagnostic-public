"""Is the large negative bias a BUG, or genuine OOD extrapolation?

Diagnosis showed 1-step IQL at 500k with critic Q ~= 13 while the MC reference
is ~330. Either:

  (a) the critic is broken / mis-implemented, or
  (b) the critic is accurate on DATASET actions but extrapolates badly to the
      POLICY's actions -- which is exactly the offline-RL phenomenon under
      study, and also why pessimism methods exist.

These are indistinguishable from the runs so far, because Q(s, a_data) was
never measured. This script trains briefly and compares both.

    python scripts/check_critic_scale.py --steps 150000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from orl.algos.iql import IQL, IQLConfig
from orl.algos.mc_reference import mc_reference
from orl.coverage.buckets import buckets_for
from orl.data.d4rl_hdf5 import D4RLDataset

ENV_ID = "halfcheetah_medium_replay-v2"
DOMAIN = "halfcheetah"
GAMMA = 0.99
BATCH = 256
N_STATES = 300


def q_at(agent, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    o = torch.as_tensor(obs, dtype=torch.float32, device=agent.cfg.device)
    a = torch.as_tensor(act, dtype=torch.float32, device=agent.cfg.device)
    with torch.no_grad():
        return agent.q1(agent._sa(o, a)).squeeze(-1).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    a = ap.parse_args()

    ds = D4RLDataset(ENV_ID, mode="ram", normalize=True)
    buckets = buckets_for(ENV_ID)
    agent = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim,
                          gamma=GAMMA, device="cuda"))
    obs, act = ds.get("observations"), ds.get("actions")
    rew, nobs = ds.get("rewards"), ds.get("next_observations")
    term = ds.get("terminals")
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    for i in range(a.steps):
        idx = rng.integers(0, ds.n, BATCH)
        out = agent.update({"observations": obs[idx], "actions": act[idx],
                            "rewards": rew[idx], "next_observations": nobs[idx],
                            "terminals": term[idx]})
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1:>8,}  q_loss={out['q_loss']:.4f}", flush=True)
    print(f"trained {a.steps:,} in {(time.perf_counter() - t0) / 60:.0f} min\n")

    # The env returns RAW observations but the policy was trained on normalised
    # ones. Without this wrapper the MC rollout would feed the wrong scale.
    norm = ds._normalizer()
    inner = agent.policy_fn()

    def policy_fn(obs_t):
        arr = np.asarray(obs_t.detach().cpu().numpy() if hasattr(obs_t, "detach") else obs_t)
        return inner(torch.as_tensor(norm(arr), dtype=torch.float32, device="cuda"))

    # Persist BEFORE analysis. A measurement bug must never cost training time
    # again -- that already happened once with this very script.
    ckpt = pathlib.Path("results/benchmarks") / f"critic_scale_{a.steps}.pt"
    torch.save({"q1": agent.q1.state_dict(),
                "pi": agent.pi.state_dict(),
                "cfg": vars(agent.cfg)}, ckpt)
    print(f"saved checkpoint -> {ckpt}\n")

    bid = np.digitize(buckets.scores, buckets.edges)
    print(f"{'bucket':>8}{'MC':>10}{'Q(s,a_data)':>14}{'Q(s,pi(s))':>14}"
          f"{'bias@data':>12}{'bias@pi':>12}")
    print("-" * 70)
    for k in (0, 1, 2):
        idx = np.flatnonzero(bid == k)
        # h5py requires increasing order for fancy indexing.
        pick = np.sort(rng.choice(idx, size=min(N_STATES, idx.size), replace=False))
        # RAW states for set_state; normalised states for the networks.
        st_raw = ds._f["observations"][pick]
        st = norm(st_raw)
        a_data = ds.get("actions")[pick]
        with torch.no_grad():
            a_pi = agent.pi.act(torch.as_tensor(st, dtype=torch.float32,
                                                device="cuda")).cpu().numpy()
        ref = mc_reference(DOMAIN, st_raw, policy_fn, gamma=GAMMA, horizon=1000)
        qd = q_at(agent, st, a_data)
        qp = q_at(agent, st, a_pi)
        print(f"{k:>8}{ref.values.mean():>10.1f}{qd.mean():>14.1f}{qp.mean():>14.1f}"
              f"{(qd - ref.values).mean():>12.1f}{(qp - ref.values).mean():>12.1f}")

    print()
    print("If bias@data is also huge -> the critic itself is broken.")
    print("If bias@data is small but bias@pi is huge -> genuine OOD extrapolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
