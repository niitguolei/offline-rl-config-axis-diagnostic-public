"""Divergence diagnostic: is the blow-up caused by n-step, and does clipping fix it?

Observed in the T3 calibration (n=5): q_loss 60 -> 480 -> 596,813 between
250k, 500k and 1M steps, while the MC reference stayed flat (324 -> 328).
So the critic diverged, and the per-bucket structure was destroyed (all three
buckets ~46,000 overestimation).

Two variants, run CONCURRENTLY (this diagnostic measures divergence, not
timing, so overlapping processes are harmless and halve the wall clock):

  onestep      plain 1-step IQL, no clipping
               -> if this ALSO diverges, n-step is not the culprit
  nstep_clip   n=5 IQL with gradient clipping at norm 10
               -> if this stays stable, clipping is the fix

    python experiments/diag_divergence.py --variant onestep --steps 1000000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from orl.algos.iql import IQL, IQLConfig
from orl.algos.mc_reference import critic_values, mc_reference, overestimation
from orl.algos.nstep import EpisodeIndexLookup, NStepIQL
from orl.coverage.buckets import buckets_for
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.paths import BENCH_DIR

ENV_ID = "halfcheetah_medium_replay-v2"
DOMAIN = "halfcheetah"
GAMMA = 0.99
BATCH = 256
CLIP_NORM = 10.0
MC_STEPS = (250_000, 500_000, 1_000_000)
STATES_PER_BUCKET = 300
MC_HORIZON = 1000
EVAL_EPISODES = 10


def measure(agent, ds, buckets) -> dict:
    from orl.envs.gym_eval import evaluate

    res = evaluate(DOMAIN, agent.policy_fn(), n_episodes=EVAL_EPISODES, seed=0)
    out = {"policy_raw": res.raw_mean, "buckets": {}}
    rng = np.random.default_rng(12345)
    bid = np.digitize(buckets.scores, buckets.edges)
    for k in (0, 1, 2):
        idx = np.flatnonzero(bid == k)
        pick = rng.choice(idx, size=min(STATES_PER_BUCKET, idx.size), replace=False)
        st = ds.get("observations")[pick]
        ref = mc_reference(DOMAIN, st, agent.policy_fn(), gamma=GAMMA, horizon=MC_HORIZON)
        q = critic_values(agent, st)
        out["buckets"][str(k)] = {
            "mc_mean": float(ref.values.mean()),
            "overest_mean": float(overestimation(q, ref.values).mean()),
            "signed_bias_mean": float((q - ref.values).mean()),
        }
    return out


def grad_norm(module) -> float:
    tot = 0.0
    for p in module.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return float(np.sqrt(tot))
def q_max_abs(agent, ds, dev) -> float:
    o = torch.as_tensor(ds.get("observations")[:2048], dtype=torch.float32, device=dev)
    a = torch.as_tensor(ds.get("actions")[:2048], dtype=torch.float32, device=dev)
    with torch.no_grad():
        return float(agent.q1(agent._sa(o, a)).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["onestep", "nstep_clip"])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--log-every", type=int, default=100_000)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"variant={a.variant}  steps={a.steps:,}  device={dev}", flush=True)

    ds = D4RLDataset(ENV_ID, mode="ram")
    buckets = buckets_for(ENV_ID)
    ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)

    if a.variant == "onestep":
        agent = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim,
                              gamma=GAMMA, device=dev))
        keys = ("observations", "actions", "rewards",
                "next_observations", "terminals")
    else:
        agent = NStepIQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim,
                                   gamma=GAMMA, clip_grad_norm=CLIP_NORM,
                                   device=dev), ep, n_step=5)
        keys = ("observations", "actions", "rewards", "next_observations",
                "terminals", "timeouts")

    arrs = [ds.get(k) for k in keys]
    rng = np.random.default_rng(0)
    records, t0 = [], time.perf_counter()

    for step in range(1, a.steps + 1):
        idx = rng.integers(0, ds.n, BATCH)
        batch = {k: arr[idx] for k, arr in zip(keys, arrs)}
        batch["indices"] = idx
        out = agent.update(batch)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "q_loss": float(out["q_loss"]),
                   "v_loss": float(out["v_loss"]),
                   "pi_loss": float(out["pi_loss"]),
                   "q_max_abs": q_max_abs(agent, ds, dev),
                   "elapsed_min": (time.perf_counter() - t0) / 60.0}
            if step in MC_STEPS:
                rec.update(measure(agent, ds, buckets))
            records.append(rec)
            line = (f"  {step:>9,} q_loss={rec['q_loss']:>12.3f} "
                    f"|Q|max={rec['q_max_abs']:>10.1f} "
                    f"({rec['elapsed_min']:.0f}m)")
            if "buckets" in rec:
                line += "  bias=" + "/".join(
                    f"{rec['buckets'][b]['signed_bias_mean']:+.1f}"
                    for b in ("0", "1", "2"))
            print(line, flush=True)

    out = {"variant": a.variant, "steps": a.steps,
           "clip_grad_norm": CLIP_NORM if a.variant == "nstep_clip" else None,
           "records": records}
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / f"diag_divergence_{a.variant}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
