"""How noisy is the BC ratio? Establish mean and spread across training seeds.

Context: the GATE 0 BC ladder uses fixed thresholds (0.85 / 0.60 / 0.30), but
two nominally identical runs disagreed by 0.17 on walker2d (0.9313 vs 0.7659)
and by 0.09 on hopper (0.3162 vs 0.4036). If the run-to-run spread is of the
same order as the thresholds, the ladder cannot serve as a single-run gate.

This measures that spread properly: N independent training seeds per domain,
deterministic (mean-action) evaluation, more eval episodes to cut eval noise.

    python experiments/diag_bc_variance.py --seeds 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import torch

from orl.algos.bc import BC, BCConfig
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import evaluate

sys.stdout.reconfigure(line_buffering=True)

BC_STEPS = 100_000
BATCH = 256
EVAL_EPISODES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def one_run(domain: str, seed: int) -> dict:
    env_id = f"{domain}_expert-v2"
    with D4RLDataset(env_id, mode="ram") as ds:
        data_ret = float(np.mean(ds.episodes.returns(ds.get("rewards"))))
        bc = BC(BCConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device=DEVICE))
        rng = np.random.default_rng(seed)
        obs, act = ds.get("observations"), ds.get("actions")
        for _ in range(BC_STEPS):
            idx = rng.integers(0, ds.n, BATCH)
            bc.update({"observations": obs[idx], "actions": act[idx]})

    res = evaluate(domain, bc.policy_fn(), n_episodes=EVAL_EPISODES, seed=0)
    return {
        "data_return": data_ret,
        "bc_return": res.raw_mean,
        "bc_return_sd": res.raw_std,
        "ratio": res.raw_mean / data_ret,
    }
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--domains", nargs="*", default=["hopper", "walker2d", "halfcheetah"])
    a = ap.parse_args()

    print(f"BC_STEPS={BC_STEPS} eval_episodes={EVAL_EPISODES} device={DEVICE}")
    print(f"seeds={a.seeds} domains={a.domains}")
    print()

    summary = {}
    for domain in a.domains:
        ratios = []
        data_ret = 0.0
        print(f"--- {domain} ---", flush=True)
        for seed in range(a.seeds):
            t0 = time.perf_counter()
            r = one_run(domain, seed)
            ratios.append(r["ratio"])
            data_ret = r["data_return"]
            el = time.perf_counter() - t0
            msg = f"  seed {seed}: data_ret={r['data_return']:.1f}"
            msg += f"  BC={r['bc_return']:.1f}  ratio={r['ratio']:.4f}  ({el:.0f}s)"
            print(msg, flush=True)
        mean = statistics.fmean(ratios)
        sd = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        summary[domain] = (mean, sd, ratios, data_ret)
        tail = f"  => mean {mean:.4f}  std {sd:.4f}  range [{min(ratios):.4f}, {max(ratios):.4f}]"
        print(tail)
        print()

    print("=" * 66)
    hdr = f"{'domain':<12}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"
    print(hdr)
    print("-" * 66)
    for domain, (mean, sd, ratios, _d) in summary.items():
        row = f"{domain:<12}{mean:>10.4f}{sd:>10.4f}{min(ratios):>10.4f}{max(ratios):>10.4f}"
        print(row)
    worst = max(v[1] for v in summary.values())
    print()
    print(f"largest across-seed std = {worst:.4f}")
    print("ladder thresholds: 0.85 / 0.60 / 0.30")
    verdict = "SINGLE-RUN UNRELIABLE" if worst > 0.05 else "usable"
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
