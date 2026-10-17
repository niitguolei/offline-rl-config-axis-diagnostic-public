"""Does BC need more training? One run, evaluated at checkpoints.

docs/GATE0报告.md section 8 concluded the BC ratio sits below the 0.85 ladder
threshold on every domain with large across-seed spread, and that the most
parsimonious explanation is undertraining rather than a pipeline defect.

This tests that directly and cheaply: train ONE policy for 500k steps and
evaluate the ratio at checkpoints. If the ratio climbs with training, BC was
undertrained and the original ladder can stand unchanged.

    python experiments/diag_bc_traincurve.py --domain hopper --steps 500000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from orl.algos.bc import BC, BCConfig
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import evaluate

sys.stdout.reconfigure(line_buffering=True)

BATCH = 256
EVAL_EPISODES = 20
DEVICE = "cuda"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="hopper")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--checkpoints", type=int, nargs="*",
                    default=[100_000, 200_000, 300_000, 400_000, 500_000])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env_id = f"{a.domain}_expert-v2"
    print(f"{env_id}  steps={a.steps}  checkpoints={a.checkpoints}  seed={a.seed}")
    print(f"eval_episodes={EVAL_EPISODES} device={DEVICE}\n")

    with D4RLDataset(env_id, mode="ram") as ds:
        data_ret = float(np.mean(ds.episodes.returns(ds.get("rewards"))))
        bc = BC(BCConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device=DEVICE))
        rng = np.random.default_rng(a.seed)
        obs, act = ds.get("observations"), ds.get("actions")
        n = ds.n

        print(f"data_expert_return = {data_ret:.1f}")
        print(f"{'step':>9} {'raw_return':>11} {'ratio':>8} {'elapsed_s':>10}")
        print("-" * 42)

        todo = sorted(c for c in a.checkpoints if c <= a.steps)
        t_start = time.perf_counter()
        done = 0
        for target in todo:
            while done < target:
                idx = rng.integers(0, n, BATCH)
                bc.update({"observations": obs[idx], "actions": act[idx]})
                done += 1
            res = evaluate(a.domain, bc.policy_fn(), n_episodes=EVAL_EPISODES, seed=0)
            el = time.perf_counter() - t_start
            ratio = res.raw_mean / data_ret
            print(f"{target:>9} {res.raw_mean:>11.1f} {ratio:>8.4f} {el:>10.0f}",
                  flush=True)

    print("\n(read the trend, not the last number: across-seed std was 0.09-0.25)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
