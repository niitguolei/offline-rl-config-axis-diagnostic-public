"""T11 — end-to-end smoke test.

Exercises every component in sequence so a broken pipe shows up as one clear
failure rather than as a confusing result three tasks later:
data -> episode index -> coverage -> BC/IQL updates -> evaluation -> record.

    python experiments/smoke.py
"""
from __future__ import annotations

import sys

import numpy as np

from orl.algos.bc import BC, BCConfig
from orl.algos.iql import IQL, IQLConfig
from orl.coverage.knn import coverage_scores
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import check_action_range, check_dims, evaluate
from orl.envs.obs_layout import ENV_IDS
from orl.paths import parse_env_id
from orl.record.logger import log_run

sys.stdout.reconfigure(line_buffering=True)

ENV_ID = "hopper_medium_replay-v2"
STEPS = 200


def main() -> int:
    domain, tier = parse_env_id(ENV_ID)
    print(f"smoke on {ENV_ID} ({domain} / {tier})")

    with D4RLDataset(ENV_ID, mode="ram") as ds:
        print(f"  [1] data       n={ds.n:,} obs={ds.obs_dim} act={ds.act_dim} "
              f"episodes={ds.episodes.n_episodes:,}")

        check_dims(ds.obs_dim, ds.act_dim, domain)
        a = ds.get("actions")
        check_action_range(domain, float(a.min()), float(a.max()))
        print(f"  [2] asserts    dims + action range OK against {ENV_IDS[domain]}")

        cov = coverage_scores(ds.get("observations"), n_ref=20_000, k=10)
        print(f"  [3] coverage   n={len(cov):,} median={np.median(cov):.3f} "
              f"p99={np.percentile(cov, 99):.3f}")

        rng = np.random.default_rng(0)
        bc = BC(BCConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device="cuda"))
        iql = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device="cuda"))
        term = ds.get("terminals")
        for _ in range(STEPS):
            batch = ds.sample(256, rng)
            batch["terminals"] = term[rng.integers(0, ds.n, 256)]
            bc.update(batch)
            iql.update(batch)
        print(f"  [4] training   {STEPS} BC + {STEPS} IQL updates")

    res = evaluate(domain, bc.policy_fn(), n_episodes=2, seed=0)
    print(f"  [5] evaluate   BC raw={res.raw_mean:.1f} score={res.score_mean:.1f} "
          "(200 updates only: a number to prove the pipe works, NOT a result)")

    p = log_run(
        run_id="smoke_s0", method="bc", dataset=ENV_ID, domain=domain, tier=tier,
        seed=0, stage="S0", score_norm=res.score_mean, score_raw=res.raw_mean,
        notes="S0 smoke test",
    )
    print(f"  [6] record     {p.name}")
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

