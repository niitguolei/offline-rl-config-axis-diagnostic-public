"""GATE 0 — closed-loop self-consistency of the evaluation pipeline.

Part 1: BC on expert data, judged by the diagnostic ladder (250k steps).
Part 2: state-tracking replay + termination agreement.

Two corrections since the first run (full trail in docs/GATE0报告.md):

1. BC_STEPS raised 100k -> 250k. At 100k, BC was simply undertrained: hopper's
   ratio jumped 0.318 -> 0.883 between 100k and 200k and then plateaued at
   0.86-0.99. That undertrained baseline, not a pipeline defect and not a
   hopper-specific pathology, was the whole problem.

2. Part 2 replaced. The original open-loop replay criterion is invalid in
   principle: D4RL observations are `qpos[1:] + qvel` and exclude `qpos[0]`,
   so the dataset's initial state cannot be reconstructed and an open-loop
   replay diverges immediately. It scored ratio -0.014 on halfcheetah (walking
   backwards) while closed-loop BC on the same data scored 1.038.
   `orl.envs.replay` provides the two valid replacements.

    python experiments/gate0.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from orl.algos.bc import BC, BCConfig
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import evaluate
from orl.envs.replay import state_replay, termination_agreement
from orl.paths import BENCH_DIR
from orl.record.logger import log_run

sys.stdout.reconfigure(line_buffering=True)

BC_STEPS = 250_000
BATCH = 256
EVAL_EPISODES = 20
REPLAY_CHECKS = (
    ("halfcheetah_expert-v2", "halfcheetah"),
    ("walker2d_expert-v2", "walker2d"),
    ("hopper_expert-v2", "hopper"),
    ("walker2d_medium_replay-v2", "walker2d"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bc_ratio(env_id: str, domain: str, seed: int = 0) -> dict:
    """Train BC on expert data; return BC_return / dataset_expert_return."""
    with D4RLDataset(env_id, mode="ram") as ds:
        data_ret = float(np.mean(ds.episodes.returns(ds.get("rewards"))))
        bc = BC(BCConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device=DEVICE))
        rng = np.random.default_rng(seed)
        obs, act = ds.get("observations"), ds.get("actions")
        t0 = time.perf_counter()
        for i in range(BC_STEPS):
            idx = rng.integers(0, ds.n, BATCH)
            bc.update({"observations": obs[idx], "actions": act[idx]})
            if (i + 1) % 50_000 == 0:
                print(f"      {i + 1:,}/{BC_STEPS:,}", flush=True)
        train_s = time.perf_counter() - t0

    res = evaluate(domain, bc.policy_fn(), n_episodes=EVAL_EPISODES, seed=0)
    ratio = res.raw_mean / data_ret if data_ret else float("nan")
    log_run(
        run_id=f"gate0_bc_{domain}_s{seed}", method="bc", dataset=env_id,
        domain=domain, tier="expert", seed=seed, stage="GATE0",
        score_norm=res.score_mean, score_raw=res.raw_mean,
        notes=f"BC ratio={ratio:.4f}; data_ret={data_ret:.1f}; steps={BC_STEPS}",
    )
    return {
        "env_id": env_id, "domain": domain,
        "data_expert_return": data_ret, "bc_return": res.raw_mean,
        "bc_score": res.score_mean, "ratio": ratio,
        "bc_steps": BC_STEPS, "train_s": train_s,
    }


def ladder_verdict(r: float) -> str:
    if r > 0.85:
        return "PASS"
    if r >= 0.60:
        return "CONDITIONAL (provisional threshold)"
    if r >= 0.30:
        return "FAIL (suspect pipeline defect)"
    return "FAIL (deterministic defect: stop and fix before S1)"


def main() -> int:
    print("=" * 72)
    print(f"GATE 0 part 1: BC diagnostic ladder ({BC_STEPS:,} steps, "
          f"{EVAL_EPISODES} eval episodes)")
    print("=" * 72)
    bc_rows = []
    for domain in ("halfcheetah", "walker2d", "hopper"):
        env_id = f"{domain}_expert-v2"
        print(f"  {env_id} ...", flush=True)
        row = bc_ratio(env_id, domain)
        bc_rows.append(row)
        line = f"    data_expert={row['data_expert_return']:.1f}"
        line += f"  BC={row['bc_return']:.1f}  ratio={row['ratio']:.4f}"
        line += f"  [{ladder_verdict(row['ratio'])}]"
        print(line, flush=True)

    ratios = [r["ratio"] for r in bc_rows]
    bc_pass = min(ratios) > 0.60
    print(f"\n  min ratio = {min(ratios):.4f}   verdict: "
          f"{'PASS' if bc_pass else 'FAIL'}")

    print("\n" + "=" * 72)
    print("GATE 0 part 2: state-tracking replay + termination agreement")
    print("=" * 72)
    replay_pass = True
    for env_id, domain in REPLAY_CHECKS:
        sr = state_replay(env_id, domain)
        ta = termination_agreement(env_id, domain)
        sr_ok = sr["relative_err"] < 1e-2
        ta_ok = ta["false_early_term"] == 0 and ta["missed_term"] == 0
        replay_pass &= sr_ok and ta_ok
        print(f"  {env_id}")
        print(f"    reward rel_err  {sr['relative_err']:.2e}  "
              f"({'OK' if sr_ok else 'MISMATCH'})")
        line = f"    termination     env={ta['env_terminated']}"
        line += f" data={ta['data_terminal']} false_early={ta['false_early_term']}"
        line += f" missed={ta['missed_term']}  ({'OK' if ta_ok else 'MISMATCH'})"
        print(line)
    print(f"\n  part 2 verdict: {'PASS' if replay_pass else 'FAIL'}")

    overall = bc_pass and replay_pass
    print("\n" + "=" * 72)
    print(f"GATE 0 OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)

    out = {
        "bc_steps": BC_STEPS, "eval_episodes": EVAL_EPISODES, "device": DEVICE,
        "bc": bc_rows, "bc_min_ratio": float(min(ratios)), "bc_pass": bool(bc_pass),
        "part2_pass": bool(replay_pass), "overall_pass": bool(overall),
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / "gate0.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

