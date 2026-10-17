"""GATE 0 post-mortem: why did hopper fail while the other two domains passed?

GATE 0 result (2026-10-04):
    halfcheetah_expert  ratio 1.0380  PASS
    walker2d_expert     ratio 0.9313  PASS
    hopper_expert       ratio 0.3162  FAIL (ladder: "suspect pipeline defect")

Episode structure explains a lot: hopper_expert has 89 terminals and a minimum
episode length of 449, versus 0 and 1 terminals for halfcheetah/walker2d.
hopper terminates on a HARD threshold (healthy z / angle range), so a policy
that is merely *different* falls over early and loses most of the return.

Decisive test: if the pipeline were broken, OPEN-LOOP replay of the logged
actions would also degrade on hopper. If only the closed-loop BC ratio
degrades, the pipeline is fine and the issue is termination sensitivity.
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.obs_layout import ENV_IDS

sys.stdout.reconfigure(line_buffering=True)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def replay_profile(env_id: str, domain: str) -> dict:
    """Replay logged actions; report drift from the recorded data."""
    with D4RLDataset(env_id, mode="stream") as ds:
        starts, ends = ds.episodes.starts(), ds.episodes.ends
        data_ret = ds.episodes.returns(ds.get("rewards"))
        actions = ds.get("actions")

        env = gym.make(ENV_IDS[domain])
        replayed = np.zeros(len(ends))
        lengths = np.zeros(len(ends), dtype=int)
        for ep, (s, e) in enumerate(zip(starts, ends)):
            env.reset(seed=int(ep))
            total, steps = 0.0, 0
            for t in range(s, e + 1):
                _, r, term, trunc, _ = env.step(actions[t])
                total += float(r)
                steps += 1
                if term or trunc:
                    break
            replayed[ep] = total
            lengths[ep] = steps
        env.close()

    data_len = ends - starts + 1
    return {
        "env_id": env_id,
        "spearman": spearman(replayed, data_ret),
        "ratio": float(replayed.mean() / data_ret.mean()),
        "data_len_med": int(np.median(data_len)),
        "replay_len_med": int(np.median(lengths)),
        "eps_terminating_early": int((lengths < data_len).sum()),
        "n_episodes": int(len(ends)),
    }


def main() -> int:
    print("%-28s %8s %8s %15s %9s %7s" % (
        "dataset", "rho", "ratio", "len_med data/rep", "eps_early", "n_eps"))
    print("-" * 82)
    for env_id, domain in (
        ("halfcheetah_expert-v2", "halfcheetah"),
        ("walker2d_expert-v2", "walker2d"),
        ("hopper_expert-v2", "hopper"),
        ("walker2d_medium_replay-v2", "walker2d"),
    ):
        r = replay_profile(env_id, domain)
        lens = "%d/%d" % (r["data_len_med"], r["replay_len_med"])
        print("%-28s %8.4f %8.4f %15s %9d %7d" % (
            r["env_id"], r["spearman"], r["ratio"], lens,
            r["eps_terminating_early"], r["n_episodes"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

