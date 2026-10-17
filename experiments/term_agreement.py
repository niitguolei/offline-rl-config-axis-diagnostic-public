"""Does gymnasium v5 terminate where the dataset says it should?

The state-tracking replay (experiments/state_replay.py) verified the
action -> reward path on all four datasets. It deliberately ignores the
termination flag, because restoring the state every step means termination
can never compound. So termination-threshold agreement is still untested --
and hopper is precisely where that matters: hopper_expert has 89 terminals
in 1027 episodes (min length 449) versus 0-1 terminals in the other two
domains, and hopper terminates on a HARD healthy-range threshold.

Test: restore the logged state, apply the logged action, and compare the env's
termination flag against the dataset's `terminals[t]`.

    python experiments/term_agreement.py
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.obs_layout import ENV_IDS

sys.stdout.reconfigure(line_buffering=True)


def agreement(env_id: str, domain: str, max_steps: int = 100_000) -> dict:
    """Compare env termination against the dataset's terminal flags."""
    with D4RLDataset(env_id, mode="stream") as ds:
        obs = ds.get("observations")
        act = ds.get("actions")
        term = ds.get("terminals")
        starts, ends = ds.episodes.starts(), ds.episodes.ends

        env = gym.make(ENV_IDS[domain])
        u = env.unwrapped
        nq, nv = int(u.model.nq), int(u.model.nv)
        env.reset(seed=0)

        tp = fp = fn = tn = 0
        budget = max_steps
        for s, e in zip(starts, ends):
            if budget <= 0:
                break
            for t in range(s, e + 1):
                if budget <= 0:
                    break
                o = obs[t]
                qpos = np.concatenate([[0.0], o[: nq - 1]])
                qvel = o[nq - 1 : nq - 1 + nv]
                u.set_state(qpos.astype(np.float64), qvel.astype(np.float64))
                _, _, terminated, truncated, _ = env.step(act[t].astype(np.float64))
                env_term = bool(terminated)
                data_flag = bool(term[t])
                if env_term and data_flag:
                    tp += 1
                elif env_term and not data_flag:
                    fp += 1
                elif data_flag and not env_term:
                    fn += 1
                else:
                    tn += 1
                budget -= 1
        env.close()

    n = tp + fp + fn + tn
    return {
        "env_id": env_id, "n_steps": n,
        "env_terminated": tp + fp, "data_terminal": tp + fn,
        "false_early_term": fp, "missed_term": fn,
    }


def main() -> int:
    print("%-28s %8s %9s %9s %11s %9s" % (
        "dataset", "n_steps", "env_term", "data_term", "false_early", "missed"))
    print("-" * 80)
    for env_id, domain in (
        ("halfcheetah_expert-v2", "halfcheetah"),
        ("walker2d_expert-v2", "walker2d"),
        ("hopper_expert-v2", "hopper"),
        ("walker2d_medium_replay-v2", "walker2d"),
    ):
        r = agreement(env_id, domain)
        print("%-28s %8d %9d %9d %11d %9d" % (
            r["env_id"], r["n_steps"], r["env_terminated"],
            r["data_terminal"], r["false_early_term"],
            r["missed_term"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

