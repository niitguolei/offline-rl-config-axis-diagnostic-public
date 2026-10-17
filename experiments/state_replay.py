"""Corrected GATE 0 part 2: state-tracking replay.

Why the original open-loop replay could not work (measured 2026-10-04):
it collapsed on ALL FOUR datasets, including the two where closed-loop BC was
perfect:

    dataset                     rho     ratio   len_med data/replay
    halfcheetah_expert-v2   -0.0004   -0.0140     1000 / 1000
    walker2d_expert-v2       0.1008    0.0517     1000 /   108
    hopper_expert-v2        -0.0408    0.1055     1000 /   145
    walker2d_medium_replay   0.6914    0.2561      192 / 127

Root cause: D4RL observations are `qpos[1:] + qvel`, which EXCLUDES `qpos[0]`
(the global x coordinate). The dataset's initial state therefore cannot be
restored from the data alone, so open-loop replay had to start from
`env.reset()` -- a different state -- and diverged immediately.
halfcheetah's ratio of -0.014 (walking backwards) is the signature of that.

The fix: replay with STATE TRACKING. Before each step, restore the env to the
logged state and then apply the logged action. Divergence is impossible by
construction, so any mismatch isolates the action -> reward path.

    python experiments/state_replay.py
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.obs_layout import ENV_IDS

sys.stdout.reconfigure(line_buffering=True)


def state_replay(env_id: str, domain: str, max_steps: int = 200_000) -> dict:
    """Apply logged actions at logged states; compare rewards to the record."""
    with D4RLDataset(env_id, mode="stream") as ds:
        obs_all = ds.get("observations")
        act_all = ds.get("actions")
        rew_all = ds.get("rewards")
        starts, ends = ds.episodes.starts(), ds.episodes.ends

        env = gym.make(ENV_IDS[domain])
        u = env.unwrapped
        nq, nv = int(u.model.nq), int(u.model.nv)
        env.reset(seed=0)

        rec_r, rep_r = [], []
        budget = max_steps
        for s, e in zip(starts, ends):
            if budget <= 0:
                break
            for t in range(s, e + 1):
                if budget <= 0:
                    break
                o = obs_all[t]
                qpos = np.concatenate([[0.0], o[: nq - 1]])
                qvel = o[nq - 1 : nq - 1 + nv]
                u.set_state(qpos.astype(np.float64), qvel.astype(np.float64))
                _, r, _, _, _ = env.step(act_all[t].astype(np.float64))
                rec_r.append(float(rew_all[t]))
                rep_r.append(float(r))
                budget -= 1
        env.close()

    rec_r = np.asarray(rec_r)
    rep_r = np.asarray(rep_r)
    err = np.abs(rec_r - rep_r)
    return {
        "env_id": env_id,
        "n_steps_compared": int(rec_r.size),
        "mean_abs_reward_err": float(err.mean()),
        "max_abs_reward_err": float(err.max()),
        "relative_err": float(err.mean() / max(1e-9, np.abs(rec_r).mean())),
        "recorded_reward_sd": float(rec_r.std()),
    }


def main() -> int:
    print("%-28s %9s %11s %11s %11s %9s" % (
        "dataset", "n_steps", "mean|err|", "max|err|", "rel_err", "rec_sd"))
    print("-" * 86)
    ok_all = True
    for env_id, domain in (
        ("halfcheetah_expert-v2", "halfcheetah"),
        ("walker2d_expert-v2", "walker2d"),
        ("hopper_expert-v2", "hopper"),
        ("walker2d_medium_replay-v2", "walker2d"),
    ):
        r = state_replay(env_id, domain)
        ok = r["relative_err"] < 1e-2
        ok_all &= ok
        print("%-28s %9d %11.3e %11.3e %11.2e %9.4f  %s" % (
            r["env_id"], r["n_steps_compared"], r["mean_abs_reward_err"],
            r["max_abs_reward_err"], r["relative_err"],
            r["recorded_reward_sd"], "OK" if ok else "MISMATCH"), flush=True)
    print()
    print(f"state-tracking replay: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

