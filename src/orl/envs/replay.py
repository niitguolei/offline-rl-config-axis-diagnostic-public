"""Pipeline verification: does the env reproduce the recorded data?

Two checks, both essential because each covers what the other cannot.

`state_replay` — apply the logged action at the logged state and compare the
reward. Restoring the state every step makes divergence impossible, so any
mismatch isolates the action -> reward path.

`termination_agreement` — compare the env's termination flag against the
dataset's `terminals[t]`. `state_replay` deliberately ignores termination
(state resets mean it can never compound), yet termination is a hard threshold
that can silently change behaviour between environment builds.

WHY NOT OPEN-LOOP REPLAY (the original GATE 0 part-2 criterion, now removed):
it fails for a structural reason. D4RL observations are `qpos[1:] + qvel` and
therefore EXCLUDE `qpos[0]`, the global x coordinate. The dataset's initial
state cannot be reconstructed from the data, so an open-loop replay must start
from `env.reset()` -- a different state -- and diverges immediately.
Measured on 2026-10-04 it collapsed on all four datasets, including the two
where closed-loop BC was near-perfect (halfcheetah replay ratio -0.014, i.e.
walking backwards).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..data.d4rl_hdf5 import D4RLDataset
from .obs_layout import ENV_IDS


def _split(obs_row: np.ndarray, nq: int, nv: int) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (qpos, qvel) from an observation.

    `qpos[0]` is the global x coordinate, which is decoupled from the dynamics
    for these tasks, so 0.0 is a valid stand-in.
    """
    qpos = np.concatenate([[0.0], obs_row[: nq - 1]]).astype(np.float64)
    qvel = obs_row[nq - 1 : nq - 1 + nv].astype(np.float64)
    return qpos, qvel


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
        rec, rep = [], []
        budget = max_steps
        for s, e in zip(starts, ends):
            if budget <= 0:
                break
            for t in range(s, e + 1):
                if budget <= 0:
                    break
                qpos, qvel = _split(obs_all[t], nq, nv)
                u.set_state(qpos, qvel)
                _, r, _, _, _ = env.step(act_all[t].astype(np.float64))
                rec.append(float(rew_all[t]))
                rep.append(float(r))
                budget -= 1
        env.close()

    rec_a = np.asarray(rec)
    rep_a = np.asarray(rep)
    err = np.abs(rec_a - rep_a)
    return {
        "env_id": env_id,
        "n_steps_compared": int(rec_a.size),
        "mean_abs_reward_err": float(err.mean()),
        "max_abs_reward_err": float(err.max()),
        "relative_err": float(err.mean() / max(1e-9, np.abs(rec_a).mean())),
    }


def termination_agreement(env_id: str, domain: str, max_steps: int = 100_000) -> dict:
    """Compare the env's termination flag against the dataset's `terminals`."""
    with D4RLDataset(env_id, mode="stream") as ds:
        obs_all = ds.get("observations")
        act_all = ds.get("actions")
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
                qpos, qvel = _split(obs_all[t], nq, nv)
                u.set_state(qpos, qvel)
                _, _, terminated, _, _ = env.step(act_all[t].astype(np.float64))
                if terminated and bool(term[t]):
                    tp += 1
                elif terminated and not bool(term[t]):
                    fp += 1
                elif bool(term[t]):
                    fn += 1
                else:
                    tn += 1
                budget -= 1
        env.close()

    return {
        "env_id": env_id,
        "n_steps": tp + fp + fn + tn,
        "env_terminated": tp + fp,
        "data_terminal": tp + fn,
        "false_early_term": fp,
        "missed_term": fn,
    }

