"""I2 — deterministic evaluation loop over gymnasium v5 MuJoCo envs.

Acceptance criteria (docs/S0工作计划.md, T4):
  * same checkpoint + SAME eval seed  -> bitwise identical (determinism gate)
  * different eval seeds              -> spread is MEASURED, never gated
  * dataset action range must lie inside env action_space
  * dataset obs dim must equal env obs dim
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
import torch

from .obs_layout import ENV_IDS, assert_dataset_matches_env
from .ref_scores import normalized_score


class BatchPolicy(Protocol):
    def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


class ZeroLikePolicy:
    """Deterministic zero-action policy.

    Useful for timing the evaluation loop (T6) without needing a trained
    model, and as a smoke-test policy.
    """

    def __init__(self, act_dim: int) -> None:
        self.act_dim = act_dim

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], self.act_dim), dtype=torch.float32)


@dataclasses.dataclass(frozen=True)
class EvalResult:
    domain: str
    raw_returns: np.ndarray
    raw_mean: float
    raw_std: float
    score_mean: float
    score_std: float
    n_episodes: int
    seed: int


def evaluate(
    domain: str,
    policy: BatchPolicy,
    n_episodes: int = 10,
    seed: int = 0,
    device: str = "cpu",
    max_steps: int = 1000,
) -> EvalResult:
    """Run `n_episodes` deterministic episodes and summarise returns."""
    env = gym.make(ENV_IDS[domain])
    rets = np.zeros(n_episodes, dtype=np.float64)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a = policy(o).squeeze(0).detach().cpu().numpy()
                obs, r, term, trunc, _ = env.step(np.asarray(a, dtype=np.float64))
                total += float(r)
                done = bool(term or trunc)
                if ep == 0 and total != total:  # NaN guard
                    total = -np.inf
                    break
            rets[ep] = total
    finally:
        env.close()

    scores = np.array([normalized_score(domain, r) for r in rets])
    return EvalResult(
        domain=domain,
        raw_returns=rets,
        raw_mean=float(rets.mean()),
        raw_std=float(rets.std()),
        score_mean=float(scores.mean()),
        score_std=float(scores.std()),
        n_episodes=n_episodes,
        seed=seed,
    )


def check_action_range(domain: str, act_min: float, act_max: float) -> None:
    """T4 acceptance: dataset actions must lie inside the env action space.

    Action-scaling bugs are a classic cause of a silently broken pipeline,
    so this is a hard assertion rather than a warning.
    """
    env = gym.make(ENV_IDS[domain])
    try:
        lo = float(env.action_space.low[0])
        hi = float(env.action_space.high[0])
    finally:
        env.close()
    if act_min < lo - 1e-6 or act_max > hi + 1e-6:
        raise AssertionError(
            f"{domain}: dataset actions [{act_min:.4f}, {act_max:.4f}] outside "
            f"env action_space [{lo}, {hi}] -- action scaling is wrong."
        )


def check_dims(dataset_obs_dim: int, dataset_act_dim: int, domain: str) -> None:
    env = gym.make(ENV_IDS[domain])
    try:
        env_obs = int(env.observation_space.shape[0])
        env_act = int(env.action_space.shape[0])
    finally:
        env.close()
    assert_dataset_matches_env(dataset_obs_dim, domain)
    if dataset_act_dim != env_act:
        raise AssertionError(
            f"{domain}: dataset act dim {dataset_act_dim} != env {env_act}"
        )
