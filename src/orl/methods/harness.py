"""Shared training/evaluation harness for the mechanism modules (M1, M2).

Both mechanisms sit on the vendored reference baseline, so both need the same
three things, and getting them wrong is exactly what invalidated the previous
round of experiments:

  1. The reference's REWARD RESCALING for locomotion domains.
     dataset['rewards'] /= (max_ret - min_ret); *= max_episode_steps
     On halfcheetah_medium_replay-v2 this is x0.1778, so the reachable fixed
     point is 55.0. Skipping it is what made our own IQL chase an unreachable
     309.3.
  2. Observation normalisation (per-dim std spans 53x raw).
  3. Gymnasium evaluation with the same normaliser the policy was trained on.

Centralising this means a mechanism cannot silently forget step 1.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np

from ..data.d4rl_hdf5 import D4RLDataset
from ..data.episode import EpisodeIndexLookup
from ..data.normalize import ObsNormalizer

MAX_EPISODE_STEPS = 1000


def _ensure_vendor_on_path(repo_root: pathlib.Path) -> None:
    for p in (repo_root / "vendor" / "IQL-PyTorch", repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


@dataclass
class Loaded:
    env_id: str
    domain: str
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    normer: ObsNormalizer
    raw_rew_mean: float
    scaled_rew_mean: float
    coverage: np.ndarray | None
    ret_range: tuple[float, float]
    normer_symmetrised: bool = False
    raw_obs_mean: np.ndarray | None = None
    raw_obs_std: np.ndarray | None = None
    reward_scale: float = 1.0
    augmented: bool = False
    n_original: int | None = None


def load(env_id: str, with_coverage: bool = True, normalize: bool = True,
         mirror_spec=None, augment: bool = False) -> Loaded:
    """Load one D4RL dataset the way the reference expects.

    `mirror_spec` (a MirrorSpec) switches observation normalisation to
    orbit-averaged statistics, which is a PRECONDITION for M1: tying weights
    enforces equivariance of the mirror of the space the network reads, so
    that space has to carry the environment's mirror faithfully. Left as
    None, the reference's plain per-dim standardisation is used unchanged.

    `augment` (requires `mirror_spec`) appends the mirrored copy of every
    transition as a second block, i.e. M1's data-augmentation arm. The
    duplicated arrays have their own episode structure; build the matching
    lookup with `episode_lookup(env_id, n_original=...)` or the n-step windows
    will cross between trajectories.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    _ensure_vendor_on_path(repo_root)

    ds = D4RLDataset(env_id, mode="ram")
    obs = ds.get("observations")
    act = ds.get("actions")
    rew_raw = ds.get("rewards").astype(np.float32)
    next_obs = ds.get("next_observations")
    term = ds.get("terminals").astype(np.float32)
    tout = ds.get("timeouts").astype(np.float32)

    rets = ds.episodes.returns(rew_raw)
    min_ret, max_ret = float(rets.min()), float(rets.max())

    # --- the step that was missing before (see module docstring)
    rew = (rew_raw / (max_ret - min_ret)) * MAX_EPISODE_STEPS

    # Raw statistics, captured BEFORE normalisation. Reporting the statistics
    # of the normalised array instead silently yields mean~0, std~1 and makes
    # every asymmetry diagnostic read as zero -- which is what happened once.
    raw_mean = obs.mean(0) if normalize else None
    raw_std = obs.std(0) + 1e-3 if normalize else None

    normer = ObsNormalizer(mean=np.zeros(obs.shape[1]), std=np.ones(obs.shape[1]))
    if normalize:
        normer = ObsNormalizer.fit(obs)
        if mirror_spec is not None:
            from .equivariance import symmetrise_normer_stats

            mu, sd = symmetrise_normer_stats(normer.mean, normer.std, mirror_spec)
            normer = ObsNormalizer(mean=mu, std=sd, eps=normer.eps)
        obs = normer(obs)
        next_obs = normer(next_obs)

    n_original = int(obs.shape[0])
    if augment:
        if mirror_spec is None:
            raise ValueError(
                "augment=True needs mirror_spec: the mirrored copy is only a "
                "transition of the same MDP if the mirror is the environment's")
        if with_coverage:
            raise ValueError(
                "augment=True with coverage is not supported: the coverage "
                "score of g(s) is a different quantity from that of s and "
                "would need its own kNN pass. M1's augmentation arm runs a "
                "fixed backup length.")
        from .augment import (
            assert_normaliser_carries_mirror,
            mirror_augment,
            require_symmetry,
        )

        # Refuse to mirror with a map that is not the environment's symmetry:
        # that would append transitions the MDP never produces.
        require_symmetry(env_id.split("_")[0], mirror_spec)
        assert_normaliser_carries_mirror(normer.mean, normer.std, mirror_spec)
        aug = mirror_augment(obs, act, rew, next_obs, term, tout, mirror_spec)
        obs, act, rew = aug.obs, aug.act, aug.rew
        next_obs, term, tout = aug.next_obs, aug.terminals, aug.timeouts

    cov = None
    if with_coverage:
        from ..coverage.buckets import load_scores
        cov = load_scores(env_id)

    return Loaded(
        env_id=env_id, domain=env_id.split("_")[0],
        obs=obs, act=act, rew=rew, next_obs=next_obs,
        terminals=term, timeouts=tout, normer=normer,
        raw_rew_mean=float(rew_raw.mean()),
        scaled_rew_mean=float(rew.mean()),
        coverage=cov, ret_range=(min_ret, max_ret),
        normer_symmetrised=mirror_spec is not None,
        raw_obs_mean=raw_mean,
        raw_obs_std=raw_std,
        reward_scale=MAX_EPISODE_STEPS / (max_ret - min_ret),
        augmented=bool(augment),
        n_original=n_original,
    )


def episode_lookup(env_id: str, n_original: int | None = None):
    """Episode structure for a dataset, augmented if `n_original` is given.

    Centralised so a runner cannot pair augmented arrays with the original's
    episode boundaries -- that mismatch would let n-step windows bootstrap from
    one trajectory onto an unrelated one, and nothing downstream would look
    wrong.
    """
    with D4RLDataset(env_id, mode="stream") as ds:
        ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)
    if n_original is None:
        return ep
    from .augment import augmented_episode_lookup

    return augmented_episode_lookup(ep, int(n_original))


def evaluate_per_episode(policy, domain: str, normer: ObsNormalizer,
                         n_episodes: int = 30, seed: int = 0) -> dict:
    """`evaluate`, but returning the per-episode returns.

    Review R1 made this a requirement rather than a convenience. The ranking
    truth is a mean over episodes, and the measured spreads between checkpoints
    (1-2.4% of the score) sit close to the 10-episode SEM (0.34-1.7%), so the
    ranking signal and the evaluation noise were the same size. With per-episode
    returns the comparisons become PAIRED -- the same episode seeds for every
    checkpoint -- and the paired difference SEM is the right statistic for
    "is checkpoint A really better than B".
    """
    import gymnasium as gym
    import torch

    from ..envs.obs_layout import ENV_IDS
    from ..envs.ref_scores import normalized_score

    dev = next(policy.parameters()).device
    env = gym.make(ENV_IDS[domain])
    rets = []
    try:
        for ep in range(n_episodes):
            o, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                ot = torch.as_tensor(normer(o[None, :]), dtype=torch.float32,
                                     device=dev)
                with torch.no_grad():
                    out = policy(ot)
                    vec = out if torch.is_tensor(out) else out.mean
                    a = vec.detach().cpu().numpy()[0]
                o, r, term, trunc, _ = env.step(np.clip(a, -1.0, 1.0))
                total += float(r)
                done = term or trunc
            rets.append(total)
    finally:
        env.close()
    rets = np.asarray(rets, dtype=np.float64)
    scores = np.asarray([normalized_score(domain, r) for r in rets], dtype=np.float64)
    return {"raw_returns": rets.tolist(), "scores": scores.tolist(),
            "raw_mean": float(rets.mean()),
            "raw_sem": float(rets.std(ddof=1) / np.sqrt(len(rets))),
            "score_mean": float(scores.mean()),
            "score_sem": float(scores.std(ddof=1) / np.sqrt(len(scores)))}


def evaluate(policy, domain: str, normer: ObsNormalizer,
             n_episodes: int = 20, seed: int = 0) -> dict:
    """Roll out with the SAME normaliser the policy was trained on."""
    import gymnasium as gym
    import torch

    from ..envs.obs_layout import ENV_IDS
    from ..envs.ref_scores import normalized_score

    dev = next(policy.parameters()).device
    env = gym.make(ENV_IDS[domain])
    rets = []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            ot = torch.as_tensor(normer(o[None, :]), dtype=torch.float32,
                                 device=dev)
            with torch.no_grad():
                dist = policy(ot)
                # Tensor first: a Tensor also has a `.mean` method.
                vec = dist if torch.is_tensor(dist) else dist.mean
                a = vec.detach().cpu().numpy()[0]
            o, r, term, trunc, _ = env.step(np.clip(a, -1.0, 1.0))
            total += float(r)
            done = term or trunc
        rets.append(total)
    env.close()
    rets = np.asarray(rets)
    return {
        "raw_mean": float(rets.mean()),
        "raw_std": float(rets.std()),
        "score_mean": normalized_score(domain, float(rets.mean())),
    }
