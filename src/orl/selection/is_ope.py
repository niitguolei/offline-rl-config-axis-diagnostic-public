"""Importance-sampling OPE: the family whose failure mode is the mirror of FQE's.

FQE failed here by being policy-INSENSITIVE: it returned ~40 for two policies
whose true values were -0.51 and 65.11, because it estimates from the data's
state distribution and never sees the policy's closed-loop consequences.
Importance sampling attacks exactly that: it reweights the OBSERVED returns by
how likely the evaluation policy would have taken the observed actions, so a
policy that behaves differently from the data gets small weights on the
transitions it would not have taken. Its failure mode is the opposite one --
weight degeneracy when the policies differ over long horizons.

THE CONVENTION PROBLEM, which is a real result and not a nuisance
---------------------------------------------------------------
Importance sampling needs a probability DENSITY for the evaluation policy. The
deterministic mean policy that `harness.evaluate`, the critic mean and FQE all
score has a degenerate (point-mass) density, so IS cannot address it at all.
Therefore:

  * the IS family evaluates the STOCHASTIC policy pi_theta(a|s);
  * its ground truth must be the stochastic policy's value too, obtained by
    rolling out sampled actions;
  * those numbers are NOT comparable with the deterministic-policy numbers from
    the FQE/critic-mean family.

That is worth stating plainly rather than papering over: "which OPE estimators
are even applicable" depends on which object you call the policy, and the
detrimental case is invisible unless you keep the two conventions apart.

The behaviour policy is not stored in D4RL, so it is fitted by maximum
likelihood (the same Gaussian BC used in GATE 0). When the evaluation policy IS
the behaviour model, every ratio is exactly 1 and the estimates must collapse to
the behaviour's own empirical returns -- that is the unit test the weights are
held to.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclasses.dataclass(eq=False)   # eq=True clears __hash__, which nn.Module needs
class BehaviourGaussian(torch.nn.Module):
    """State-dependent Gaussian behaviour model, fitted by maximum likelihood."""

    obs_dim: int
    act_dim: int
    hidden: int = 256
    n_hidden: int = 2

    def __post_init__(self) -> None:
        super().__init__()
        from src.util import mlp

        self.net = mlp([self.obs_dim, *([self.hidden] * self.n_hidden),
                        2 * self.act_dim])

    def forward(self, obs: torch.Tensor):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mu, log_std = self(obs)
        # diagonal Gaussian log density, constants included so a ratio against a
        # reference policy is exact rather than up to a constant
        var = torch.exp(2.0 * log_std)
        return (-0.5 * ((act - mu) ** 2 / var + 2.0 * log_std
                        + np.log(2.0 * np.pi))).sum(-1)


def fit_behaviour(obs: np.ndarray, act: np.ndarray, *, steps: int = 40_000,
                  batch_size: int = 256, lr: float = 3e-4, device: str = "cuda",
                  seed: int = 0) -> BehaviourGaussian:
    """MLE fit of a Gaussian behaviour model. D4RL stores no behaviour densities."""
    torch.manual_seed(seed)
    n, obs_dim = obs.shape[0], obs.shape[1]
    act_dim = act.shape[1]
    model = BehaviourGaussian(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    O = torch.as_tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
    A = torch.as_tensor(np.ascontiguousarray(act), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        j = torch.as_tensor(rng.integers(0, n, batch_size), device=device)
        loss = -model.log_prob(O[j], A[j]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def log_ratio(eval_log_prob, behaviour, obs: torch.Tensor,
              act: torch.Tensor) -> np.ndarray:
    """log pi_eval(a|s) - log pi_beta(a|s), per transition."""
    return (eval_log_prob(obs, act) - behaviour.log_prob(obs, act)).cpu().numpy()


def episode_slices(starts: np.ndarray, ends: np.ndarray):
    return list(zip(np.asarray(starts).tolist(), np.asarray(ends).tolist()))


def weighted_is(rewards: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                log_r: np.ndarray, gamma: float = 0.99,
                clip: float | None = None) -> dict:
    """Self-normalised whole-episode importance sampling, with diagnostics.

    Returns the estimate plus the numbers that say whether it is usable:
    `ess` (effective sample size) and `max_weight_share`. A self-normalised IS
    estimate whose ESS is ~1 is not an estimate of anything -- it is one lucky
    episode, and reporting it without the ESS would hide that completely.
    """
    log_r = np.asarray(log_r, dtype=np.float64)
    if clip is not None:
        log_r = np.clip(log_r, -clip, clip)
    w, g = [], []
    for s, e in episode_slices(starts, ends):
        w.append(float(np.exp(log_r[s:e + 1].sum())))
        disc = gamma ** np.arange(e - s + 1)
        g.append(float((rewards[s:e + 1] * disc).sum()))
    w = np.asarray(w)
    g = np.asarray(g)
    total = w.sum()
    if total <= 0:
        return {"value": float("nan"), "ess": 0.0, "max_weight_share": 1.0,
                "n_episodes": len(w), "note": "all weights underflowed"}
    share = w / total
    return {"value": float((share * g).sum()),
            "ess": float(1.0 / np.square(share).sum()),
            "max_weight_share": float(share.max()),
            "n_episodes": int(len(w)),
            "mean_return_unweighted": float(g.mean())}


def pdis(rewards: np.ndarray, starts: np.ndarray, ends: np.ndarray,
         log_r: np.ndarray, gamma: float = 0.99,
         clip: float | None = None) -> dict:
    """Per-decision importance sampling (NOT self-normalised).

        V_hat = (1/n) * sum_i sum_t gamma^t * rho^i_{1:t} * r^i_t

    Unbiased for V^pi under the behaviour distribution, and high variance --
    the standard counterpart to WIS's low-variance/biased pair. Per-decision
    partial products rather than whole-episode products keep the variance finite
    over a 1000-step episode.

    The first version of this file implemented a self-normalised per-decision
    variant whose normalisation I could not justify: with unit ratios it returns
    `mean(G) / sum_t gamma^t` rather than `mean(G)`, which a unit test caught
    (it was off by 0.70 on a toy). Dropped rather than shipped with a hand-waved
    normaliser. The property that matters is testable and now holds: with unit
    ratios this reduces exactly to the mean discounted episode return.
    """
    log_r = np.asarray(log_r, dtype=np.float64)
    if clip is not None:
        log_r = np.clip(log_r, -clip, clip)
    total = 0.0
    partials, n_ep = [], 0
    for s, e in episode_slices(starts, ends):
        cum = 0.0
        for k, t in enumerate(range(s, e + 1)):
            cum += log_r[t]
            rho = float(np.exp(cum))
            total += (gamma ** k) * rho * rewards[t]
            partials.append((gamma ** k) * rho)
        n_ep += 1
    p = np.asarray(partials, dtype=np.float64)
    share = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    return {"value": float(total / max(n_ep, 1)),
            "ess": float(1.0 / np.square(share).sum()),
            "max_weight_share": float(share.max()), "n_episodes": n_ep}
