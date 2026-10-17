"""Tanh-squashed Gaussian policy, transcribed from the CQL authors' rlkit fork.

Source (read):
    https://github.com/aviralkumar2907/CQL
    d4rl/rlkit/torch/sac/policies.py  -- TanhGaussianPolicy, TanhNormal

Why this exists, and why it is not just a style choice
-----------------------------------------------------
The vendored IQL reference in this repo uses an UNSQUASHED `GaussianPolicy`.
CQL cannot. Two places in CQL are defined on the bounded action space:

  1. the importance weights subtract `log(0.5 ** A)`, the log-density of the
     uniform distribution on [-1, 1]^A, from the sampled Q values -- that
     baseline only means anything if the policy's density lives on the same set;
  2. the Bellman backup evaluates `Q(s', pi(s'))`. With an unbounded Gaussian the
     actor can put that action arbitrarily far outside the data, the critic
     extrapolates, the actor chases the extrapolation, and the target feeds back.

Point 2 is not hypothetical here. The first version of this arm used the
vendored unsquashed policy, and on walker2d-medium-replay at the real 66k-step
budget the critic mean reached 5.1e14 and the evaluated policy scored -0.27.
The run was killed by the pre-run gate; see `docs/GATES.md`.

Interface: identical to the vendored `GaussianPolicy` (call it to get a
distribution-like object with `.mean`, `.rsample()`, `.log_prob(actions)`, and
`log_prob` summed over action dimensions so its shape matches `TwinQ`'s output).
That is what lets `ConservativeQLearning` accept either policy unchanged.

Constants are the source's: LOG_SIG in [-5, 2], mean clamped to [-9, 9]. The
source learns a separate log-std head; this keeps the vendored policy's single
`log_std` parameter instead, so the two algorithms differ in the squashing and
nothing else that is not already recorded.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.util import mlp  # vendored reference

LOG_SIG_MIN = -5.0
LOG_SIG_MAX = 2.0
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class SquashedNormal:
    """tanh(N(mu, sigma)) with the change-of-variables log-density."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self._normal = torch.distributions.Normal(mean, std)
        self.mean = torch.tanh(mean)
        self.eps = eps

    def rsample(self) -> torch.Tensor:
        return torch.tanh(self._normal.rsample())

    def sample(self) -> torch.Tensor:
        return torch.tanh(self._normal.sample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Summed over action dimensions, so the shape matches `TwinQ`."""
        bounded = action.clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(bounded)
        elementwise = (self._normal.log_prob(pre_tanh)
                       - torch.log(1.0 - bounded * bounded + self.eps))
        return elementwise.sum(dim=-1)


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_hidden: int = 2) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> SquashedNormal:
        mean = self.net(obs).clamp(MEAN_MIN, MEAN_MAX)
        std = torch.exp(self.log_std.clamp(LOG_SIG_MIN, LOG_SIG_MAX))
        return SquashedNormal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False,
            enable_grad: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()
