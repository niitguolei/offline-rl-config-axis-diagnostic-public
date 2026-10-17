"""Small network builders.

DEPRECATED for the IQL baseline: `mlp(..., layer_norm=True)` and
`TanhGaussianPolicy` were written for our own IQL, which is withdrawn.
The reference uses plain `Linear -> ReLU` with a GLOBAL `log_std`
parameter (not one produced by the network) and a Tanh output head.
See vendor/IQL-PyTorch/src/{util,policy}.py and docs/BASELINE_STATUS.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: int = 256,
    n_hidden: int = 2,
    activation: type[nn.Module] = nn.ReLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    """The standard 2x256 MLP used by IQL / CQL / TD3+BC on D4RL.

    `layer_norm=True` adds LayerNorm after each hidden activation. Standard
    offline-RL implementations (e.g. CORL) put it in the critic and value
    nets specifically to stabilise bootstrapped value learning. Without it,
    expectile regression at tau=0.7 sits above the mean of Q, which makes Q
    strictly exceed its own expectation at every backup -- a runaway loop.
    Measured without it: q_loss 8.6e20 at 100k steps, 7.2e21 at 150k.
    """
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), activation()]
        if layer_norm:
            layers += [nn.LayerNorm(hidden)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """Squashed-Gaussian actor.

    Used as a deterministic policy at evaluation time by returning the mean,
    which keeps evaluation free of sampling variance. That matters for GATE 0:
    a sampled BC policy would add noise and make the diagnostic ladder
    incomparable between runs.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden, n_hidden)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action = tanh(mu)."""
        mu, _ = self.forward(obs)
        return torch.tanh(mu)
