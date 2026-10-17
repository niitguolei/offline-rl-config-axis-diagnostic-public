"""T9 — plain behaviour cloning.

GATE 0's PRIMARY criterion depends on this module, so two properties matter
more than performance:

1. It must be the *plainest possible* BC — max-likelihood Gaussian, no reward
   weighting, no trajectory filtering, no elite selection. Anything cleverer
   would make the test measure "how good is this BC variant" instead of
   "is the closed-loop pipeline correct".
2. At evaluation it must return the distribution MEAN, never a sample.
   Sampling would inject variance into every episode and make the GATE 0
   diagnostic ladder incomparable between runs.

Actions are used as-is: the retained domains' actions already lie in the env
action space (asserted by orl.envs.gym_eval.check_action_range), so no
rescaling is applied.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from ..nn import mlp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclasses.dataclass
class BCConfig:
    obs_dim: int
    act_dim: int
    hidden: int = 256
    n_hidden: int = 2
    lr: float = 3e-4
    act_low: float = -1.0
    act_high: float = 1.0
    device: str = "cuda"


class BC:
    def __init__(self, cfg: BCConfig) -> None:
        self.cfg = cfg
        self.net = mlp(cfg.obs_dim, 2 * cfg.act_dim, cfg.hidden, cfg.n_hidden).to(cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        dev = self.cfg.device
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=dev)
        act = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)

        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        # Gaussian negative log-likelihood, constants dropped
        nll = (0.5 * ((act - mu) / std).pow(2) + log_std).sum(-1).mean()
        self.opt.zero_grad(set_to_none=True)
        nll.backward()
        self.opt.step()
        return {"bc_nll": float(nll.item()), "bc_mu_absmean": float(mu.abs().mean().item())}

    @torch.no_grad()
    def policy_fn(self):
        """Deterministic policy = distribution mean, clipped to the action space."""
        self.net.eval()
        lo, hi = self.cfg.act_low, self.cfg.act_high

        def fn(obs: torch.Tensor) -> torch.Tensor:
            mu, _ = self.net(obs.to(self.cfg.device)).chunk(2, dim=-1)
            return mu.clamp(lo, hi)

        return fn

