"""DEPRECATED -- DO NOT USE AS A BASELINE. Written from memory, not from the
reference implementation, and it trained to the wrong target.

Why it is wrong (all four verified by measurement, 2026-10-05):

  1. `next_observations` fell through to the bool branch of the reader, so
     V(s') saw an all-ones array. Symptom: corr(Q, r) = 0.9986.
  2. No observation normalisation. halfcheetah per-dim std spans 53x
     (0.170 .. 9.106), so the first-layer gradient was dominated by the
     joint-velocity dims.
  3. LayerNorm was added here on the false claim that "CORL standardises the
     critic". CORL does not. It suppressed divergence (1e15 -> bounded)
     but is a deviation from the reference, not a fix.
  4. ROOT CAUSE: the reference rescales rewards for the locomotion domains,
         dataset['rewards'] /= (max_ret - min_ret); *= max_episode_steps
     On halfcheetah_medium_replay-v2 that is x0.1778, so the reachable
     fixed point is 55.0, not the 309.3 this file was chasing. The physics
     was right; the target was wrong, so nothing could validate.

Use the vendored reference instead:
    vendor/IQL-PyTorch/src/iql.py     (gwthomas/IQL-PyTorch, MIT)
    experiments/run_reference_iql.py  (runner wired to our HDF5 loader)

Mechanism modules (M1 symmetry, M2 n-step) are to be ported onto the
reference, not onto this file. See docs/BASELINE_STATUS.md.

---

Original docstring follows.

IQL — Implicit Q-Learning (Kostrikov et al., 2021).

S0 scope: a correct, runnable implementation. The plan deliberately does NOT
require literature-grade reproduction here — GATE 0 judges the pipeline via
BC self-consistency, not by matching published IQL numbers.

The Bellman target is isolated in `compute_target` on purpose: mechanism M2
(support-aware backup length) will replace that single method, so M2 becomes
an override rather than a rewrite. M2 is NOT implemented in S0 (explicit
non-goal).
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn import TanhGaussianPolicy, mlp


@dataclasses.dataclass
class IQLConfig:
    obs_dim: int
    act_dim: int
    hidden: int = 256
    n_hidden: int = 2
    expectile: float = 0.7      # tau
    beta: float = 3.0           # inverse temperature for AWR
    gamma: float = 0.99
    tau: float = 0.005          # polyak coefficient for Q targets
    lr: float = 3e-4
    max_weight: float = 100.0
    # None disables clipping. The T3 calibration diverged without it
    # (q_loss 60 -> 480 -> 596,813); see experiments/diag_divergence.py.
    clip_grad_norm: float | None = None
    layer_norm: bool = True
    device: str = "cuda"


class IQL:
    def __init__(self, cfg: IQLConfig) -> None:
        self.cfg = cfg
        d = cfg.device
        self.q1 = mlp(cfg.obs_dim + cfg.act_dim, 1, cfg.hidden, cfg.n_hidden,
                      layer_norm=cfg.layer_norm).to(d)
        self.q2 = mlp(cfg.obs_dim + cfg.act_dim, 1, cfg.hidden, cfg.n_hidden,
                      layer_norm=cfg.layer_norm).to(d)
        self.q1_targ = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_targ = copy.deepcopy(self.q2).requires_grad_(False)
        self.v = mlp(cfg.obs_dim, 1, cfg.hidden, cfg.n_hidden,
                     layer_norm=cfg.layer_norm).to(d)
        self.pi = TanhGaussianPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.n_hidden).to(d)

        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _sa(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, act], dim=-1)

    def _min_q_targ(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.min(self.q1_targ(self._sa(obs, act)), self.q2_targ(self._sa(obs, act)))

    # ------------------------------------------------------- the M2 seam
    def compute_target(
        self,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        not_done: torch.Tensor,
        obs: torch.Tensor | None = None,
        act: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard IQL target: r + gamma * (1 - terminal) * V(s').

        M2 will override this to use a per-sample backup length chosen from the
        coverage module. Keeping it as one method means M2 is an override
        rather than a rewrite.

        `not_done` uses `terminals` only, NOT `timeouts`: D4RL trajectories
        are truncated at 1000 steps, not terminated, and treating a timeout as
        terminal would truncate the bootstrap at every episode boundary.
        """
        with torch.no_grad():
            return reward + self.cfg.gamma * not_done * self.v(next_obs).squeeze(-1)

    # -------------------------------------------------------------- updates
    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.cfg
        dev = cfg.device
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=dev)
        act = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)
        rew = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=dev).squeeze(-1)
        nobs = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=dev)
        term = torch.as_tensor(batch["terminals"], dtype=torch.float32, device=dev).squeeze(-1)
        not_done = 1.0 - term

        # --- critic: regress toward r + gamma * V(s')
        with torch.no_grad():
            target = self.compute_target(rew, nobs, not_done, obs, act)
        q1_pred = self.q1(self._sa(obs, act)).squeeze(-1)
        q2_pred = self.q2(self._sa(obs, act)).squeeze(-1)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                cfg.clip_grad_norm,
            )
        self.opt_q.step()

        # --- value: expectile regression toward min(Q_target(s,a))
        with torch.no_grad():
            q_min = self._min_q_targ(obs, act).squeeze(-1)
        v_pred = self.v(obs).squeeze(-1)
        diff = q_min - v_pred
        weight = torch.abs(cfg.expectile - (diff < 0).float())
        v_loss = (weight * diff.pow(2)).mean()
        self.opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), cfg.clip_grad_norm)
        self.opt_v.step()

        # --- actor: advantage-weighted regression
        with torch.no_grad():
            adv = q_min - self.v(obs).squeeze(-1)
            awr_w = torch.clamp(torch.exp(cfg.beta * adv), max=cfg.max_weight)
        mu, log_std = self.pi(obs)
        std = log_std.exp()
        # log pi(a|s) for the squashed-Gaussian, up to the tanh Jacobian which
        # is constant w.r.t. parameters at fixed action.
        logp = (-0.5 * ((act - mu) / std).pow(2) - log_std).sum(-1)
        pi_loss = -(awr_w * logp).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.clip_grad_norm)
        self.opt_pi.step()

        # --- polyak update of the Q targets
        with torch.no_grad():
            for net, targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p, tp in zip(net.parameters(), targ.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)

        return {
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "q_mean": float(q1_pred.mean().item()),
        }

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def policy_fn(self):
        """Return a BatchPolicy for orl.envs.gym_eval.evaluate."""
        self.pi.eval()

        def fn(obs: torch.Tensor) -> torch.Tensor:
            return self.pi.act(obs.to(self.cfg.device))

        return fn
