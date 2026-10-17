"""TD3+BC, transcribed from the authors' description, as the third algorithm.

Source: Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement
Learning", arXiv:2106.06860.

    critic:   min_theta  E[ (Q_i(s,a) - y)^2 ],  y = r + gamma * min_i Q'_i(s', pi'(s') + eps)
    actor:    max_phi    E[ lambda * Q_1(s, pi(s)) - (pi(s) - a)^2 ],
                         lambda = alpha / (1/N) sum |Q_1(s,a)|, detached
    targets updated softly; the actor and targets update every `policy_delay`
    critic steps.

WHY THIS ALGORITHM, AND WHAT IT IS FOR
--------------------------------------
The mechanism under test is: an estimator inherits the hyperparameter that its
TRAINING OBJECTIVE contains. IQL and CQL both put the compared knob inside the
critic's objective (tau in the expectile value update, alpha in the conservative
penalty), and both critics turn out to be deterministic functions of it -- rank
correlation with the axis is exactly +/-1.00 in 48 of 48 slices.

TD3+BC puts its alpha in the ACTOR loss only. The critic is a plain TD error.
So the mechanism predicts the opposite here, which is the strongest available
test of it: a case it predicts will NOT inherit the knob. The prediction is
registered in docs/第十一轮_同批对照.md section 7 before these runs exist.

One indirect path is recorded up front rather than discovered later: the critic
target contains pi'(s'), and pi' depends on alpha, so alpha does reach the
critic indirectly. The prediction is therefore "substantially weaker than for
IQL/CQL", not "no dependence at all".
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.util import DEFAULT_DEVICE, update_exponential_moving_average  # vendored


class TD3BC(nn.Module):
    def __init__(
        self,
        qf,
        policy,
        optimizer_factory,
        max_steps: int,
        *,
        alpha: float = 2.5,
        discount: float = 0.99,
        soft_target_tau: float = 5e-3,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if device is not None else DEFAULT_DEVICE
        self.qf = qf.to(self.device)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(self.device)
        self.policy = policy.to(self.device)
        self.policy_target = copy.deepcopy(policy).requires_grad_(False).to(self.device)

        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        # The actor only steps every `policy_delay` critic steps, so the schedule
        # is sized in actor steps. Stepping it on the epochs where the optimizer
        # did not run would both stretch the schedule and trip PyTorch's
        # "lr_scheduler.step() before optimizer.step()" guard.
        self.policy_lr_schedule = CosineAnnealingLR(
            self.policy_optimizer, max(1, max_steps // policy_delay))

        self.alpha = alpha
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self._epoch = 0

    def update(self, observations, actions, next_observations, rewards, terminals,
               return_stats: bool = False):
        self._epoch += 1

        # --- critic ------------------------------------------------------
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.policy_target(next_observations) + noise).clamp(-1.0, 1.0)
            target_q = self.q_target(next_observations, next_actions)
            target_q = rewards + (1.0 - terminals) * self.discount * target_q

        q1, q2 = self.qf.both(observations, actions)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # --- actor, delayed ----------------------------------------------
        policy_loss = None
        if self._epoch % self.policy_delay == 0:
            pi = self.policy(observations)
            q_pi = self.qf.q1(torch.cat([observations, pi], dim=1))
            # The authors' lambda: scale the Q term by alpha relative to the Q
            # magnitude, so alpha sets the Q-versus-BC balance rather than an
            # absolute scale. Detached, as in the source.
            with torch.no_grad():
                q_data = self.qf.q1(torch.cat([observations, actions], dim=1))
                lmbda = self.alpha / q_data.abs().mean().clamp(min=1e-6)
            policy_loss = -lmbda * q_pi.mean() + F.mse_loss(pi, actions)
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_optimizer.step()
            self.policy_lr_schedule.step()
            update_exponential_moving_average(self.policy_target, self.policy,
                                              self.soft_target_tau)

        update_exponential_moving_average(self.q_target, self.qf, self.soft_target_tau)

        if not return_stats:
            return None
        return {"q_loss": float(q_loss.detach()),
                "policy_loss": float(policy_loss.detach()) if policy_loss is not None
                else float("nan")}
