"""CQL(H) as the second training algorithm, transcribed from the authors' code.

Source (read, not vendored for execution):
    https://github.com/aviralkumar2907/CQL
    d4rl/rlkit/torch/sac/cql.py   -- class CQLTrainer, method train_from_torch
    Kumar, Zhou, Tucker, Levine. "Conservative Q-Learning for Offline
    Reinforcement Learning." arXiv:2006.04779.

Why transcribe instead of importing
-----------------------------------
The authors' repo ships its own 2020-era `d4rl` and `rlkit` and expects
`gym` + `mujoco-py`, while this project runs `gymnasium` + `mujoco` 3.x and a
different data layer. Running their stack would mean a second environment, and
the two algorithms would then differ in their data pipeline as well as in their
algorithm -- which is exactly the kind of confound this paper is about.

So the loss is transcribed here line by line, and both algorithms share one
harness: same reward rescaling (`harness.load`), same observation normaliser,
same episode boundaries, same evaluator, same estimators. Two deviations are
deliberate and are recorded because they change what the numbers mean:

  1. POLICY CLASS. This arm uses a tanh-squashed policy (`orl.algos.tanh_policy`,
     transcribed from the same repo), while the IQL arm keeps the vendored
     reference's unsquashed `GaussianPolicy`. CQL cannot use the unsquashed one:
     its importance weights are defined against the uniform density on
     [-1, 1]^A, and its backup evaluates Q(s', pi(s')), so an unbounded actor
     pushes the critic outside the data and the target feeds back. That is not a
     prediction -- the first version of this arm used the unsquashed policy and
     diverged on walker2d-medium-replay at the real 66k-step budget (critic mean
     5.1e14, evaluated score -0.27). The two arms therefore differ in policy
     class as well as in algorithm; that is recorded rather than hidden, and the
     cross-algorithm comparison is not the paper's claim -- the claim is about
     the mechanism inside each arm.

  2. BC WARM-UP. The authors set `policy_eval_start=40000`, i.e. the first 40k
     gradient steps train the actor by behaviour cloning. Their MuJoCo runs take
     ~3M steps, so that is 1.3% of the run; at this project's 100k budget the
     same absolute number would be 40%. The default here is 0 and the parameter
     is exposed, so the deviation is a knob rather than a hidden constant.

Everything else follows the source: `min_q_version=3` (CQL(H), the
importance-sampled variant), `temp=1.0`, `num_random=10`, twin Q-functions with
separate targets and a soft update, automatic entropy tuning for the SAC-style
actor temperature, and `deterministic_backup` defaulting to True as in the
paper's Equation 3.

The quantity this project cares about is `min_q_weight`, the authors' alpha:
it multiplies the CQL term, so it is the conservatism knob and the analogue of
IQL's expectile tau.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.util import DEFAULT_DEVICE, update_exponential_moving_average  # vendored


class ConservativeQLearning(nn.Module):
    """CQL(H) on the vendored twin-Q / Gaussian-policy networks."""

    def __init__(
        self,
        qf,
        policy,
        optimizer_factory,
        max_steps: int,
        *,
        min_q_weight: float = 1.0,
        temp: float = 1.0,
        num_random: int = 10,
        min_q_version: int = 3,
        discount: float = 0.99,
        soft_target_tau: float = 5e-3,
        reward_scale: float = 1.0,
        deterministic_backup: bool = True,
        use_automatic_entropy_tuning: bool = True,
        policy_eval_start: int = 0,
        target_entropy: float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        # The vendored `ImplicitQLearning` moves its networks to DEFAULT_DEVICE
        # inside the constructor; this mirrors that, while letting a caller pin a
        # device explicitly (the tests run on CPU on a machine that has a GPU).
        self.device = torch.device(device) if device is not None else DEFAULT_DEVICE
        self.qf = qf.to(self.device)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(self.device)
        self.policy = policy.to(self.device)

        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.temp = temp
        self.min_q_weight = min_q_weight
        self.min_q_version = min_q_version
        self.num_random = num_random
        self.deterministic_backup = deterministic_backup
        self.policy_eval_start = policy_eval_start

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if use_automatic_entropy_tuning:
            # target entropy = -dim(A), as in the source (soft actor-critic default)
            self.target_entropy = (target_entropy if target_entropy is not None
                                   else -float(policy.log_std.shape[0]))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optimizer_factory([self.log_alpha])
        self._epoch = 0

    # --- helpers ---------------------------------------------------------
    def _sample_action(self, obs):
        """Reparameterised action and its log-density, from the vendored policy."""
        dist = self.policy(obs)
        action = dist.rsample()
        return action, dist.log_prob(action)

    def _q_values(self, obs, actions):
        """q1/q2 on (B, N, A) actions; returns two (B, N, 1) tensors.

        Mirrors the source's `_get_tensor_values`: observations are repeated so
        that every sampled action is scored against its own state.
        """
        n_actions = actions.shape[1]
        obs_rep = obs.unsqueeze(1).repeat(1, n_actions, 1).reshape(
            obs.shape[0] * n_actions, obs.shape[1])
        act_rep = actions.reshape(obs.shape[0] * n_actions, actions.shape[-1])
        q1, q2 = self.qf.both(obs_rep, act_rep)
        return q1.reshape(obs.shape[0], n_actions, 1), q2.reshape(obs.shape[0], n_actions, 1)

    def _cql_actions(self, obs, next_obs):
        """The action sets the conservative term scores, drawn once per update.

        Hoisted out of `_cql_term` so that the action samples come from the same
        policy snapshot the source uses: the source computes them before its
        actor step, and the actor step is moved earlier here for a graph-safety
        reason (see `update`). Detached, because the source zeroes the policy
        gradients that the critic loss would otherwise accumulate.
        """
        batch, action_dim = obs.shape[0], self.policy.log_std.shape[0]
        random_actions = torch.empty(
            batch * self.num_random, action_dim,
            dtype=obs.dtype, device=obs.device).uniform_(-1.0, 1.0)
        random_actions = random_actions.reshape(batch, self.num_random, action_dim)
        with torch.no_grad():
            curr_actions, curr_log_pis = self._sample_action(
                obs.unsqueeze(1).repeat(1, self.num_random, 1).reshape(
                    batch * self.num_random, obs.shape[1]))
            next_actions, next_log_pis = self._sample_action(
                next_obs.unsqueeze(1).repeat(1, self.num_random, 1).reshape(
                    batch * self.num_random, next_obs.shape[1]))
        shape = (batch, self.num_random, action_dim)
        return {
            "random": random_actions,
            "curr": curr_actions.reshape(*shape),
            "next": next_actions.reshape(*shape),
            "curr_log_pi": curr_log_pis.reshape(batch, self.num_random, 1),
            "next_log_pi": next_log_pis.reshape(batch, self.num_random, 1),
        }

    def _cql_term(self, obs, q_preds, action_sets):
        """The conservative penalty, exactly as `min_q_version=3` in the source."""
        action_dim = self.policy.log_std.shape[0]
        # One Q evaluation over the concatenated action sets instead of three.
        # The source calls `_get_tensor_values` once per set; grouping them is
        # the same arithmetic on the same tensors -- the split below reproduces
        # the source's `cat([rand, next, curr])` exactly -- and it costs one
        # observation repeat and two MLP calls instead of three of each. The
        # step is launch-bound (578 kernels per step, 86% of the wall clock),
        # so removing launches is what buys time.
        n = self.num_random
        merged = torch.cat([action_sets["random"], action_sets["next"],
                            action_sets["curr"]], dim=1)
        q1, q2 = self._q_values(obs, merged)
        q1_rand, q1_next, q1_curr = q1[:, :n], q1[:, n:2 * n], q1[:, 2 * n:]
        q2_rand, q2_next, q2_curr = q2[:, :n], q2[:, n:2 * n], q2[:, 2 * n:]
        curr_log_pis, next_log_pis = action_sets["curr_log_pi"], action_sets["next_log_pi"]

        if self.min_q_version == 3:
            # importance-sampled CQL(H): subtract the density of each action
            random_density = float(np.log(0.5 ** action_dim))
            cat_q1 = torch.cat([q1_rand - random_density,
                                q1_next - next_log_pis,
                                q1_curr - curr_log_pis], dim=1)
            cat_q2 = torch.cat([q2_rand - random_density,
                                q2_next - next_log_pis,
                                q2_curr - curr_log_pis], dim=1)
        elif self.min_q_version == 2:
            # CQL(rho): no importance weights, the data action included
            cat_q1 = torch.cat([q1_rand, q_preds[0].unsqueeze(1), q1_next, q1_curr], dim=1)
            cat_q2 = torch.cat([q2_rand, q_preds[1].unsqueeze(1), q2_next, q2_curr], dim=1)
        else:
            raise ValueError(f"unsupported min_q_version={self.min_q_version}")

        def penalty(cat_q, q_pred):
            lse = torch.logsumexp(cat_q / self.temp, dim=1).mean() * self.temp
            return (lse - q_pred.mean()) * self.min_q_weight

        return penalty(cat_q1, q_preds[0]), penalty(cat_q2, q_preds[1])

    # --- update ----------------------------------------------------------
    def update(self, observations, actions, next_observations, rewards, terminals,
               return_stats: bool = False):
        """One gradient step.

        `return_stats` is off by default and that is deliberate: building the
        dict costs three `.item()` calls, each of which forces a device
        synchronisation, and every caller in this repo ignores the return value.
        Paying three syncs per step for nothing is the kind of cost that hides
        inside a launch-bound loop.
        """
        self._epoch += 1
        # One policy snapshot for the whole update, taken before the actor step,
        # as in the source.
        action_sets = self._cql_actions(observations, next_observations)
        q1_pred, q2_pred = self.qf.both(observations, actions)

        # --- actor -------------------------------------------------------
        # The source computes the actor loss first but backpropagates it LAST,
        # after both critic steps. PyTorch 2.x rejects that order: stepping the
        # critic modifies parameters the actor's graph still needs, and the
        # backward raises "a variable needed for gradient computation has been
        # modified by an inplace operation". Stepping the actor first is the same
        # update -- both versions score the actor against the pre-critic Q -- and
        # it is graph-safe. The tests exercise the gradient path directly.
        new_actions, log_pi = self._sample_action(observations)
        # NOT detached: the actor gradient flows through the reparameterised
        # action, which is what `policy_loss.backward()` is for in the source.
        q_new_actions = self.qf(observations, new_actions)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = torch.ones((), dtype=observations.dtype, device=observations.device)
        if self._epoch < self.policy_eval_start:
            # The source's warm-up: behaviour-clone the actor for the first epochs.
            policy_loss = (alpha * log_pi - self.policy(observations).log_prob(actions)).mean()
        else:
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        # --- critic ------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_pi = self._sample_action(next_observations)
            target_q = self.q_target(next_observations, next_actions)
            if not self.deterministic_backup:
                target_q = target_q - alpha * next_log_pi
            target_q = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_q

        # The source keeps two separate optimisers, one per Q, each with its own
        # MSE plus its own CQL term. Summing here gives each parameter exactly the
        # same gradient, so a single optimiser is equivalent -- averaging would
        # halve the critic step size.
        td_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        cql1, cql2 = self._cql_term(observations, (q1_pred, q2_pred), action_sets)
        q_loss = td_loss + cql1 + cql2

        # Clears the Q-side gradients the actor's backward just accumulated, as
        # the source does before its critic backward.
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        update_exponential_moving_average(self.q_target, self.qf, self.soft_target_tau)

        if not return_stats:
            return None
        return {"td_loss": float(td_loss.detach()), "q_loss": float(q_loss.detach()),
                "cql_term": float((cql1 + cql2).detach())}
