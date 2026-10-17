"""M2 -- support-aware per-sample backup length, on top of reference IQL.

Mechanism
---------
An n-step Bellman target accumulates error: each V estimate it consumes is
imperfect, and the errors compound with n. Where the data is dense, V is
reliable and a long backup is affordable -- it buys the low bias that
truncating a long-horizon return would otherwise cost. Where the data is
sparse, V is unreliable and compounding hurts, so the backup is cut short.

The backup length is therefore chosen *per transition* from local coverage:

    high coverage (dense)  -> long  n
    mid                     -> middle
    low  coverage (sparse)  -> short n

Two correctness rules, both unit-tested:

  1. The n-step window NEVER crosses an episode boundary. D4RL stores a flat
     concatenation with no index, so boundaries come from `terminals | timeouts`.
  2. Bootstrapping happens only while the state after the window is still
     inside the same episode. At a trajectory end, `next_obs` belongs to the
     next episode, so using it would be wrong.

Why this is a minimal override
------------------------------
The V update and the actor update are the reference's, unchanged. Only the
critic target changes. That keeps attribution clean: any measured difference
is the mechanism, not a re-implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from src.iql import ImplicitQLearning  # vendored reference

from ..data.episode import EpisodeIndexLookup


class SupportAwareIQL(ImplicitQLearning):
    """Reference IQL with a coverage-conditioned backup length."""

    def __init__(
        self,
        *args,
        episode_lookup: EpisodeIndexLookup,
        coverage: np.ndarray | None = None,
        n_min: int = 1,
        n_max: int = 10,
        fixed_n: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ep = episode_lookup
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        n_data = int(episode_lookup.episode_of.shape[0])
        if coverage is None:
            # Fixed-n mode. Two callers need it:
            #   * GATE 1's n-sweep, where the backup length is the manipulated
            #     variable and must be identical for every transition;
            #   * the "M1 only" cell of GATE 2's 2x2, where the mechanism under
            #     test is equivariance, so coverage must play no role.
            # Uniform coverage is the honest encoding of "no coverage
            # dependence": it keeps this the SAME code path, so the 2x2 cells
            # differ in numbers only, never in which code ran.
            if fixed_n is None:
                raise ValueError("coverage=None requires fixed_n")
            if not (self.n_min <= fixed_n <= self.n_max):
                raise ValueError(
                    f"fixed_n={fixed_n} outside [n_min, n_max] = "
                    f"[{self.n_min}, {self.n_max}]")
            cov = np.zeros(n_data, dtype=np.float64)
            self.edges = np.array([np.nan, np.nan])
            self._requested = np.full(n_data, int(fixed_n), dtype=np.int64)
        else:
            cov = np.asarray(coverage, dtype=np.float64)
            if cov.shape[0] != n_data:
                raise ValueError("coverage length must match the dataset length")
            # Dense = small kNN distance. Same 33/67 percentiles as
            # orl.coverage.buckets so the mechanism and the diagnostic agree.
            self.edges = np.percentile(cov, [33.0, 67.0])
            mid = max(self.n_min, (self.n_min + self.n_max) // 2)
            self._requested = np.where(
                cov <= self.edges[0], self.n_max,
                np.where(cov <= self.edges[1], mid, self.n_min),
            ).astype(np.int64)
        self.fixed_n = None if coverage is not None else int(fixed_n)
        self._edge_lo = float(self.edges[0])
        self._edge_hi = float(self.edges[1])

    def requested_n_summary(self) -> dict:
        uniq, counts = np.unique(self._requested, return_counts=True)
        return {
            "edges": [self._edge_lo, self._edge_hi],
            "n_min": self.n_min, "n_max": self.n_max,
            "fixed_n": self.fixed_n,
            "requested_n_hist": {int(u): int(c) for u, c in zip(uniq, counts)},
        }

    def _post_optimizer_step(self, which: str) -> None:
        """Hook called after each optimiser step. M1 re-ties weights here.

        A no-op in this class, so M2 remains a verbatim override of the
        reference update. Adding M1 later must not require duplicating 45
        lines of that loop.
        """

    def nstep_target(
        self,
        indices: np.ndarray,
        dataset_rewards: np.ndarray,
        dataset_next_obs: np.ndarray,
        dataset_terminals: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Coverage-conditioned n-step return, clipped at episode boundaries.

        `dataset_*` are WHOLE-DATASET arrays indexed by absolute transition id,
        while `indices` selects this batch. Window rows are gathered on the
        host and only the batch-sized slice is uploaded: uploading the whole
        array cost 21 MB per step (1.2 MB rewards + 20 MB next_obs) and was the
        single largest cost of the mechanism.
        """
        rem = self.ep.steps_remaining(indices)
        m = np.minimum(self._requested[indices], rem)
        N = len(dataset_rewards)

        rew_all = np.asarray(dataset_rewards)
        nobs_all = np.asarray(dataset_next_obs)

        total = torch.zeros(len(indices), device=device)
        g = self.discount
        for k in range(self.n_max):
            active_np = (m > k).astype(np.float32)
            if not active_np.any():
                break
            r_at = np.minimum(indices + k, N - 1)
            r_k = torch.as_tensor(rew_all[r_at], dtype=torch.float32, device=device)
            active = torch.as_tensor(active_np, device=device)
            total = total + active * (g**k) * r_k

        inside_np = (m < rem).astype(np.float32)
        if inside_np.any():
            b_at = np.minimum(indices + m - 1, N - 1)
            nobs_b = torch.as_tensor(nobs_all[b_at], dtype=torch.float32,
                                     device=device)
            ended = torch.as_tensor(
                (dataset_terminals[b_at] > 0).astype(np.float32), device=device
            )
            m_t = torch.as_tensor(m.astype(np.float32), device=device)
            coef = torch.as_tensor(inside_np, device=device) * (g**m_t) * (1.0 - ended)
            # no_grad is REQUIRED: the V optimizer step below modifies V's
            # parameters in place, which would invalidate a graph built here.
            with torch.no_grad():
                v_at = self.vf(nobs_b).squeeze(-1)
            total = total + coef * v_at
        return total

    def update(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        indices: np.ndarray | None = None,
        dataset_rewards: np.ndarray | None = None,
        dataset_next_obs: np.ndarray | None = None,
        dataset_terminals: np.ndarray | None = None,
        log_dict: dict | None = None,
    ):
        """Reference update with ONLY the critic target replaced by M2's.

        The value update, the actor update, the target-network EMA and the LR
        schedule are the reference's, unchanged.
        """
        import torch.nn.functional as F

        from src.iql import EXP_ADV_MAX, asymmetric_l2_loss
        from src.util import update_exponential_moving_average

        log_dict = {} if log_dict is None else log_dict

        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        if indices is None:
            targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        else:
            targets = self.nstep_target(
                indices, dataset_rewards, dataset_next_obs,
                dataset_terminals, observations.device,
            )

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        self._post_optimizer_step("value")
        log_dict["value_loss"] = v_loss.item()

        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self._post_optimizer_step("critic")
        log_dict["q_loss"] = q_loss.item()

        update_exponential_moving_average(self.q_target, self.qf, self.alpha)
        self._post_optimizer_step("target")

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self._post_optimizer_step("policy")
        self.policy_lr_schedule.step()
        log_dict["actor_loss"] = policy_loss.item()
        return log_dict
