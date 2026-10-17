"""M1 (+ optional M2) as a drop-in subclass of the vendored reference IQL.

M1 -- symmetry-equivariant pessimism
----------------------------------
Weights are tied onto the equivariant subspace AFTER every optimiser step, so
the critic, the value function and (optionally) the actor are exactly
equivariant at all times, not just at initialisation. Verified 2026-10-06 on
walker2d_medium_replay-v2 over20k steps:

    re-symmetrised OFF : V=37.23  eq_V=2.79e-01  eq_Q=2.74e-01
    re-symmetrised ON  : V=38.24eq_V=2.55e-08  eq_Q=2.81e-08

Two things this settles:

  * Adam survives the projection. Value trajectories are within ~3% of the
    untied run, so the moment estimates do not fight the tying: gradients of an
    equivariant loss live in the same subspace the projection enforces.
  * Equivariance error GROWS during ordinary training (0.094 -> 0.279 over
    20k steps). So the reference critic drifts AWAY from the environment's
    symmetry the longer it trains. That is a sharper motivation for M1 than
    "symmetry is nice": it is a measurable training-time drift.

M2 -- support-aware backup length
---------------------------------
Composes with M1 so GATE 2's 2x2 (equivariance x support) runs from one class.
"""

from __future__ import annotations

import numpy as np
import torch

from .blocks import concat_blocks
from .equivariance import (
    MirrorSpec,
    apply_mirror_act,
    apply_mirror_obs,
    derive_mirror,
    prepare_mirror_tensors,
    symmetrize_linear_stack,
)
from .support_aware import SupportAwareIQL


class EquivariantIQL(SupportAwareIQL):
    """M1 on the reference IQL, optionally composed with M2.

    Construction
    ------------
    The critic sees the concatenated state-action vector, so equivariance
    Q(g.s, g.a) == Q(s, a) comes from tying the FIRST and LAST Linear of each
    Q head against the *concatenated* mirror `[obs_perm, act_perm]` (with the
    block offset applied -- see blocks.py for why that arithmetic lives in one
    place). The value network's output is a scalar, so its output transform is
    the identity.

    Tying must be re-applied AFTER every optimiser step. Adam updates the free
    parameters and breaks the tie on the very next step, so tying once would
    make equivariance last exactly one step. Verified on walker2d over 20k
    steps: eq_V/eq_Q fall from ~2.8e-01 (never re-tied, and GROWING during
    training) to ~2.5e-08 (re-tied every step).

    Composition with M2 is by inheritance: `EquivariantIQL` IS a
    `SupportAwareIQL`, so GATE 2's 2x2 (equivariance x support) is four cells
    of this one class differing only in arguments. `coverage=None` with
    `fixed_n=k` is the "support off" setting, which runs the identical n-step
    target code at a uniform length.

    Scope: M1's claim is about *where pessimism is allocated*, so what gets
    tied is the critic and the value function. The actor is optionally tied but
    OFF by default, because a tied actor would be only PARTIALLY equivariant:
    the Gaussian's `log_std` is a free parameter and sampling noise is not
    transformed, so `pi(g.s) != g.pi(s)` exactly. Tying the mean alone would
    claim an equivariance the policy does not have. Whether tying it helps is
    measured (actor equivariance error), not assumed.

    Precondition
    -----------
    The network reads NORMALISED observations, so the mirror being enforced is
    the mirror of the normalised space. Per-dimension standardisation carries
    the environment's mirror only if the statistics are themselves
    mirror-symmetric, and on walker2d_medium_replay-v2 they are not (the
    residual `n(g.s) - g(n(s))` reaches 3.22 std). Pass
    `harness.load(..., mirror_spec=spec)` to orbit-average the statistics;
    `experiments/run_m1.py --mode ref_sym` is the matched control for that
    change, so the two effects are never conflated.
    """

    def __init__(
        self,
        *args,
        domain: str,
        obs_dim: int,
        act_dim: int,
        tie_policy: bool = False,
        verify_symmetry: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.domain = domain
        self.spec: MirrorSpec = derive_mirror(
            domain, obs_dim=int(obs_dim), act_dim=int(act_dim))
        self.tie_policy = bool(tie_policy)
        self.n_ties = 0
        if verify_symmetry:
            # Tying imposes V(g.s) == V(s). If g is not the environment's
            # symmetry, that is not a constraint but a WRONG constraint, and
            # the runs would silently fit a value the MDP contradicts.
            from .augment import require_symmetry

            self.symmetry_defect = require_symmetry(domain, self.spec)
        else:
            self.symmetry_defect = None

        # The concatenated state-action mirror, offset in ONE place.
        self.sa_perm = concat_blocks(self.spec.obs_perm, self.spec.act_perm)
        self.sa_sign = np.concatenate([self.spec.obs_sign, self.spec.act_sign])

        # Mirror indices uploaded once. See prepare_mirror_tensors for why.
        dev = next(self.qf.parameters()).device
        ident = (np.arange(1), np.ones(1))
        self._prep_v = prepare_mirror_tensors(
            self.spec.obs_perm, self.spec.obs_sign, *ident, dev)
        self._prep_q = prepare_mirror_tensors(
            self.sa_perm, self.sa_sign, *ident, dev)
        self._prep_pi = (prepare_mirror_tensors(
            self.spec.obs_perm, self.spec.obs_sign,
            self.spec.act_perm, self.spec.act_sign, dev)
            if self.tie_policy else None)
        self.resymmetrise()

    # ------------------------------------------------------------------ tying
    @torch.no_grad()
    def _tie_value(self) -> None:
        symmetrize_linear_stack(self.vf.v, self.spec.obs_perm, self.spec.obs_sign,
                                np.arange(1), np.ones(1), self._prep_v)

    @torch.no_grad()
    def _tie_q(self, twin: torch.nn.Module) -> None:
        for q in (twin.q1, twin.q2):
            symmetrize_linear_stack(q, self.sa_perm, self.sa_sign,
                                    np.arange(1), np.ones(1), self._prep_q)

    @torch.no_grad()
    def resymmetrise(self) -> None:
        """Project every tied network onto the equivariant subspace."""
        self._tie_value()
        self._tie_q(self.qf)
        self._tie_q(self.q_target)
        if self.tie_policy:
            symmetrize_linear_stack(self.policy.net, self.spec.obs_perm,
                                    self.spec.obs_sign, self.spec.act_perm,
                                    self.spec.act_sign, self._prep_pi)
        self.n_ties += 1

    def _post_optimizer_step(self, which: str) -> None:
        """Called by SupportAwareIQL.update after each optimiser step."""
        if which == "value":
            self._tie_value()
        elif which == "critic":
            self._tie_q(self.qf)
        elif which == "target":
            # The EMA writes into q_target in place, so the tie must be
            # restored afterwards or the target critic drifts off the
            # equivariant subspace over training.
            self._tie_q(self.q_target)
        elif which == "policy" and self.tie_policy:
            symmetrize_linear_stack(self.policy.net, self.spec.obs_perm,
                                    self.spec.obs_sign, self.spec.act_perm,
                                    self.spec.act_sign, self._prep_pi)
        self.n_ties += 1

    # ------------------------------------------------------------- monitoring
    @torch.no_grad()
    def equivariance_error(self, obs: torch.Tensor,
                           act: torch.Tensor) -> dict:
        """Relative error |f(g.x) - g.f(x)| / (1 + |f(x)|) for V, Q and pi.

        Measured on whatever observation space the network actually sees, i.e.
        the NORMALISED one. The mirror is therefore exact for the
        normaliser-induced group; how far the normaliser itself breaks the
        raw-state mirror is a separate, measurable quantity that
        experiments/run_m1.py reports.
        """
        gs = apply_mirror_obs(self.spec, obs)
        ga = apply_mirror_act(self.spec, act)

        def rel(gx: torch.Tensor, fx: torch.Tensor) -> float:
            return float((gx - fx).abs().mean() / (1.0 + fx.abs().mean()))

        v = self.vf(obs).squeeze(-1)
        eq_v = rel(self.vf(gs).squeeze(-1), v)
        q = self.qf.q1(torch.cat([obs, act], -1)).squeeze(-1)
        eq_q = rel(self.qf.q1(torch.cat([gs, ga], -1)).squeeze(-1), q)
        mean = self.policy.net(obs)
        eq_pi = rel(apply_mirror_act(self.spec, self.policy.net(gs)),
                    apply_mirror_act(self.spec, mean))
        return {"eq_V": eq_v, "eq_Q": eq_q, "eq_pi": eq_pi}

    def mirror_summary(self) -> dict:
        """How much structure this domain's mirror actually has."""
        obs_dim = len(self.spec.obs_perm)
        act_dim = len(self.spec.act_perm)
        return {
            "domain": self.domain,
            "obs_dims_moved": int((self.spec.obs_perm != np.arange(obs_dim)).sum()),
            "obs_dims_total": obs_dim,
            "act_dims_moved": int((self.spec.act_perm != np.arange(act_dim)).sum()),
            "act_dims_total": act_dim,
            "obs_dims_flipped": int((self.spec.obs_sign == -1).sum()),
            "tie_policy": self.tie_policy,
            "n_ties": self.n_ties,
        }
