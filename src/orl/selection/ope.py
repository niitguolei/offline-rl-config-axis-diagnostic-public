"""Offline policy evaluation estimators, for path A's model-selection chain.

Path A asks where OPE-based selection fails, so the estimators need to be
standard and cheap enough to run at every checkpoint. Two are implemented here
to start:

  * `fitted_q_evaluation` -- FQE. Fits Q^pi on the offline transitions with the
    *evaluation policy's* action at s', then averages Q over episode-start
    states. This is the workhorse OPE baseline in the offline RL literature.
  * `critic_mean_value` -- the trained critic's own V averaged over start
    states. Almost free, and a useful naive reference: method-selection papers
    frequently show it is competitive, which is itself part of the story.

Two conventions carried over from the code the rest of the project already
verified, because getting them wrong is how this project has silently broken
before:

  1. The policy is evaluated at its DETERMINISTIC MEAN action, matching
     `harness.evaluate`. FQE must estimate the value of the same object the
     true-return measurement scores, or the oracle gap is meaningless.
  2. Observations arrive NORMALISED (the same normaliser the policy was trained
     with). Nothing here re-normalises, so a caller cannot double-apply it.

Note on importance sampling: PDIS-style estimators need the behaviour policy's
action probabilities, which D4RL does not store, so they require fitting a
behaviour model first. That is deliberately left for a later step rather than
approximated here -- an estimator whose behaviour model is silently wrong would
corrupt exactly the failure map path A is trying to build.

Budget fairness (measured, not assumed): in the T0 smoke, FQE at 300 gradient
steps returned an almost constant estimate (it had not converged) while the
near-free critic mean tracked the truth perfectly. That comparison is not
evidence about the estimators, it is evidence that the comparison must give each
estimator a budget at which IT has converged. T2 has to establish those budgets
before any "estimator A beats B" claim, or the failure map would be measuring
optimisation schedules instead of estimator quality.

The trap is that FQE's LOSS does not signal convergence. Measured at a 4000-step
budget the loss sat near 0.1 while the returned value was ~-0.2 against a truth
of ~42: the target network is still climbing, so the regression tracks a moving
target and stays locally accurate while remaining globally wrong. Hence
`value_trace` below -- convergence has to be judged on the estimate stabilising,
not on the loss falling.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def policy_action(policy, obs: torch.Tensor, low: float = -1.0,
                  high: float = 1.0, clip: bool = True) -> torch.Tensor:
    """Deterministic mean action, CLIPPED to the environment's action space.

    The clip is not a detail. Measured on halfcheetah after 20k IQL steps, the
    policy's raw mean spans [-1.305, 1.195] while the action space is [-1, 1].
    `harness.evaluate` clips before stepping -- it has to, the environment
    rejects out-of-range actions -- so an estimator that queries the critic at
    the UNCLIPPED mean action is scoring a different policy than the true-return
    measurement scores. Worse, that is exactly where a learned critic is
    extrapolating, so the bootstrap compounds the error: measured FQE values
    went 11.9 -> 1.1 and oscillated instead of converging, against a ground truth
    of 64.9.

    Keeping the convention in one place is the point: `harness.evaluate`, FQE
    and any future estimator must agree on what "the policy" is.
    """
    out = policy(obs)
    # NOTE the discriminator: a plain Tensor also has a `.mean` METHOD, so
    # `hasattr(out, "mean")` is true for tensors too and would silently hand back
    # an unbound method to `.clamp`. Any policy that returns a tensor (BC's
    # closure does) would break. Test the tensor case first.
    act = out if torch.is_tensor(out) else out.mean
    return act.clamp(low, high) if clip else act


def critic_mean_value(vf, start_states: torch.Tensor) -> float:
    """Mean of the agent's own V over episode-start states. Nearly free."""
    with torch.no_grad():
        return float(vf(start_states).mean())


def fitted_q_evaluation(
    policy,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    start_states: torch.Tensor,
    *,
    gamma: float = 0.99,
    steps: int = 4000,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    seed: int = 0,
    device=None,
    target_agg: str = "mean",
    loss: str = "huber",
    huber_beta: float = 10.0,
    grad_clip: float | None = 10.0,
    alpha: float = 0.05,
    hard_every: int | None = None,
) -> dict:
    """Fit Q^pi and return its mean over `start_states`.

    The defaults were changed after measuring against an MC-anchored truth; see
    the module docstring and docs/T0报告.md section 6. The first version used
    MSE, a min-of-two-heads target and an EMA with alpha=0.005, and it did not
    reach the truth (4.24 against 64.905 after 60k steps) while the faster EMA
    oscillated (10.6 at 8k falling to 1.8 at 20k).

    Design choices, each with a reason rather than a habit:

      target_agg="mean"   FQE estimates the value of a FIXED policy, so the
                          conservative min-of-two-heads is the wrong estimator --
                          it biases downward. (The previous docstring claimed
                          mean while the code did min; the docstring was right
                          about the intent and the code was wrong.)
      loss="huber"        bootstrapped targets move, so a few batches carry huge
                          errors and MSE lets them dominate. Huber with beta=10
                          is quadratic for the errors we cannot fix slowly and
                          linear for the outliers that cause the blow-up.
      grad_clip=10        bounds a single bad batch's step.
      alpha=0.05          the EMA time constant sets the propagation rate:
                          with alpha=0.005 the target lags 200 gradient steps, so
                          Q-iteration needs ~230 x 200 = 46k steps just to
                          propagate the fixed point -- which is exactly why the
                          slow run was still at 4.24 after 60k.
      hard_every=None     optional DQN-style hard target copy, for comparison.
    """
    from src.util import mlp
    from src.value_functions import TwinQ

    dev = device or next(policy.parameters()).device
    torch.manual_seed(seed)
    n = observations.shape[0]
    qf = TwinQ(observations.shape[1], policy_action(
        policy, observations[:1]).shape[1],
        hidden_dim=hidden_dim, n_hidden=n_hidden).to(dev)
    q_target = copy.deepcopy(qf).requires_grad_(False)
    opt = torch.optim.Adam(qf.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    losses = []
    value_trace = []
    probe = start_states[: min(64, start_states.shape[0])]
    for i in range(steps):
        j = torch.as_tensor(rng.integers(0, n, batch_size), device=dev)
        o, no, r, t = observations[j], next_observations[j], rewards[j], terminals[j]
        with torch.no_grad():
            a_next = policy_action(policy, no)
            if target_agg == "mean":
                q1, q2 = q_target.both(no, a_next)
                base = 0.5 * (q1 + q2)
            elif target_agg == "min":
                base = q_target(no, a_next)
            else:
                raise ValueError(f"unknown target_agg {target_agg!r}")
            tgt = r + (1.0 - t) * gamma * base
        a = policy_action(policy, o)
        qs = qf.both(o, a)
        if loss == "huber":
            per = [F.smooth_l1_loss(q, tgt, beta=huber_beta) for q in qs]
        elif loss == "mse":
            per = [F.mse_loss(q, tgt) for q in qs]
        else:
            raise ValueError(f"unknown loss {loss!r}")
        total_loss = sum(per) / len(per)
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(qf.parameters(), grad_clip)
        opt.step()
        with torch.no_grad():
            if hard_every is not None and (i + 1) % hard_every == 0:
                q_target.load_state_dict(qf.state_dict())
            else:
                for pt, ps in zip(q_target.parameters(), qf.parameters()):
                    pt.mul_(1.0 - alpha).add_(ps, alpha=alpha)
        if i % max(1, steps // 4) == 0:
            losses.append(float(total_loss.detach()))
        if i % max(1, steps // 10) == 0:
            with torch.no_grad():
                value_trace.append((i, float(qf(probe, policy_action(policy, probe)).mean())))

    with torch.no_grad():
        a_star = policy_action(policy, start_states)
        value = float(qf(start_states, a_star).mean())
        # Report the fit's own residual so a caller can tell "FQE disagrees"
        # apart from "FQE never converged".
        spread = float(qf(start_states, a_star).std())
    return {"value": value, "value_std": spread, "q_loss_trace": losses,
            "value_trace": value_trace, "steps": steps}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, so a monotone but mis-scaled estimator is not punished."""

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            for k, c in enumerate(counts):
                if c > 1:
                    r[inv == k] = r[inv == k].mean()
        return r

    ra, rb = rank(np.asarray(a, dtype=np.float64)), rank(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
