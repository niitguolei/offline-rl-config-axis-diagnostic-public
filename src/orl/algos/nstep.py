"""T2 — n-step Bellman targets. LOGIC IS SOUND, BASE IS NOT.

The episode-boundary clipping below was verified by unit tests and is the
piece worth keeping. But `NStepIQL` subclasses our own withdrawn IQL
(see docs/BASELINE_STATUS.md), and the reference scales rewards, so any
number produced through this class is void.

To use it: port `nstep_target` onto the vendored reference
(vendor/IQL-PyTorch/src/iql.py) and run M2 from there.

---

T2 — n-step Bellman targets for IQL.

The subtle part is episode boundaries. A naive `i, i+1, ..., i+n-1` window
silently walks past the end of a trajectory into the next episode, mixing in
rewards from an unrelated state. D4RL episodes are a flat concatenation with
no index, so the boundary must be recovered (see orl.data.episode).

Rules implemented here:
  * the n-step window NEVER crosses an episode end;
  * the effective length `n_i` is shortened at the boundary;
  * bootstrapping with `V(s_{i+n_i})` happens ONLY while that state is still
    inside the same episode. At a trajectory end we do not bootstrap: the
    "next" array entry there belongs to a different episode, so using it would
    be wrong and would quietly contaminate every target.

The target is therefore a mixed n-step / episodic-return target. That is the
only well-defined choice on this data.
"""

from __future__ import annotations

import numpy as np
import torch

from .iql import IQL


from ..data.episode import EpisodeIndexLookup  # noqa: E402,F401  (re-export)


class NStepIQL(IQL):
    """IQL whose critic target uses a per-sample n-step return."""

    def __init__(self, cfg, episode_lookup: EpisodeIndexLookup, n_step: int = 5) -> None:
        super().__init__(cfg)
        self.ep = episode_lookup
        self.n_step = int(n_step)

    def nstep_target(
        self,
        idx: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
        device: str,
    ) -> torch.Tensor:
        """Discounted n-step return, clipped at the episode boundary."""
        g = self.cfg.gamma
        rem = self.ep.steps_remaining(idx)
        m = np.minimum(self.n_step, rem)              # effective length
        last = len(rewards) - 1

        rew = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        nobs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        target = torch.zeros(len(idx), device=device)
        for k in range(self.n_step):
            r_at = np.minimum(idx + k, last)
            active = torch.as_tensor((m > k).astype(np.float32), device=device)
            r_k = rew[torch.as_tensor(r_at, device=device)]
            target = target + active * (g**k) * r_k

        # Bootstrap only if the state after the window is still in-episode.
        inside = (m < rem).astype(np.float32)
        if float(inside.sum()) > 0:
            b_at = np.minimum(idx + m - 1, last)
            b_no = nobs[torch.as_tensor(b_at, device=device)]
            ended = torch.as_tensor(
                ((terminals[b_at] > 0) | (timeouts[b_at] > 0)).astype(np.float32),
                device=device,
            )
            m_t = torch.as_tensor(m.astype(np.float32), device=device)
            coef = torch.as_tensor(inside, device=device) * (g**m_t) * (1.0 - ended)
            target = target + coef * self.v(b_no).squeeze(-1)

        return target

    def update(self, batch: dict) -> dict[str, float]:
        """Full IQL update whose critic target is the n-step return.

        The parent `update()` calls the 1-step `compute_target`, so without this
        override the n-step machinery would be dead code. All three losses are
        kept: V and the actor do not depend on n, but if they were dropped the
        policy would never move and the MC reference would be meaningless.
        """
        import torch.nn.functional as F

        cfg = self.cfg
        dev = cfg.device
        idx = np.asarray(batch["indices"])
        o = torch.as_tensor(batch["observations"], dtype=torch.float32, device=dev)
        a = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)

        def flat(key: str) -> np.ndarray:
            """D4RL stores rewards/terminals/timeouts as 1-D, but callers may
            pass (n, 1). Accept both."""
            return np.asarray(batch[key]).reshape(len(idx), -1)[:, 0]

        target = self.nstep_target(
            idx,
            flat("rewards"),
            np.asarray(batch["next_observations"]),
            flat("terminals"),
            flat("timeouts"),
            dev,
        )
        q1p = self.q1(self._sa(o, a)).squeeze(-1)
        q2p = self.q2(self._sa(o, a)).squeeze(-1)
        q_loss = F.mse_loss(q1p, target) + F.mse_loss(q2p, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                cfg.clip_grad_norm,
            )
        self.opt_q.step()

        with torch.no_grad():
            q_min = self._min_q_targ(o, a).squeeze(-1)
        v_pred = self.v(o).squeeze(-1)
        diff = q_min - v_pred
        w = torch.abs(cfg.expectile - (diff < 0).float())
        v_loss = (w * diff.pow(2)).mean()
        self.opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), cfg.clip_grad_norm)
        self.opt_v.step()

        with torch.no_grad():
            adv = q_min - self.v(o).squeeze(-1)
            awr = torch.clamp(torch.exp(cfg.beta * adv), max=cfg.max_weight)
        mu, log_std = self.pi(o)
        std = log_std.exp()
        logp = (-0.5 * ((a - mu) / std).pow(2) - log_std).sum(-1)
        pi_loss = -(awr * logp).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.clip_grad_norm)
        self.opt_pi.step()

        with torch.no_grad():
            for net, targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p, tp in zip(net.parameters(), targ.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)

        out = {"q_loss": float(q_loss.detach()),
               "v_loss": float(v_loss.detach()),
               "pi_loss": float(pi_loss.detach()),
               "q_mean": float(q1p.detach().mean())}
        self._last = out
        return out
