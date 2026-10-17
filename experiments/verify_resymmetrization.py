"""Verify choice A: re-symmetrise the weights after EVERY optimiser step.

Why this needs verifying rather than assuming:

  Weight tying makes f equivariant ONLY if the weights are tied. Adam updates
  the free parameters and breaks the tie immediately, so re-symmetrising
  after each step is mandatory -- otherwise equivariance lasts one step.

  The risk is that Adam's moment estimates (exp_avg / exp_avg_sq) become
  inconsistent with the averaged weights: the optimiser carries momentum
  about a point that no longer exists. That can bias the fixed point or make
  the value diverge.

So measure two things at once on walker2d (the only domain with a non-trivial
mirror among ours):

  * equivariance error   eq = |V(g.s) - V(s)| / (1 + |V(s)|)
  * value trajectory    is V still climbing toward the fixed point?

    python experiments/verify_resymmetrization.py --steps 20000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.methods.blocks import concat_blocks  # noqa: E402
from orl.methods.equivariance import (  # noqa: E402
    apply_mirror_act,
    apply_mirror_obs,
    derive_mirror,
    symmetrize_linear_stack,
)
from orl.methods.harness import load  # noqa: E402
def tie_value(net, spec) -> None:
    symmetrize_linear_stack(net.v, spec.obs_perm, spec.obs_sign,
                            np.arange(1), np.ones(1))


def tie_q(net, spec) -> None:
    # Block-relative -> global, in one place (see orl.methods.blocks for why
    # this offset has been gotten wrong twice already).
    sa_perm = concat_blocks(spec.obs_perm, spec.act_perm)
    sa_sign = np.concatenate([spec.obs_sign, spec.act_sign])
    for q in (net.q1, net.q2):
        symmetrize_linear_stack(q, sa_perm, sa_sign, np.arange(1), np.ones(1))


def _eq_error(fn, spec, obs, act=None) -> float:
    """Relative equivariance error of `fn` under the mirror."""
    with torch.no_grad():
        s = obs
        a = act
        lhs = fn(apply_mirror_obs(spec, s)) if a is None else \
            fn(apply_mirror_obs(spec, s), apply_mirror_act(spec, a))
        rhs = fn(s) if a is None else fn(s, a)
        ref = 1.0 + rhs.abs().mean()
        return float((lhs - rhs).abs().mean() / ref)


def run(env_id, symmetrise, steps, every, lr, gamma, expectile, seed):
    import torch.nn.functional as F

    from src.iql import asymmetric_l2_loss

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    d = load(env_id, with_coverage=False, normalize=True)
    obs_dim, act_dim = d.obs.shape[1], d.act.shape[1]
    spec = derive_mirror(d.domain, obs_dim=obs_dim, act_dim=act_dim)
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32,
                                 device=dev)  # noqa: E731
    O, A, R = T(d.obs), T(d.act), T(d.rew)
    NO, TE = T(d.next_obs), T(d.terminals)

    qf = TwinQ(obs_dim, act_dim).to(dev)
    vf = ValueFunction(obs_dim).to(dev)
    q_targ = TwinQ(obs_dim, act_dim).to(dev)
    q_targ.load_state_dict(qf.state_dict())
    for p in q_targ.parameters():
        p.requires_grad_(False)
    if symmetrise:
        tie_value(vf, spec)
        tie_q(qf, spec)
        tie_q(q_targ, spec)
    opt_q = torch.optim.Adam(qf.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vf.parameters(), lr=lr)

    probe = np.sort(np.random.default_rng(7).choice(len(O), 2048, replace=False))
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, steps + 1):
        j = rng.integers(0, len(O), 256)
        oj, aj, rj, nj, tj = O[j], A[j], R[j], NO[j], TE[j]
        with torch.no_grad():
            next_v = vf(nj)
        tgt = rj + (1.0 - tj) * gamma * next_v.detach()
        qloss = sum(F.mse_loss(q, tgt) for q in qf.both(oj, aj)) / 2
        opt_q.zero_grad(set_to_none=True)
        qloss.backward()
        opt_q.step()
        if symmetrise:
            tie_q(qf, spec)
        with torch.no_grad():
            q_min = torch.min(q_targ(oj, aj), q_targ(oj, aj))
        adv = q_min - vf(oj)
        vloss = asymmetric_l2_loss(adv, expectile)
        opt_v.zero_grad(set_to_none=True)
        vloss.backward()
        opt_v.step()
        if symmetrise:
            tie_value(vf, spec)
        with torch.no_grad():
            for pt, ps in zip(q_targ.parameters(), qf.parameters()):
                pt.mul_(0.995).add_(ps, alpha=0.005)
            if symmetrise:
                tie_q(q_targ, spec)

        if i % every == 0 or i == steps:
            with torch.no_grad():
                vmean = float(vf(O[probe]).mean())
            rows.append({
                "step": i,
                "V": vmean,
                "eq_V": _eq_error(lambda s: vf(s).squeeze(-1), spec, O[probe]),
                "eq_Q": _eq_error(lambda s, a: qf.q1(torch.cat([s, a], -1)).squeeze(-1),
                                  spec, O[probe], A[probe]),
                "q_loss": float(qloss),
            })
    return rows
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="walker2d_medium_replay-v2")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--every", type=int, default=2_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"env={a.env}  steps={a.steps:,}  lr={a.lr}")
    d = load(a.env, with_coverage=False, normalize=True)
    print(f"reward scaled mean={d.scaled_rew_mean:.4f}  "
          f"=> fixed point ~{d.scaled_rew_mean / (1 - a.gamma):.1f}")
    spec = derive_mirror(d.domain, obs_dim=d.obs.shape[1], act_dim=d.act.shape[1])
    moved = int((spec.act_perm != np.arange(len(spec.act_perm))).sum())
    print(f"mirror moves {moved}/{len(spec.act_perm)} actions\n")
    out = {}
    for tag, sym in (("OFF", False), ("ON", True)):
        t0 = time.perf_counter()
        rows = run(a.env, sym, a.steps, a.every, a.lr, a.gamma, a.expectile, a.seed)
        out[tag] = rows
        print(f"--- re-symmetrise {tag} ({time.perf_counter() - t0:.0f}s) ---")
        print(f"{'step':>8}{'V':>10}{'eq_V':>12}{'eq_Q':>12}")
        for r in rows:
            print(f"{r['step']:>8,}{r['V']:>10.2f}{r['eq_V']:>12.2e}{r['eq_Q']:>12.2e}", flush=True)
        print()
    print("=" * 62)
    on, off = out["ON"], out["OFF"]
    print(f"final V    : OFF {off[-1]['V']:.2f}   ON {on[-1]['V']:.2f}")
    print(f"final eq_V : OFF {off[-1]['eq_V']:.2e}   ON {on[-1]['eq_V']:.2e}")
    print(f"final eq_Q : OFF {off[-1]['eq_Q']:.2e}   ON {on[-1]['eq_Q']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
