"""T7 -- M1: symmetry-equivariant pessimism, on the vendored reference IQL.

Same data pipeline, same reward rescaling and same optimiser as the verified
reference run (docs/BASELINE_STATUS.md). Four modes, because M1 needs TWO
changes and attributing both to "equivariance" would be wrong:

    --mode ref      plain normaliser, no tying, verbatim reference update
    --mode ref_sym  mirror-symmetrised normaliser, no tying  <- matched control
    --mode m1       + weight tying after every optimiser step
    --mode m1_m2    + coverage-driven n, i.e. M1 composed with M2

Why the extra control: tying makes the network equivariant under the mirror of
the space it READS, and the network reads per-dimension normalised
observations. Measured on walker2d_medium_replay-v2 those statistics are not
mirror-symmetric (rooty drifts 0.74 std), so plain standardisation enforces a
contaminated symmetry. Fixing that needs orbit-averaged statistics, which is
itself a change -- hence ref_sym, without which "M1's effect" would be the sum
of two interventions. `--sym-normer` overrides the default and exposes the
fourth cell (tying under plain standardisation).

walker2d is the default because it is the only one of our three domains whose
mirror is not trivial (8/17 obs dims moved). On halfcheetah and hopper the
mechanism is close to a no-op, which is itself a result.

The script measures the four things GATE 2 needs, and refuses to guess them:

  1. equivariance error of V, Q and the actor mean (mechanism actually on?)
  2. the value trajectory (does tying damage learning?)
  3. policy score (the dependent variable)
  4. wall-clock per step (the cost of tying, needed for the budget)

plus one honesty diagnostic: observations are normalised per dimension, and
the per-dimension mirror only commutes with that normaliser if the dataset's
mean and std are themselves mirror-symmetric. How far they are from that is
reported, because it bounds the claim we are allowed to make.

    python experiments/run_m1.py --smoke
    python experiments/run_m1.py --mode ref --steps 100000
    python experiments/run_m1.py --mode m1  --steps 100000
    python experiments/run_m1.py --mode m1_m2 --steps 100000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(ROOT))

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.data.normalize import ObsNormalizer  # noqa: E402
from orl.methods.equivariance import (  # noqa: E402
    derive_mirror,
    normaliser_commutes,
)
from orl.methods.equivariant_iql import EquivariantIQL  # noqa: E402
from orl.methods.harness import evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR, parse_env_id  # noqa: E402
from orl.record.logger import RunGuard  # noqa: E402


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def normaliser_mirror_mismatch(normer, spec) -> dict:
    """How badly per-dim normalisation breaks the mirror, in std units.

    With n(s)_i = (s_i - mu_i)/sd_i and g(s)_i = sign_i * s_{perm(i)}, the two
    commute iff sd_i == sd_{perm(i)} AND mu_i == sign_i * mu_{perm(i)}.
    Neither is guaranteed: the statistics come from a behaviour policy, not
    from the symmetry group. Reported as a magnitude, not a pass/fail.
    """
    mu = np.asarray(normer.mean, dtype=np.float64)
    sd = np.asarray(normer.std, dtype=np.float64)
    sd = np.where(sd < 1e-12, 1.0, sd)
    perm, sign = spec.obs_perm, spec.obs_sign
    scale = np.abs(sd - sd[perm]) / sd
    offset = np.abs(mu - sign * mu[perm]) / sd
    return {
        "max_scale_asymmetry": float(scale.max()),
        "mean_scale_asymmetry": float(scale.mean()),
        "max_offset_asymmetry": float(offset.max()),
        "mean_offset_asymmetry": float(offset.mean()),
    }


def build(args, d, ep):
    """Select the agent class. This is the whole experimental contrast."""
    common = dict(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        max_steps=args.steps, tau=args.tau, beta=args.beta,
        alpha=args.alpha, discount=args.discount, episode_lookup=ep,
    )
    if args.mode == "ref":
        # fixed_n is required by the class but unused: this mode passes
        # indices=None, so the update takes the reference's own 1-step target.
        return SupportAwareIQL(coverage=None, fixed_n=1,
                               n_min=1, n_max=args.n_max, **common)
    if args.mode == "ref_sym":
        # Identical to `ref` except for the normalisation statistics, so
        # (m1 - ref_sym) isolates the tying rather than the reparametrisation.
        return SupportAwareIQL(coverage=None, fixed_n=1,
                               n_min=1, n_max=args.n_max, **common)
    if args.mode == "m1":
        return EquivariantIQL(coverage=None, fixed_n=args.fixed_n,
                              n_min=1, n_max=args.n_max,
                              domain=d.domain, obs_dim=d.obs.shape[1],
                              act_dim=d.act.shape[1],
                              tie_policy=args.tie_policy, **common)
    return EquivariantIQL(coverage=d.coverage, n_min=args.n_min,
                          n_max=args.n_max, domain=d.domain,
                          obs_dim=d.obs.shape[1], act_dim=d.act.shape[1],
                          tie_policy=args.tie_policy, **common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="walker2d_medium_replay-v2")
    ap.add_argument("--mode", choices=("ref", "ref_sym", "m1", "m1_m2"),
                    default="m1")
    ap.add_argument("--sym-normer", choices=("default", "none", "mirror"),
                    default="default",
                    help="mirror-symmetrise the normalisation statistics")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--every", type=int, default=5_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7, help="expectile")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--fixed-n", type=int, default=1)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--tie-policy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.every, args.eval_episodes = 2_000, 500, 2
    tag = args.tag or f"m1_{args.mode}_{args.env.split('_')[0]}"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    # The mirror must be known BEFORE loading, because whether the
    # normalisation carries it is part of the intervention under test.
    with D4RLDataset(args.env, mode="stream") as ds:
        domain, obs_dim, act_dim = ds.domain, ds.obs_dim, ds.act_dim
        ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)
    spec = derive_mirror(domain, obs_dim=obs_dim, act_dim=act_dim)
    symmetrise = (args.mode != "ref") if args.sym_normer == "default" \
        else (args.sym_normer == "mirror")

    d = load(args.env, with_coverage=(args.mode == "m1_m2"), normalize=True,
             mirror_spec=spec if symmetrise else None)
    print(f"mode={args.mode}  env={d.env_id}  device={dev}")
    print(f"normaliser    = {'mirror-symmetrised' if symmetrise else 'plain per-dim'}")
    print(f"returns range  = [{d.ret_range[0]:.1f}, {d.ret_range[1]:.1f}]")
    print(f"reward         = raw {d.raw_rew_mean:.4f} -> scaled {d.scaled_rew_mean:.4f}")
    print(f"fixed point    = {d.scaled_rew_mean / (1 - args.discount):.1f} (scaled)")

    agent = build(args, d, ep)

    # Honest diagnostic: the share of the dataset sitting on an episode-final
    # transition, i.e. where the n-step target and the reference's 1-step
    # target necessarily differ.
    rem = ep.steps_remaining(np.arange(ep.episode_of.shape[0], dtype=np.int64))
    boundary_frac = float((rem == 1).mean())
    print(f"episode-final transitions = {boundary_frac * 100:.2f}% of the data")

    # The dataset's own asymmetry is a property of the behaviour policy and is
    # identical in every mode; what changes is whether the normaliser inherits
    # it. Both are reported because they are different claims.
    raw = ObsNormalizer(mean=d.raw_obs_mean, std=d.raw_obs_std)
    data_asym = normaliser_mirror_mismatch(raw, spec)
    comm_used = normaliser_commutes(d.normer.mean, d.normer.std, spec)
    print(f"mirror moves {int((spec.act_perm != np.arange(len(spec.act_perm))).sum())}"
          f"/{len(spec.act_perm)} actions, "
          f"{int((spec.obs_perm != np.arange(len(spec.obs_perm))).sum())}"
          f"/{len(spec.obs_perm)} obs dims")
    print(f"data asymmetry   : scale {data_asym['max_scale_asymmetry']:.3f}  "
          f"offset {data_asym['max_offset_asymmetry']:.3f} (std units, "
          f"same in all modes)")
    print(f"n(g.s)-g(n(s))   : {comm_used['max_abs_residual_std']:.2e} max "
          f"(normaliser actually used)")
    if args.mode not in ("ref", "ref_sym"):
        print(f"mirror summary  = {agent.mirror_summary()}")
        print(f"n-step policy   = {agent.requested_n_summary()}")

    O, A, R = to_t(d.obs, dev), to_t(d.act, dev), to_t(d.rew, dev)
    NO, TE = to_t(d.next_obs, dev), to_t(d.terminals, dev)

    probe = np.sort(np.random.default_rng(7).choice(len(O), 4096, replace=False))
    probe_o, probe_a = O[probe], A[probe]
    rng = np.random.default_rng(args.seed)
    rows = []
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    # One file per run, keyed like the JSON record. Appending to a shared name
    # mixed rows from different step counts and seeds together, which would
    # silently corrupt any later analysis of the curve.
    curve_path = BENCH_DIR / f"{tag}_{d.env_id}_s{args.seed}_{args.steps}_curve.jsonl"
    curve_path.write_text("", encoding="utf-8")

    print(f"\n{'step':>9}{'V':>9}{'Q':>9}{'q_loss':>11}{'eq_V':>11}"
          f"{'eq_Q':>11}{'eq_pi':>11}{'ms/step':>9}")
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        j = np.sort(rng.integers(0, len(O), args.batch_size))
        kwargs = {}
        if args.mode in ("m1", "m1_m2"):
            # M1 and M2 both go through the n-step target, so GATE 2's cells
            # differ in the tying and in n only -- never in which code ran.
            kwargs = dict(indices=j, dataset_rewards=d.rew,
                          dataset_next_obs=d.next_obs,
                          dataset_terminals=d.terminals)
        out = agent.update(O[j], A[j], NO[j], R[j], TE[j], **kwargs)

        if i % args.every == 0 or i == args.steps:
            with torch.no_grad():
                V = agent.vf(O[probe]).squeeze(-1)
                Q = agent.q_target(probe_o, probe_a).squeeze(-1)
            if args.mode in ("ref", "ref_sym"):
                nan = float("nan")
                eq = {"eq_V": nan, "eq_Q": nan, "eq_pi": nan}
            else:
                eq = agent.equivariance_error(probe_o, probe_a)
            ms = (time.perf_counter() - t0) * 1000.0 / i
            row = {"step": i, "V": float(V.mean()), "Q": float(Q.mean()),
                   "q_loss": out["q_loss"], "v_loss": out["value_loss"],
                   "actor_loss": out["actor_loss"], "ms_per_step": ms, **eq}
            rows.append(row)
            with curve_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{i:>9,}{row['V']:>9.2f}{row['Q']:>9.2f}"
                  f"{row['q_loss']:>11.3f}{row['eq_V']:>11.2e}"
                  f"{row['eq_Q']:>11.2e}{row['eq_pi']:>11.2e}{ms:>9.2f}",
                  flush=True)
    train_min = (time.perf_counter() - t0) / 60.0

    ev = evaluate(agent.policy, d.domain, d.normer, n_episodes=args.eval_episodes)
    print(f"\ntrain {train_min:.1f} min")
    print(f"policy raw={ev['raw_mean']:.1f} +- {ev['raw_std']:.1f}  "
          f"normalized={ev['score_mean']:.1f}")

    domain, tier = parse_env_id(d.env_id)
    with RunGuard(run_id=f"{tag}_s{args.seed}", method=f"iql+{args.mode}",
                  dataset=d.env_id, domain=domain, tier=tier, seed=args.seed,
                  stage="S1", score_norm=ev["score_mean"],
                  score_raw=ev["raw_mean"],
                  config=json.dumps(vars(args), sort_keys=True),
                  notes=f"n_ties={getattr(agent, 'n_ties', 0)}"):
        rec = {
            "env": d.env_id, "mode": args.mode, "seed": args.seed,
            "steps": args.steps, "args": vars(args), "eval": ev,
            "curve": rows, "train_min": train_min,
            "ms_per_step": rows[-1]["ms_per_step"],
            "boundary_frac": boundary_frac,
            "normaliser_symmetrised": symmetrise,
            "data_asymmetry": data_asym,
            "normaliser_commutation": comm_used,
            "mirror": (agent.mirror_summary()
                       if args.mode in ("m1", "m1_m2") else {"domain": d.domain}),
            "n_policy": (agent.requested_n_summary()
                         if args.mode in ("m1", "m1_m2") else None),
            "ret_range": list(d.ret_range),
        }
        out = BENCH_DIR / f"{tag}_{d.env_id}_s{args.seed}_{args.steps}.json"
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"wrote {out.name} and {curve_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
