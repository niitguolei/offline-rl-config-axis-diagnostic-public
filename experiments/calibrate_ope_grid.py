"""Pick FQE's configuration against ground truth, not by taste.

The first FQE implementation (MSE loss, min-of-two-heads target, EMA alpha=0.005)
never reached the truth: 4.24 against an MC-anchored 64.905 after 60k steps.
Switching to a faster EMA (alpha=0.05) climbed to 10.6 by 8k steps and then fell
back to 1.8 by 20k -- error amplification through the bootstrap rather than slow
convergence.

Reasoning gives a direction but not a winner, so this runs the candidates against
the same MC truth and reports the error of each:

  * `alpha` sets the propagation RATE: the Q-iteration fixed point needs about
    ln(0.1)/ln(gamma) ~ 230 effective target updates to be reached, and one
    effective update costs 1/alpha gradient steps. alpha=0.005 therefore needs
    ~46k steps just to propagate, which is why the slow run was still at 4.24.
  * Huber rather than MSE: bootstrapped targets move, so a few batches carry
    huge errors and MSE lets them dominate the update.
  * mean rather than min over the two heads: FQE evaluates a FIXED policy, so
    the conservative minimum is simply a downward bias.
  * gradient clipping and an optional hard target copy bound a single bad batch.

    python experiments/calibrate_ope_grid.py --smoke
    python experiments/calibrate_ope_grid.py --steps 20000
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

from orl.algos.mc_reference import mc_reference_batched, reference_callables  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.methods.harness import episode_lookup, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.ope import fitted_q_evaluation  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

CONFIGS = [
    ("old mse/min/ema.005", dict(loss="mse", target_agg="min", alpha=0.005,
                                 grad_clip=None, lr=3e-4)),
    ("huber/mean/ema.05", dict(loss="huber", target_agg="mean", alpha=0.05,
                              grad_clip=10.0, lr=1e-3)),
    ("huber/mean/ema.02", dict(loss="huber", target_agg="mean", alpha=0.02,
                              grad_clip=10.0, lr=1e-3)),
    ("huber/mean/hard2000", dict(loss="huber", target_agg="mean",
                                 hard_every=2000, grad_clip=10.0, lr=1e-3)),
]


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--train-steps", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=20_000, help="FQE steps")
    ap.add_argument("--mc-episodes", type=int, default=64)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_steps, args.steps = 2000, 1500
        args.mc_episodes, args.mc_horizon = 8, 100

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep = episode_lookup(d.env_id)
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    with D4RLDataset(d.env_id, mode="stream") as ds:
        raw_obs = ds.get("observations")
    raw_starts = raw_obs[starts][: args.mc_episodes].astype(np.float64)

    O, A = to_t(d.obs, dev), to_t(d.act, dev)
    R, NO, TE = to_t(d.rew, dev), to_t(d.next_obs, dev), to_t(d.terminals, dev)
    ss = O[torch.as_tensor(starts[: args.mc_episodes], device=dev).long()]

    torch.manual_seed(args.seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.train_steps, tau=0.7, beta=3.0, alpha=0.005,
        discount=0.99, episode_lookup=ep,
        coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(args.seed)
    for _ in range(args.train_steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(O[j], A[j], NO[j], R[j], TE[j])

    pol_fn, _ = reference_callables(agent, d.normer, device=dev)
    mc = mc_reference_batched(d.domain, raw_starts, pol_fn, gamma=0.99,
                              horizon=args.mc_horizon, device=dev, batch_size=32)
    truth = float((d.reward_scale * mc.values).mean())
    sem = float(d.reward_scale * mc.values.std() / np.sqrt(len(mc.values)))
    print(f"MC truth = {truth:.3f} +- {sem:.3f}  (n={len(raw_starts)})")
    print(f"FQE budget = {args.steps:,} steps per configuration\n")

    rows = []
    for name, cfg in CONFIGS:
        res = fitted_q_evaluation(agent.policy, O, NO, R, TE, ss, gamma=0.99,
                                  steps=args.steps, seed=args.seed,
                                  device=dev, **cfg)
        # Denominator must not be a near-zero truth: a weak policy can have
        # |value| below its own MC standard error, and dividing by that turns a
        # 0.3 absolute miss into a 9000% "error".
        denom = max(abs(truth), 3.0 * sem, 1e-6)
        err = abs(res["value"] - truth) / denom
        trace = [round(v, 2) for _, v in res["value_trace"]]
        ok = "OK " if err <= 0.10 else "   "
        print(f"  [{ok}] {name:<22} value {res['value']:8.2f}  "
              f"|err| {abs(res['value']-truth):7.2f}  rel {err:6.1%}")
        print(f"       trace {trace}")
        rows.append({"name": name, "config": cfg, "value": res["value"],
                     "rel_err": err, "trace": res["value_trace"]})

    best = min(rows, key=lambda r: r["rel_err"])
    print(f"\nbest: {best['name']}  (err {best['rel_err']:.1%})")
    passing = [r["name"] for r in rows if r["rel_err"] <= 0.10]
    print(f"within 10% of truth: {passing or 'NONE -- FQE not usable yet'}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("ope_grid_smoke.json" if args.smoke else "ope_grid.json")
    p.write_text(json.dumps({"env": d.env_id, "steps": args.steps,
                             "mc_truth": truth, "mc_sem": sem, "rows": rows,
                             "within_10pct": passing,
                             "elapsed_min": (time.perf_counter() - t0) / 60.0},
                            indent=2), encoding="utf-8")
    print(f"wrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
