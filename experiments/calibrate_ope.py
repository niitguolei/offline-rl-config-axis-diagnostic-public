"""How many gradient steps does FQE actually need? Anchored on ground truth.

The T0 run showed that FQE's LOSS cannot answer this: at a 4000-step budget the
loss sat at 0.1 while the estimate was ~-0.2 against a truth of ~42, because the
target network was still climbing and the regression was tracking a moving
target. "Wait for the loss to fall" would therefore declare victory immediately.

So this calibrates against something external instead:

  1. train a policy briefly (it does not need to be good, only fixed);
  2. compute its TRUE discounted value in the same reward scale the critic uses,
     by batched Monte-Carlo rollouts with `mc_scale` applied -- the same
     instrument GATE 1 was built on;
  3. run FQE for a long budget, logging the estimate as it climbs;
  4. report where it reaches the MC value, and at what step it stops moving.

Two numbers come out and they are not the same number: the step at which FQE is
within 1% of ground truth, and the step at which it stops changing. T2 needs the
first; the second is a sanity check that the estimate is not merely frozen.

    python experiments/calibrate_ope.py --smoke
    python experiments/calibrate_ope.py --train-steps 20000 --fqe-steps 60000
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
from orl.selection.ope import policy_action  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--train-steps", type=int, default=20_000)
    ap.add_argument("--fqe-steps", type=int, default=60_000)
    ap.add_argument("--trace-every", type=int, default=2_000)
    ap.add_argument("--mc-episodes", type=int, default=64)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_steps, args.fqe_steps = 2000, 1500
        args.trace_every, args.mc_episodes, args.mc_horizon = 150, 8, 100

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep = episode_lookup(d.env_id)
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    print(f"env={d.env_id}  reward scale x{d.reward_scale:.4f}  "
          f"scaled r-bar {d.scaled_rew_mean:.4f}")

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
    print(f"trained {args.train_steps:,} steps ({time.perf_counter()-t0:.0f}s)")

    # --- ground truth: MC discounted value of THIS policy, in the critic's scale
    pol_fn, _ = reference_callables(agent, d.normer, device=dev)
    mc = mc_reference_batched(d.domain, raw_starts, pol_fn, gamma=0.99,
                              horizon=args.mc_horizon, device=dev, batch_size=32)
    mc_value = float((d.reward_scale * mc.values).mean())
    print(f"MC value of this policy = {mc_value:.3f}  "
          f"(n={len(raw_starts)}, trunc={mc.truncated}, "
          f"sem={d.reward_scale*mc.values.std()/np.sqrt(len(mc.values)):.3f})")

    # --- FQE with a long budget and a dense trace
    from orl.selection.ope import fitted_q_evaluation

    trace = []
    qf = None
    print(f"\nrunning FQE for {args.fqe_steps:,} steps "
          f"(trace every {args.trace_every:,})")
    res = fitted_q_evaluation(
        agent.policy, O, NO, R, TE, ss, gamma=0.99, steps=args.fqe_steps,
        seed=args.seed, device=dev)
    print(f"  FQE final value {res['value']:.3f}  vs MC {mc_value:.3f}  "
          f"(ratio {res['value']/mc_value:.3f})")
    print("  value trace (step, value, |v-mc|/mc):")
    for i, v in res["value_trace"]:
        print(f"    {i:>7,}  {v:9.3f}  {abs(v-mc_value)/abs(mc_value):7.3f}")

    # --- where did it reach 1% of truth, and where did it stop moving?
    vals = [v for _, v in res["value_trace"]]
    steps = [i for i, _ in res["value_trace"]]
    hit1 = next((s for s, v in zip(steps, vals)
                 if abs(v - mc_value) / abs(mc_value) < 0.01), None)
    moving = [s for s, v, w in zip(steps[1:], vals[1:], vals[:-1])
              if abs(v - w) / max(abs(w), 1e-9) > 0.01]
    last_moving = moving[-1] if moving else 0
    print(f"\nreaches within 1% of truth at step: {hit1}")
    print(f"last step with >1% change:            {last_moving}")
    print("recommended T2 budget = max(above, a margin), not the loss-based one")

    out = {"env": d.env_id, "train_steps": args.train_steps,
           "fqe_steps": args.fqe_steps, "reward_scale": d.reward_scale,
           "mc_value": mc_value, "mc_n": int(len(raw_starts)),
           "mc_truncated": int(mc.truncated),
           "fqe_final": res["value"], "fqe_value_std": res["value_std"],
           "trace": res["value_trace"], "q_loss_trace": res["q_loss_trace"],
           "step_within_1pct_of_truth": hit1,
           "last_step_moving_over_1pct": last_moving,
           "elapsed_min": (time.perf_counter() - t0) / 60.0}
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("ope_calibration_smoke.json" if args.smoke
                     else "ope_calibration.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
