"""T3 -- convergence calibration, on the vendored REFERENCE IQL (S1 restart).

Runs ONE training and records, at four checkpoints, the two quantities the
downstream decisions depend on:

  1. policy evaluation score          -> baseline step budget for the main table
  2. critic overestimation per bucket -> whether GATE 1 is valid at all

These are different quantities. Offline RL policies are support-constrained
and plateau early while the critic keeps improving, so calibrating on the
policy score alone would let GATE 1 run against an unconverged critic.

n=5 is deliberate: it is the middle of GATE 1's n-grid, so this run doubles as
one of GATE 1's runs.

What changed from the withdrawn version
----------------------------------------
The previous S1 round trained our own IQL, which omitted the reference's reward
rescaling and therefore chased an unreachable fixed point (309.3 against a
reachable 55.0). Everything is now built on `harness.load`, which applies that
rescaling, and on `SupportAwareIQL`, a subclass of the vendored reference whose
control path is unit-tested to be parameter-identical to it.

Checkpoint range reduced from 250k/500k/1M/2M to 50k/100k/250k/500k, from
measured evidence rather than preference:

  * the verified reference run on this dataset converges: V increments fall
    14.87, 9.22, 5.99, 3.41, 2.32, 1.19, 0.77, 0.48, 0.21 per 10k steps, i.e.
    0.21 on a base of 61.3 (0.34%) by 100k;
  * it reaches normalized 43.4 at 100k against ~44-46 published, so the policy
    is already at the right answer there;
  * n=5 measures 12.96 ms/step on this machine, so 2M steps is 5.9 h against a
    7.4 h budget that must also cover four MC measurements. The reduced grid
    costs about 1.8 h.

If the critic bias is still drifting at 500k the checkpoints will show it and
the range can be extended; that is the question this run exists to answer.

    python experiments/calibrate.py
    python experiments/calibrate.py --smoke
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

from orl.algos.mc_reference import per_bucket_bias, reference_callables  # noqa: E402
from orl.coverage.buckets import buckets_for  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.methods.harness import MAX_EPISODE_STEPS, evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR, parse_env_id  # noqa: E402
from orl.record.logger import RunGuard  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CHECKPOINTS = (50_000, 100_000, 250_000, 500_000)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def nstep_kwargs(d, j):
    """Whole-dataset arrays the n-step target indexes by absolute id."""
    return {"indices": j, "dataset_rewards": d.rew,
            "dataset_next_obs": d.next_obs, "dataset_terminals": d.terminals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--checkpoints", default=None,
                    help="comma separated; overrides the default grid")
    ap.add_argument("--n-step", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--states-per-bucket", type=int, default=1000)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.checkpoints = "500,1000,2000"
        args.eval_episodes = 2
        args.states_per_bucket = 24
        args.mc_horizon = 50

    checkpoints = ([int(c) for c in args.checkpoints.split(",")]
                   if args.checkpoints else list(DEFAULT_CHECKPOINTS))
    final_step = checkpoints[-1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"calibration: {args.env}  n_step={args.n_step}  seed={args.seed}")
    print(f"checkpoints: {checkpoints}   device={dev}")
    print(f"MC: {args.states_per_bucket} states/bucket, horizon {args.mc_horizon}")
    print(f"eval: {args.eval_episodes} episodes\n")

    d = load(args.env, with_coverage=False, normalize=True)
    print(f"reward: raw {d.raw_rew_mean:.4f} -> scaled {d.scaled_rew_mean:.4f} "
          f"(x{d.reward_scale:.4f})")
    print(f"reachable fixed point = {d.scaled_rew_mean / (1 - args.discount):.1f}")
    expect = MAX_EPISODE_STEPS / (d.ret_range[1] - d.ret_range[0])
    if abs(d.reward_scale - expect) > 1e-9:
        raise AssertionError(f"reward_scale {d.reward_scale} vs range {expect}")

    buckets = buckets_for(args.env)
    print(f"bucket edges={tuple(round(e, 3) for e in buckets.edges)}  "
          f"support={buckets.support_threshold:.3f}  "
          f"sizes={buckets.bucket_sizes()}")

    # RAW observations: mc_reference restores a simulator state from them, and
    # the critic/policy callables normalise internally.
    with D4RLDataset(args.env, mode="stream") as ds:
        raw_obs = ds.get("observations")
        ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        max_steps=final_step, tau=args.tau, beta=args.beta, alpha=args.alpha,
        discount=args.discount, episode_lookup=ep,
        coverage=None, fixed_n=args.n_step, n_min=1, n_max=args.n_step,
    )
    print(f"n-step policy = {agent.requested_n_summary()}\n")

    O, A, R = to_t(d.obs, dev), to_t(d.act, dev), to_t(d.rew, dev)
    NO, TE = to_t(d.next_obs, dev), to_t(d.terminals, dev)
    policy_fn, q_fn = reference_callables(agent, d.normer, device=dev)
    bucket_id = np.digitize(buckets.scores, buckets.edges)

    probe = np.sort(np.random.default_rng(7).choice(len(O), 4096, replace=False))
    probe_o, probe_a = O[probe], A[probe]

    def measure() -> dict:
        ev = evaluate(agent.policy, d.domain, d.normer,
                      n_episodes=args.eval_episodes)
        b = per_bucket_bias(
            d.domain, raw_obs, bucket_id, buckets.edges, policy_fn, q_fn,
            gamma=args.discount, horizon=args.mc_horizon,
            states_per_bucket=args.states_per_bucket, device=dev,
            mc_scale=d.reward_scale)
        with torch.no_grad():
            v_mean = float(agent.vf(probe_o).mean())
            q_mean = float(agent.q_target(probe_o, probe_a).mean())
        return {"policy_raw": ev["raw_mean"], "policy_std": ev["raw_std"],
                "policy_score": ev["score_mean"], "V_mean": v_mean,
                "Q_mean": q_mean, "buckets": b}

    rng = np.random.default_rng(args.seed)
    records, done, last = [], 0, {}
    t0 = time.perf_counter()
    for cp in checkpoints:
        while done < cp:
            j = np.sort(rng.integers(0, len(O), args.batch_size))
            last = agent.update(O[j], A[j], NO[j], R[j], TE[j],
                                **nstep_kwargs(d, j))
            done += 1
            if done % 25_000 == 0:
                rate = done / (time.perf_counter() - t0)
                print(f"  {done:>9,}/{cp:,}  "
                      f"({(time.perf_counter() - t0) / 60:.0f} min, "
                      f"{1000 * (time.perf_counter() - t0) / done:.1f} ms/step, "
                      f"eta {(cp - done) / rate / 60:.0f} min)", flush=True)
        rec = measure()
        rec["step"] = cp
        rec["elapsed_min"] = (time.perf_counter() - t0) / 60.0
        # The plan asks for q_loss / v_loss / pi_loss / q_mean per checkpoint.
        # SupportAwareIQL.update reports them as q_loss / value_loss /
        # actor_loss, so they are renamed here rather than at the call site.
        rec["q_loss"] = last["q_loss"]
        rec["v_loss"] = last["value_loss"]
        rec["pi_loss"] = last["actor_loss"]
        records.append(rec)
        print(f"\nstep {cp:>9,}  ({rec['elapsed_min']:.0f} min)")
        print(f"  policy raw={rec['policy_raw']:9.1f} "
              f"score={rec['policy_score']:7.1f} "
              f"V={rec['V_mean']:8.2f} Q={rec['Q_mean']:8.2f} "
              f"q_loss={rec['q_loss']:.4f} v_loss={rec['v_loss']:.4f}")
        for k, b in rec["buckets"].items():
            line = f"  bucket {k}: n={b['n_used']:>5} mc={b['mc_mean']:8.2f}"
            line += f"+-{b['mc_sem']:.2f} critic={b['critic_mean']:8.2f}"
            line += f" over={b['overest_mean']:7.3f}+-{b['overest_sem']:.3f}"
            line += f" signed={b['signed_bias_mean']:+8.3f}"
            line += f"+-{b['signed_bias_sem']:.3f} trunc={b['n_truncated']}"
            print(line, flush=True)

    out = {
        "env_id": args.env, "n_step": args.n_step, "seed": args.seed,
        "gamma": args.discount, "checkpoints": checkpoints, "device": dev,
        # A smoke calibration must never be able to set GATE 1's step budget;
        # `gate1.py` refuses to read one.
        "smoke": bool(args.smoke),
        "reward_scale": d.reward_scale,
        "ret_range": list(d.ret_range),
        "mc_states_per_bucket": args.states_per_bucket,
        "mc_horizon": args.mc_horizon,
        "bucket_edges": list(buckets.edges),
        "support_threshold": buckets.support_threshold,
        "baseline": "vendored gwthomas/IQL-PyTorch via SupportAwareIQL",
        "records": records,
    }
    domain, tier = parse_env_id(args.env)
    # A smoke run must not be able to overwrite the real calibration, which is
    # the file `gate1.py` reads to set its step budget.
    out_name = "calibration_smoke.json" if args.smoke else "calibration.json"
    with RunGuard(run_id=f"calibrate_{args.env}_n{args.n_step}_s{args.seed}",
                  method=f"iql+fixed_n{args.n_step}", dataset=args.env,
                  domain=domain, tier=tier, seed=args.seed, stage="S1",
                  score_norm=records[-1]["policy_score"],
                  score_raw=records[-1]["policy_raw"],
                  # Passed explicitly: this guard wraps only the file write, so
                  # its own timer would report ~0 min. `t0` starts before the
                  # training loop and is what the log's progress lines use.
                  wall_clock_min=(time.perf_counter() - t0) / 60.0,
                  notes="T3 calibration on the reference baseline"):
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        p = BENCH_DIR / out_name
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
