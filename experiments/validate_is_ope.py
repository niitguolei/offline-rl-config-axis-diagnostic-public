"""Validate the importance-sampling family against ground truth.

FQE failed by being policy-insensitive: it returned ~40 for policies whose true
values were -0.51 and 65.11. Importance sampling attacks exactly that, because
it reweights OBSERVED returns rather than trusting a learned critic. Its expected
failure mode is the opposite -- weight degeneracy over a 1000-step episode.

This script measures which of the two happens here, against ground truth:

  1. fit the behaviour model (D4RL stores no behaviour densities);
  2. control: with the behaviour model as the EVALUATION policy every ratio is
     exactly 1, so WIS must return the empirical discounted return;
  3. evaluate the stochastic IQL policy: its ground truth is an MC rollout with
     SAMPLED actions, because a density is only defined for a stochastic policy;
  4. report WIS and PDIS with and without weight clipping, alongside the two
     numbers that decide usability: effective sample size and the largest
     weight's share.

Reporting the ESS is the point. A self-normalised IS estimate whose ESS is near
1 is one lucky episode wearing the costume of an estimate.

    python experiments/validate_is_ope.py --smoke
    python experiments/validate_is_ope.py
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

from orl.algos.mc_reference import mc_reference_batched  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.methods.harness import episode_lookup, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis, weighted_is  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--iql-steps", type=int, default=20_000)
    ap.add_argument("--behaviour-steps", type=int, default=40_000)
    ap.add_argument("--mc-episodes", type=int, default=64)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iql_steps, args.behaviour_steps = 1000, 1000
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
    torch.manual_seed(args.seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.iql_steps, tau=0.7, beta=3.0, alpha=0.005,
        discount=0.99, episode_lookup=ep,
        coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(args.seed)
    for _ in range(args.iql_steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(O[j], A[j], NO[j], R[j], TE[j])
    beh = fit_behaviour(d.obs, d.act, steps=args.behaviour_steps,
                        device=dev, seed=args.seed)
    print(f"trained IQL {args.iql_steps:,} and behaviour {args.behaviour_steps:,} "
          f"({time.perf_counter()-t0:.0f}s)")
    rew = np.asarray(d.rew, dtype=np.float64)
    ep_s, ep_e = starts, np.asarray(ep.ends)
    ctrl = weighted_is(rew, ep_s, ep_e, np.zeros_like(rew), gamma=0.99)
    disc_mean = float(np.mean([
        float((rew[s:e + 1] * (0.99 ** np.arange(e - s + 1))).sum())
        for s, e in zip(ep_s, ep_e)]))
    print("\ncontrol (behaviour model as the evaluation policy):")
    print(f"  WIS {ctrl['value']:.6f}   empirical discounted mean "
          f"{disc_mean:.6f}   ess {ctrl['ess']:.1f}/{ctrl['n_episodes']}")
    pol = agent.policy
    with torch.no_grad():
        log_ratio = (pol(O).log_prob(A) - beh.log_prob(O, A)).cpu().numpy()
    mu = torch.as_tensor(d.normer.mean, dtype=torch.float64, device=dev)
    sd = torch.as_tensor(d.normer.std, dtype=torch.float64, device=dev)

    def sample_fn(obs_raw):
        o = ((obs_raw.to(dev).to(torch.float64) - mu) / sd).to(torch.float32)
        with torch.no_grad():
            return pol(o).sample().clamp(-1.0, 1.0)

    mc = mc_reference_batched(d.domain, raw_starts, sample_fn, gamma=0.99,
                              horizon=args.mc_horizon, device=dev, batch_size=32)
    truth = float((d.reward_scale * mc.values).mean())
    sem = float(d.reward_scale * mc.values.std() / np.sqrt(len(mc.values)))
    print(f"\nstochastic-policy MC truth = {truth:.3f} +- {sem:.3f}")
    print(f"log-ratio stats: mean {log_ratio.mean():.4f} "
          f"std {log_ratio.std():.4f}  min {log_ratio.min():.2f} "
          f"max {log_ratio.max():.2f}")
    rows = []
    for name, fn in (("wis", weighted_is), ("pdis", pdis)):
        for clip in (None, 2.0, 0.5):
            out = fn(rew, ep_s, ep_e, log_ratio, gamma=0.99, clip=clip)
            val = float(out["value"]) * 1.0     # rew is already scaled
            denom = max(abs(truth), 3.0 * sem, 1e-6)
            rel = abs(val - truth) / denom
            rows.append({"estimator": name, "clip": clip, "value": val,
                         "rel_bias": rel, "ess": out.get("ess"),
                         "max_weight_share": out.get("max_weight_share")})
            print(f"  {name:<5} clip={str(clip):<5} value {val:8.2f}  "
                  f"|err| {abs(val-truth):7.2f}  rel {rel:6.1%}  "
                  f"ess {float(out.get('ess') or float('nan')):8.2f}"
                  f"/{out.get('n_episodes', 0)}  maxshare "
                  f"{float(out.get('max_weight_share') or float('nan')):.3f}")
    usable = [r for r in rows
              if r["rel_bias"] <= 0.25 and (r["ess"] or 0) >= 10]
    verdict = (f"IS USABLE: {[(r['estimator'], r['clip']) for r in usable]}"
               if usable else
               "IS NOT USABLE HERE: no configuration is both accurate and "
               "non-degenerate")
    print(f"\n{verdict}")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("is_ope_smoke.json" if args.smoke else "is_ope.json")
    p.write_text(json.dumps(
        {"env": d.env_id, "control": ctrl, "control_empirical": disc_mean,
         "mc_truth_stochastic": truth, "mc_sem": sem,
         "log_ratio_stats": {"mean": float(log_ratio.mean()),
                             "std": float(log_ratio.std()),
                             "min": float(log_ratio.min()),
                             "max": float(log_ratio.max())},
         "rows": rows, "verdict": verdict,
         "elapsed_min": (time.perf_counter() - t0) / 60.0}, indent=2),
        encoding="utf-8")
    print(f"wrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
