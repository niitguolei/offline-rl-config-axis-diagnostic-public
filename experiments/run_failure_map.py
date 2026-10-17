"""Fill the selection-failure map: condition x checkpoint x estimator.

Both axes now exist and were verified before this script was written:

  * x axis -- gap type. T1 measured that a state ball removes a real hole while
    leaving the action support alone (2.41 / 1.02), and that narrowing actions
    leaves the state side alone (0.94 / 0.58). Registered in GATES.md.
  * y axis -- estimator family. T0 measured three DIFFERENT failure modes,
    each anchored on MC ground truth:
      - critic mean: nearly free, ranked perfectly (Spearman +0.80/+1.00), but
        far from converged in absolute value;
      - FQE: converges, yet is policy-INSENSITIVE (40 for truths -0.51, 65.11);
      - WIS/PDIS: weights correct (control exact to 6 dp) but variance-degenerate
        (ESS 1-6, or 98% error for a healthy ESS).

Each cell records what a selection decision depends on: the estimator's value,
the checkpoint's TRUE normalised score, the rank correlation across checkpoints
within the condition, the oracle gap, and the ESS for the IS family. Ranking is
the primary metric -- selection needs ordering, not accurate values, which is
itself a measured finding.

Every cell is appended to a JSONL as it completes and is skipped on re-run:
this grid is ~35 minutes of GPU and the project has already lost 15 hours to a
crash that discarded finished work.

    python experiments/run_failure_map.py --smoke
    python experiments/run_failure_map.py
"""
from __future__ import annotations

import argparse
import copy
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
from orl.gaps.construct import action_gap, state_gap, subsample_episodes  # noqa: E402
from orl.methods.harness import episode_lookup, evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis, weighted_is  # noqa: E402
from orl.selection.ope import (  # noqa: E402
    critic_mean_value,
    fitted_q_evaluation,
    policy_action,
    spearman,
)

sys.stdout.reconfigure(line_buffering=True)

FQE_CFG = dict(loss="huber", target_agg="mean", alpha=0.05, grad_clip=10.0,
               lr=1e-3)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def build_condition(name: str, base: dict, fraction: float, seed: int):
    if name == "none":
        return None
    if name == "ball":
        return state_gap(name, base, drop_quantile=fraction, seed=seed, mode="ball")
    if name == "action":
        return action_gap(name, base, keep_inner=fraction, seed=seed)
    if name == "sub":
        return subsample_episodes(name, base, fraction, seed=seed)
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--conditions", default="none,ball,action,sub")
    ap.add_argument("--fraction", type=float, default=0.3,
                    help="ball drop quantile / sub episode fraction")
    ap.add_argument("--action-fraction", type=float, default=0.7,
                    help="action gap keeps this central fraction; 0.3 destroyed "
                         "the policy in the first slice (truth 1.2 vs 42)")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--checkpoints", default="10000,20000,30000,40000")
    ap.add_argument("--fqe-steps", type=int, default=8000)
    ap.add_argument("--behaviour-steps", type=int, default=20_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--mc-episodes", type=int, default=16)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.conditions = "ball,action"
        args.steps, args.checkpoints = 2000, "1000,2000"
        args.fqe_steps, args.behaviour_steps = 300, 500
        args.eval_episodes, args.mc_episodes, args.mc_horizon = 2, 4, 50
    cps = [int(c) for c in args.checkpoints.split(",")]
    if cps[-1] != args.steps:
        raise SystemExit("the last checkpoint must equal --steps")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    starts = np.concatenate(([0], ep_base.ends[:-1] + 1))
    with D4RLDataset(d.env_id, mode="stream") as ds:
        raw_obs = ds.get("observations")
    raw_starts = raw_obs[starts][: args.mc_episodes].astype(np.float64)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": starts, "ends": ep_base.ends}
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    # The tag carries the SCHEMA version: the first slice's cache lacks
    # mc_det_value, and silently reusing it produced a KeyError rather than a
    # wrong number -- lucky, but the fix is to never share a cache across
    # schemas.
    tag = "smoke2" if args.smoke else "map2"
    cells_path = BENCH_DIR / f"failure_map_{tag}.jsonl"
    done = set()
    if cells_path.exists():
        for line in cells_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["step"]))
    print(f"env={d.env_id}  conditions={args.conditions}  cps={cps}  "
          f"cached cells={len(done)}")
    # ONE behaviour model on the FULL base data. The behaviour policy is a
    # property of the dataset, not of the condition, and fitting it per condition
    # is what made the first slice's IS weights reflect fragment length (a ball
    # gap shortened mean episodes to 6.1 steps, so weighted_is evaluated a
    # 6-step return of ~3.3 rather than the policy's value).
    beh = fit_behaviour(d.obs, d.act, steps=args.behaviour_steps, device=dev,
                        seed=args.seed)
    rew_base = np.asarray(d.rew, dtype=np.float64)
    O_base, A_base = to_t(d.obs, dev), to_t(d.act, dev)
    base_ep_s = starts
    base_ep_e = np.asarray(ep_base.ends)
    print(f"behaviour model fitted on the full base data "
          f"({args.behaviour_steps:,} steps)")

    for cond in args.conditions.split(","):
        frac = args.action_fraction if cond == "action" else args.fraction
        g = build_condition(cond, base, frac, args.seed)
        obs = base["obs"] if g is None else g.obs
        act = base["act"] if g is None else g.act
        rew = base["rew"] if g is None else g.rew
        nobs = base["next_obs"] if g is None else g.next_obs
        term = base["terminals"] if g is None else g.terminals
        ep = ep_base if g is None else g.episode_lookup()
        O, A = to_t(obs, dev), to_t(act, dev)
        R, NO, TE = to_t(rew, dev), to_t(nobs, dev), to_t(term, dev)
        ss_idx = starts[: args.mc_episodes]
        ss = O[torch.as_tensor(ss_idx, device=dev).long()]
        torch.manual_seed(args.seed)
        agent = SupportAwareIQL(
            qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
            vf=ValueFunction(d.obs.shape[1]),
            policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
            max_steps=args.steps, tau=0.7, beta=3.0, alpha=0.005,
            discount=0.99, episode_lookup=ep,
            coverage=None, fixed_n=1, n_min=1, n_max=1,
        )
        rng = np.random.default_rng(args.seed)
        done_steps = 0
        for cp in cps:
            while done_steps < cp:
                j = np.sort(rng.integers(0, len(O), 256))
                agent.update(O[j], A[j], NO[j], R[j], TE[j])
                done_steps += 1
            if (cond, cp) in done:
                print(f"  {cond}@{cp:,} cached, skipping")
                continue
            pol = copy.deepcopy(agent.policy)
            ev = evaluate(pol, d.domain, d.normer, n_episodes=args.eval_episodes)
            cri = critic_mean_value(agent.vf, ss)
            fqe = fitted_q_evaluation(pol, O, NO, R, TE, ss, gamma=0.99,
                                      steps=args.fqe_steps, seed=args.seed,
                                      device=dev, **FQE_CFG)
            mu = torch.as_tensor(d.normer.mean, dtype=torch.float64, device=dev)
            sd = torch.as_tensor(d.normer.std, dtype=torch.float64, device=dev)

            def mean_fn(obs_raw):
                o = ((obs_raw.to(dev).to(torch.float64) - mu) / sd).to(torch.float32)
                with torch.no_grad():
                    return policy_action(pol, o)

            def sample_fn(obs_raw):
                o = ((obs_raw.to(dev).to(torch.float64) - mu) / sd).to(torch.float32)
                with torch.no_grad():
                    return pol(o).sample().clamp(-1.0, 1.0)

            # LEVEL anchor for critic/FQE: the deterministic mean policy, same
            # object they score, in the same reward scale.
            mc_det = mc_reference_batched(d.domain, raw_starts, mean_fn, gamma=0.99,
                                          horizon=args.mc_horizon, device=dev,
                                          batch_size=16)
            mc = mc_reference_batched(d.domain, raw_starts, sample_fn,
                                      gamma=0.99, horizon=args.mc_horizon,
                                      device=dev, batch_size=16)
            # IS on the ORIGINAL episodes with the shared behaviour model: the
            # condition changes what the learner sees, never what IS estimates.
            with torch.no_grad():
                lr_ = (pol(O_base).log_prob(A_base)
                       - beh.log_prob(O_base, A_base)).cpu().numpy()
            wis = weighted_is(rew_base, base_ep_s, base_ep_e, lr_, gamma=0.99,
                              clip=0.5)
            pdi = pdis(rew_base, base_ep_s, base_ep_e, lr_, gamma=0.99, clip=2.0)
            row = {"condition": cond, "step": cp,
                   "n": int(obs.shape[0]), "n_episodes": int(len(ep.ends)),
                   "mean_fragment_len": (float(g.lengths().mean())
                                         if g is not None else 0.0),
                   "true_raw": ev["raw_mean"], "true_score": ev["score_mean"],
                   "critic_mean": cri, "fqe": fqe["value"],
                   "wis_clip0.5": wis.get("value"), "wis_ess": wis.get("ess"),
                   "pdis_clip2.0": pdi.get("value"), "pdis_ess": pdi.get("ess"),
                   "mc_det_value": float((d.reward_scale * mc_det.values).mean()),
                   "mc_stochastic": float((d.reward_scale * mc.values).mean()),
                   "log_ratio_std": float(lr_.std())}
            with cells_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"  {cond:<7}@{cp:>7,}  true {ev['score_mean']:6.2f}  "
                  f"mc_det {row['mc_det_value']:6.2f}  "
                  f"critic {cri:7.2f}  fqe {fqe['value']:7.2f}  "
                  f"wis {float(wis.get('value') or float('nan')):8.2f} "
                  f"(ess {float(wis.get('ess') or 0):.1f})  "
                  f"pdis {float(pdi.get('value') or float('nan')):8.2f} "
                  f"(ess {float(pdi.get('ess') or 0):.1f})")

    # --- per-condition ranking quality
    rows = [json.loads(l) for l in cells_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print("\nranking within each condition (Spearman across checkpoints):")
    summary = {}
    for cond in sorted({r["condition"] for r in rows}):
        sel = sorted([r for r in rows if r["condition"] == cond],
                     key=lambda r: r["step"])
        truth = np.array([r["true_score"] for r in sel])
        if len(sel) < 2:
            continue
        entry = {}
        for key in ("critic_mean", "fqe", "pdis_clip2.0", "wis_clip0.5"):
            est = np.array([float(r[key]) for r in sel])
            order = np.argsort(-est)
            oracle = int(np.argmax(truth))
            gap = float(1.0 - truth[order[0]] / max(abs(truth[oracle]), 1e-9))
            entry[key] = {"spearman": spearman(est, truth), "oracle_gap": gap}
        # LEVEL, now in a common unit: critic and FQE score the deterministic
        # mean policy, and mc_det_value is that policy's MC value in the same
        # (scaled-reward) unit. The first slice compared them against a D4RL
        # normalised score, which is a different yardstick entirely.
        for key in ("critic_mean", "fqe"):
            if any("mc_det_value" not in r for r in sel):
                continue          # stale-schema row; do not guess an anchor
            est = np.array([float(r[key]) for r in sel])
            anch = np.array([float(r["mc_det_value"]) for r in sel])
            entry[key]["rel_bias_vs_mc_det"] = float(
                np.mean(np.abs(est - anch) / np.maximum(np.abs(anch), 1e-9)))
        summary[cond] = entry
        print(f"  {cond}:")
        for key, v in entry.items():
            extra = (f"  rel-bias-vs-MC {v['rel_bias_vs_mc_det']:.0%}"
                     if "rel_bias_vs_mc_det" in v else "")
            print(f"    {key:<14} spearman {v['spearman']:+.2f}  "
                  f"gap {v['oracle_gap']:+.3f}{extra}")
    (BENCH_DIR / f"failure_map_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote failure_map_{tag}.jsonl + _summary.json "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
