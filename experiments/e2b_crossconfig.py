"""E2b: selection BETWEEN configurations, where no free rule exists.

Review V1 found the structural problem behind every earlier experiment: they all
selected a checkpoint WITHIN one training run, which is the easiest version of
selection and precisely where "take the last checkpoint" wins. E2c then confirmed
that the within-run problem is trivial in most regimes and that in the one
non-trivial regime (1% data) no estimator ranks well anyway.

Real model selection chooses between CONFIGURATIONS -- here six (learning rate x
expectile), each trained once at a 1% episode subsample -- and there "the last
one" is not a rule at all. The question is therefore well posed:

    can any estimator pick the best configuration better than chance?

Free baselines: `random` (expected gap when one of the six is picked uniformly)
and, if a hyperparameter happens to order the true scores monotonically, that
hyperparameter. A useful estimator must beat random by more than the seed-level
SEM -- an estimator that cannot is not selecting, whatever its rank correlation.

Cost note: an 8-config grid x 3 seeds x 3 domains with three estimators costs
~20 h, so the grid is 6 configs (the learning-rate axis fully, the expectile axis
at its extremes) x 3 seeds x 2 domains = 36 runs, ~7 h.

    python experiments/e2b_crossconfig.py --smoke
    python experiments/e2b_crossconfig.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(ROOT))

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.gaps.construct import full_dataset, subsample_episodes  # noqa: E402
from orl.methods.harness import (  # noqa: E402
    episode_lookup,
    evaluate_per_episode,
    load,
)
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis  # noqa: E402
from orl.selection.ope import critic_mean_value, fitted_q_evaluation  # noqa: E402
from orl.selection.report import oracle_gap  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

FQE_CFG = dict(loss="huber", target_agg="mean", alpha=0.05, grad_clip=10.0,
               lr=1e-3)
# Round 13/C sensitivity grid. The base entry deliberately duplicates the
# historical setting so a reader can verify that the extra path is wired to the
# same estimator rather than to a subtly different one.
FQE_SENSITIVITY = {
    "base": {},
    "mse": {"loss": "mse"},
    "long_slow": {"steps": 12000, "lr": 3e-4},
}
DEFAULT_LRS = (1e-4, 3e-4, 1e-3)
DEFAULT_TAUS = (0.5, 0.9)
CONFIGS = [(lr, tau) for lr in DEFAULT_LRS for tau in DEFAULT_TAUS]

# Per-domain non-trivial regime, from the E0c regime search (13 probes). The
# window is domain-dependent and that is itself the finding: walker2d is
# non-trivial in BOTH seeds at 5% (17.8% / 18.0% last gap, level spread x2.2);
# halfcheetah only at 1-5% (5% was 1 of 2 seeds); hopper is non-trivial at FULL
# data (5.7%), which matters because it avoids the "2 episodes is not a dataset"
# objection that the 1% regime invites.
REGIME = {
    "walker2d_medium_replay-v2": 0.05,
    "halfcheetah_medium_replay-v2": 0.01,
    "hopper_medium_replay-v2": 1.00,
}


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def run_config(env_id: str, lr: float, tau: float, seed: int, args,
               dev: str) -> dict:
    d = load(env_id, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    frac = REGIME.get(env_id, args.subsample)
    small = (subsample_episodes("sub", base, frac, seed=seed)
             if frac < 1.0 else full_dataset("full", base))
    ep = small.episode_lookup()
    O, A = to_t(small.obs, dev), to_t(small.act, dev)
    R, NO, TE = to_t(small.rew, dev), to_t(small.next_obs, dev), to_t(small.terminals, dev)
    ep_s = np.concatenate(([0], np.asarray(ep.ends)[:-1] + 1))
    anchor_idx = ep_s[: min(len(ep_s), 32)]
    anchor = O[torch.as_tensor(anchor_idx, device=dev).long()]
    beh = (None if args.no_is else
           fit_behaviour(small.obs, small.act, steps=args.behaviour_steps,
                         device=dev, seed=seed))
    torch.manual_seed(seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=lr),
        max_steps=args.steps, tau=tau, beta=3.0, alpha=0.005, discount=0.99,
        episode_lookup=ep, coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(O[j], A[j], NO[j], R[j], TE[j])
    ev = evaluate_per_episode(agent.policy, d.domain, d.normer,
                              args.eval_episodes, seed=0)
    pol = agent.policy
    fqe = fitted_q_evaluation(pol, O, NO, R, TE, anchor, gamma=0.99,
                              steps=args.fqe_steps, seed=seed, device=dev,
                              **FQE_CFG)
    fqe_sensitivity = None
    if getattr(args, "fqe_sensitivity", False):
        fqe_sensitivity = {}
        for name, override in FQE_SENSITIVITY.items():
            cfg = dict(FQE_CFG)
            cfg.update(override)
            sens_steps = cfg.pop("steps", args.fqe_steps)
            if args.smoke:
                sens_steps = min(sens_steps, args.fqe_steps)
            fqe_sensitivity[name] = fitted_q_evaluation(
                pol, O, NO, R, TE, anchor, gamma=0.99,
                steps=sens_steps, seed=seed, device=dev,
                **cfg,
            )
    pd = {"value": None, "ess": None}
    if beh is not None:
        with torch.no_grad():
            log_r = (pol(O).log_prob(A) - beh.log_prob(O, A)).cpu().numpy()
        pd = pdis(np.asarray(small.rew, dtype=np.float64), ep_s,
                  np.asarray(ep.ends), log_r, gamma=0.99, clip=2.0)
    def matched_value(states):
        """The functional the CQL arm has always used: min(Q1,Q2) at pi's mean.

        Review M2 of the 2026-10-09 audit: the IQL arm read its free baseline
        off `V(s)` (an expectile of Q over DATASET actions) while the CQL arm
        read `min(Q1,Q2)(s, pi(s).mean)`. Those are different objects, so the
        cross-arm contrast -- the paper's central figure -- was not like for
        like. Recording both here lets that contrast be made on a matched
        functional without discarding the V-based numbers the earlier runs
        produced.
        """
        with torch.no_grad():
            return agent.qf(states, agent.policy(states).mean)

    return {"env": d.env_id, "lr": lr, "tau": tau, "seed": seed,
            "n_data": int(small.n), "n_episodes": int(len(ep.ends)),
            "true_score": ev["score_mean"], "score_sem": ev["score_sem"],
            "per_episode": ev["scores"],
            "critic_mean": critic_mean_value(agent.vf, anchor),
            "critic_mean_q": critic_mean_value(matched_value, anchor),
            "fqe": fqe["value"], "fqe_sensitivity": fqe_sensitivity,
            "pdis": pd["value"], "pdis_ess": pd.get("ess")}


def analyse(runs: list[dict]) -> dict:
    out = {}
    for env_id in sorted({r["env"] for r in runs}):
        per_seed = {}
        for seed in sorted({r["seed"] for r in runs if r["env"] == env_id}):
            sel = [r for r in runs if r["env"] == env_id and r["seed"] == seed]
            if len(sel) < 3:
                continue
            truth = np.array([r["true_score"] for r in sel])
            ent = {"n_configs": len(sel),
                   "true_scores": truth.tolist(),
                   # random: expected gap over a uniform pick of the configs
                   "random_expected_gap": float(np.mean(
                       [oracle_gap(truth, i) for i in range(len(truth))])),
                   "oracle_index": int(np.argmax(truth))}
            for key in ("critic_mean", "fqe", "pdis"):
                if any(r.get(key) is None for r in sel):
                    continue          # --no-is leaves PDIS unmeasured
                est = np.array([float(r[key]) for r in sel])
                pick = int(np.argmax(est))
                ent[key] = {"picked": pick, "gap": oracle_gap(truth, pick)}
            per_seed[seed] = ent
        if not per_seed:
            continue
        # A mean over seeds is only a mean of like things if every seed faced the
        # same number of configurations: the uniform-pick null depends on the
        # count. Extending an axis on some seeds but not others (which happened
        # when tau=0.7 was added to seeds 0-2 only) silently makes the summary
        # average a 9-way selection problem with a 6-way one.
        counts = sorted({v["n_configs"] for v in per_seed.values()})
        mixed = len(counts) > 1
        if mixed:
            print(f"  !! {env_id}: per-seed config counts differ {counts} -- "
                  f"a mean over these is NOT a mean of like things", flush=True)
        summary = {"per_seed": per_seed, "mixed_grid": mixed,
                   "config_counts": counts}
        for key in ("critic_mean", "fqe", "pdis"):
            if key not in next(iter(per_seed.values())):
                continue
            gaps = np.array([v[key]["gap"] for v in per_seed.values()])
            rnd = np.array([v["random_expected_gap"] for v in per_seed.values()])
            # PAIRED test. Each seed contributes one estimator gap and one
            # uniform-pick gap computed from the SAME truth, so the difference is
            # paired by construction. The original flag compared the two means
            # with only one side's SEM, which is not a test at all -- the
            # 2026-10-09 audit found that under a paired t every IQL cell goes
            # non-significant while both CQL critic cells stay significant.
            diff = rnd - gaps
            n = len(diff)
            if n > 1:
                paired_sem = float(diff.std(ddof=1) / np.sqrt(n))
                paired_t = float(diff.mean() / paired_sem) if paired_sem else float("nan")
                paired_p = float(2 * stats.t.sf(abs(paired_t), df=n - 1)) \
                    if np.isfinite(paired_t) else float("nan")
            else:
                paired_sem = paired_t = paired_p = float("nan")
            summary[key] = {
                "mean_gap": float(gaps.mean()),
                "mean_random_gap": float(rnd.mean()),
                "beats_random_by": float(rnd.mean() - gaps.mean()),
                "seed_gap_sem": float(gaps.std(ddof=1) / np.sqrt(len(gaps)))
                if len(gaps) > 1 else float("nan"),
                "paired_beats": float(diff.mean()),
                "paired_sem": paired_sem,
                "paired_t": paired_t,
                "paired_p": paired_p,
                "n_seeds": n,
                # primary verdict: the paired test, two-sided at 0.05
                "useful": bool(np.isfinite(paired_p) and paired_p < 0.05
                               and diff.mean() > 0),
                # kept so earlier documents can still be reproduced
                "useful_unpaired": bool(gaps.mean() < rnd.mean()
                                        - (gaps.std(ddof=1) / np.sqrt(len(gaps))
                                           if len(gaps) > 1 else 0.0)),
            }
        out[env_id] = summary
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="walker2d_medium_replay-v2,"
                                     "halfcheetah_medium_replay-v2,"
                                     "hopper_medium_replay-v2")
    ap.add_argument("--no-is", action="store_true",
                    help="skip the behaviour model and PDIS: the IS family has "
                         "been measured degenerate at every turn (ESS 1-2, or "
                         "policy-independent under clipping), so spending an "
                         "hour of a ten-hour run on it buys nothing")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lrs", default=",".join(f"{x:g}" for x in DEFAULT_LRS),
                    help="learning-rate axis of the config grid")
    ap.add_argument("--taus", default=",".join(f"{x:g}" for x in DEFAULT_TAUS),
                    help="expectile axis; widening it is how S3-2 turns the tau "
                         "axis from 2 levels into a analysable selection problem")
    ap.add_argument("--subsample", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=66_000)
    ap.add_argument("--fqe-steps", type=int, default=6000)
    ap.add_argument("--fqe-sensitivity", action="store_true",
                    help="also evaluate the three registered FQE settings; "
                         "used by the Round 13/C sensitivity check")
    ap.add_argument("--behaviour-steps", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="write to e2b_runs_<tag>.jsonl and skip the summary; the "
                         "step is launch-bound, so extensions are sharded")
    args = ap.parse_args()
    if args.smoke:
        args.envs, args.seeds = "walker2d_medium_replay-v2", "0"
        args.steps, args.fqe_steps, args.behaviour_steps = 1500, 300, 300
        args.eval_episodes = 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [(float(lr), float(t)) for lr in args.lrs.split(",")
               for t in args.taus.split(",")]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        name = "e2b_runs_smoke.jsonl"
    elif args.tag:
        name = f"e2b_runs_{args.tag}.jsonl"
    else:
        name = "e2b_runs.jsonl"
    p = BENCH_DIR / name
    runs, done = [], set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs.append(r)
                done.add((r["env"], r["lr"], r["tau"], r["seed"]))
    todo = [(e, lr, tau, s) for e in args.envs.split(",")
            for (lr, tau) in configs
            for s in [int(x) for x in args.seeds.split(",")]
            if (e, lr, tau, s) not in done]
    print(f"E2b: {len(todo)} config-runs to go, {len(done)} cached "
          f"({len(configs)} configs x seeds x envs)")
    t0 = time.perf_counter()
    for env_id, lr, tau, seed in todo:
        r = run_config(env_id, lr, tau, seed, args, dev)
        runs.append(r)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"  {env_id.split('-v2')[0]:<12} lr={lr:.0e} tau={tau} seed={seed}"
              f"  true {r['true_score']:6.2f} +- {r['score_sem']:4.2f}  "
              f"critic {r['critic_mean']:7.2f}  fqe {r['fqe']:7.2f}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    if args.tag:
        print(f"\nshard '{args.tag}' done")
        return 0

    res = analyse(runs)
    print("\ncross-configuration selection (higher gap = worse):")
    for env_id, s in res.items():
        print(f"  {env_id}")
        for key in ("critic_mean", "fqe", "pdis"):
            if key not in s:
                continue
            v = s[key]
            print(f"    {key:<12} gap {v['mean_gap']:6.1%}  "
                  f"random {v['mean_random_gap']:6.1%}  "
                  f"beats by {v['beats_random_by']:+6.1%}  "
                  f"-> {'USEFUL' if v['useful'] else 'no better than random'}")
    out = BENCH_DIR / ("e2b_summary_smoke.json" if args.smoke else "e2b_summary.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
