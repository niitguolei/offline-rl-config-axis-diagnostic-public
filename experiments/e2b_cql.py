"""E2b, second algorithm: CQL(H) instead of IQL. Same task, same estimators.

Review item 1 of the coverage plan: the paper's scope was three MuJoCo domains
with ONE algorithm, which is the first thing a reviewer attacks. This arm runs
the cross-configuration selection task with CQL, transcribing the authors' loss
(see `orl.algos.cql` for the provenance and the two documented deviations).

Why CQL specifically, rather than just "a second algorithm"
----------------------------------------------------------
The paper's claim is that a selection signal is untrustworthy to the extent that
it is a function of the axis being compared, and that this can be measured with
`corr(estimate, hyperparameter)` before any truth is known. IQL offers one such
axis (the expectile tau). CQL offers an independent one: `min_q_weight`, the
authors' alpha, which multiplies the conservative penalty and therefore drives
the critic's scale directly.

That makes this arm a PROSPECTIVE test of the diagnostic, not just extra
coverage. The prediction is registered here, before the runs:

    corr(critic_mean, min_q_weight) will be strongly positive, and the critic
    mean will then be no better than random at picking the best configuration.

The estimators are unchanged (`analyse` is imported from the IQL arm, so the
criterion is literally the same code, not a copy). One definition has to change
and is stated explicitly: IQL's free baseline is the mean of its V over anchor
states, while CQL has no V, so the analogue used here is
mean min(Q1,Q2)(s, pi(s)) over the same anchor states -- the algorithm's own
value estimate of its own policy, which is what the free baseline means.

Usage:
    python experiments/e2b_cql.py --smoke
    python experiments/e2b_cql.py --no-is
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
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.value_functions import TwinQ  # noqa: E402

from e2b_crossconfig import REGIME, analyse, to_t  # noqa: E402
from orl.algos.cql import ConservativeQLearning  # noqa: E402
from orl.algos.tanh_policy import TanhGaussianPolicy  # noqa: E402
from orl.gaps.construct import full_dataset, subsample_episodes  # noqa: E402
from orl.methods.harness import (  # noqa: E402
    episode_lookup,
    evaluate_per_episode,
    load,
)
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis  # noqa: E402
from orl.selection.ope import critic_mean_value, fitted_q_evaluation  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

FQE_CFG = dict(loss="huber", target_agg="mean", alpha=0.05, grad_clip=10.0, lr=1e-3)
# The authors' README recommends min_q_weight of 5.0 or 10.0 with policy_lr of
# 1e-4 or 3e-4 for MuJoCo. The grid keeps their two orders of magnitude apart so
# that the leak, if it exists, has room to show: 1.0 (under-regularised) to 10.0.
# The authors' README recommends min_q_weight of 5.0 or 10.0 for MuJoCo. The
# grid started one order of magnitude lower (1.0) to widen the axis, and the
# pre-run gate killed it: at alpha=1.0 on walker2d-medium-replay the critic mean
# reached 2224 and the policy scored -0.24, while alpha=10 kept the critic
# bounded (see experiments/diag_cql_divergence.py). Pure TD (alpha=0) diverges
# even faster, so the conservatism is load-bearing here and 1.0 is simply not
# enough of it. The axis is therefore the two values the authors recommend.
DEFAULT_LRS = (1e-4, 3e-4, 1e-3)
DEFAULT_WEIGHTS = (5.0, 10.0)


def run_config(env_id: str, lr: float, min_q_weight: float, seed: int, args,
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
    agent = ConservativeQLearning(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        policy=TanhGaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=lr),
        max_steps=args.steps,
        min_q_weight=min_q_weight,
        discount=0.99,
        soft_target_tau=5e-3,
        deterministic_backup=True,
        policy_eval_start=args.policy_eval_start,
        device=dev,
    )
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(observations=O[j], actions=A[j], next_observations=NO[j],
                     rewards=R[j], terminals=TE[j])
    ev = evaluate_per_episode(agent.policy, d.domain, d.normer,
                              args.eval_episodes, seed=0)
    pol = agent.policy
    fqe = fitted_q_evaluation(pol, O, NO, R, TE, anchor, gamma=0.99,
                              steps=args.fqe_steps, seed=seed, device=dev,
                              **FQE_CFG)
    pd = {"value": None, "ess": None}
    if beh is not None:
        with torch.no_grad():
            log_r = (pol(O).log_prob(A) - beh.log_prob(O, A)).cpu().numpy()
        pd = pdis(np.asarray(small.rew, dtype=np.float64), ep_s,
                  np.asarray(ep.ends), log_r, gamma=0.99, clip=2.0)

    def own_value(states):
        """CQL's free baseline: min(Q1,Q2) at the mean policy action."""
        with torch.no_grad():
            return agent.qf(states, agent.policy(states).mean)

    critic = critic_mean_value(own_value, anchor)
    fqe_value = float(fqe["value"])
    # The pre-run gate caught a diverged run only by eye, so the runner now says
    # so itself. The reference is FQE, which is fitted to the same data under the
    # same policy: a critic more than 20x away from it has left the data behind.
    # (The gate's diverged run read critic 2224 against FQE 30, a ratio of 74.)
    ratio = abs(critic) / max(abs(fqe_value), 1e-6)
    diverged = (not np.isfinite(critic)) or ratio > 20.0
    return {"env": d.env_id, "lr": lr, "min_q_weight": min_q_weight, "seed": seed,
            "n_data": int(small.n), "n_episodes": int(len(ep.ends)),
            "true_score": ev["score_mean"], "score_sem": ev["score_sem"],
            "per_episode": ev["scores"],
            "critic_mean": critic, "fqe": fqe_value,
            "critic_over_fqe": ratio, "diverged": bool(diverged),
            "pdis": pd["value"], "pdis_ess": pd.get("ess")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="walker2d_medium_replay-v2,"
                                     "halfcheetah_medium_replay-v2,"
                                     "hopper_medium_replay-v2")
    ap.add_argument("--no-is", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lrs", default=",".join(f"{x:g}" for x in DEFAULT_LRS),
                    help="learning-rate axis of the config grid")
    ap.add_argument("--weights", default=",".join(f"{x:g}" for x in DEFAULT_WEIGHTS),
                    help="min_q_weight axis (the conservatism knob); narrowing "
                         "this is how the pre-run gate keeps its cost small")
    ap.add_argument("--subsample", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=66_000)
    ap.add_argument("--fqe-steps", type=int, default=6000)
    ap.add_argument("--behaviour-steps", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--policy-eval-start", type=int, default=0,
                    help="the authors' BC warm-up length; 0 because their 40000 "
                         "is calibrated to a ~3M-step run")
    ap.add_argument("--tag", default=None,
                    help="write to e2b_cql_runs_<tag>.jsonl and skip the summary. "
                         "The step is launch-bound and single-threaded while the "
                         "GPU sits at ~19%, so the arm is sharded across "
                         "processes and merged by experiments/e2b_cql_report.py")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.envs, args.seeds = "walker2d_medium_replay-v2", "0"
        args.steps, args.fqe_steps, args.behaviour_steps = 1500, 300, 300
        args.eval_episodes = 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [(float(lr), float(w)) for lr in args.lrs.split(",")
               for w in args.weights.split(",")]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        name = "e2b_cql_runs_smoke.jsonl"
    elif args.tag:
        name = f"e2b_cql_runs_{args.tag}.jsonl"
    else:
        name = "e2b_cql_runs.jsonl"
    p = BENCH_DIR / name
    runs, done = [], set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs.append(r)
                done.add((r["env"], r["lr"], r["min_q_weight"], r["seed"]))
    todo = [(e, lr, w, s) for e in args.envs.split(",")
            for (lr, w) in configs
            for s in [int(x) for x in args.seeds.split(",")]
            if (e, lr, w, s) not in done]
    print(f"E2b-CQL: {len(todo)} config-runs to go, {len(done)} cached "
          f"({len(configs)} configs x seeds x envs)")
    t0 = time.perf_counter()
    for env_id, lr, w, seed in todo:
        r = run_config(env_id, lr, w, seed, args, dev)
        runs.append(r)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"  {env_id.split('-v2')[0]:<12} lr={lr:.0e} w={w:4.1f} seed={seed}"
              f"  true {r['true_score']:6.2f} +- {r['score_sem']:4.2f}  "
              f"critic {r['critic_mean']:8.2f}  fqe {r['fqe']:7.2f}  "
              f"{'DIVERGED' if r['diverged'] else 'ok':>8}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    if args.tag:
        print(f"\nshard '{args.tag}' done; merge with "
              f"experiments/e2b_cql_report.py")
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
    print("\nleak probe (registered before the runs):")
    for env_id in sorted({r["env"] for r in runs}):
        sub = [r for r in runs if r["env"] == env_id]
        w = np.array([float(r["min_q_weight"]) for r in sub])
        c = np.array([float(r["critic_mean"]) for r in sub])
        t = np.array([float(r["true_score"]) for r in sub])
        print(f"  {env_id:<32} corr(critic,w)={np.corrcoef(c, w)[0, 1]:+.2f}  "
              f"corr(critic,true)={np.corrcoef(c, t)[0, 1]:+.2f}")
    out = BENCH_DIR / ("e2b_cql_summary_smoke.json" if args.smoke else "e2b_cql_summary.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
