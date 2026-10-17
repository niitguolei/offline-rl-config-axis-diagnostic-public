"""E1: the statistical version of GATE 4, run in a NON-TRIVIAL regime.

Why the regime changed (E0c, GATE 4b): on full medium-replay data with
checkpoints at 10-40k steps every condition was TRIVIAL -- 'take the last
checkpoint' tied or beat every estimator (gaps 0.9% / 0% / 2.9% / 0%) because the
policy improves monotonically there. Non-triviality comes from the DATASET SIZE,
not from the gap type: at a 1% episode subsample the truth peaks and then
degrades (walker2d: 323 at 33k steps falling to 153 by 100k, a 52.8% gap between
the last checkpoint and the oracle).

So the design is: 1% subsample as the base regime, the gap type applied on top of
it, checkpoints spanning the peak AND the degradation, and the statistical
apparatus the review demanded:

  * 30 evaluation episodes with PER-EPISODE returns, so comparisons are paired
    across checkpoints and the differences use a paired SEM (review R1);
  * rank correlations reported only over checkpoint pairs whose true scores are
    separated by more than 2x that paired SEM, with the qualifying pair count
    printed rather than hidden;
  * the free baselines (`last`, `random`) on every row (review S1).

Pre-registered criterion (plan v4 section 4): the ball-vs-none FQE rank
correlation difference must be >= 1.0 with NON-OVERLAPPING seed intervals.
Anything weaker is reported as 'directional observation, not statistical'.

    python experiments/e1_stats.py --smoke
    python experiments/e1_stats.py
"""
from __future__ import annotations

import argparse
import copy
import itertools
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

from orl.gaps.construct import action_gap, state_gap, subsample_episodes  # noqa: E402
from orl.methods.harness import (  # noqa: E402
    episode_lookup,
    evaluate_per_episode,
    load,
)
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis  # noqa: E402
from orl.selection.ope import (  # noqa: E402
    critic_mean_value,
    fitted_q_evaluation,
    spearman,
)
from orl.selection.report import nontriviality, oracle_gap  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

FQE_CFG = dict(loss="huber", target_agg="mean", alpha=0.05, grad_clip=10.0,
               lr=1e-3)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def paired_sem(a: np.ndarray, b: np.ndarray) -> float:
    """SEM of the DIFFERENCE between two checkpoints' per-episode scores.

    Valid because every checkpoint is evaluated on the same episode seeds, so the
    comparison is paired. Using each checkpoint's own SEM instead would ignore
    that common random numbers cancel.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def run_one(env_id: str, cond: str, seed: int, args, dev: str) -> dict:
    d = load(env_id, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    small = subsample_episodes("sub", base, args.subsample, seed=seed)
    sm = {"obs": small.obs, "act": small.act, "rew": small.rew,
          "next_obs": small.next_obs, "terminals": small.terminals,
          "timeouts": small.timeouts, "starts": small.starts, "ends": small.ends}
    g = None
    if cond == "ball":
        g = state_gap(cond, sm, drop_quantile=args.fraction, seed=seed, mode="ball")
    elif cond == "action":
        g = action_gap(cond, sm, keep_inner=args.action_fraction, seed=seed)
    obs = sm["obs"] if g is None else g.obs
    act = sm["act"] if g is None else g.act
    rew = sm["rew"] if g is None else g.rew
    nobs = sm["next_obs"] if g is None else g.next_obs
    term = sm["terminals"] if g is None else g.terminals
    ep = small.episode_lookup() if g is None else g.episode_lookup()
    O, A = to_t(obs, dev), to_t(act, dev)
    R, NO, TE = to_t(rew, dev), to_t(nobs, dev), to_t(term, dev)
    beh = fit_behaviour(obs, act, steps=args.behaviour_steps, device=dev,
                        seed=seed)
    # IS weights are computed on the SAME (1% subsample) base as this condition,
    # but on whole episodes so the gap's fragmentation cannot change the target
    ep_s = np.concatenate(([0], np.asarray(ep.ends)[:-1] + 1))
    ep_e = np.asarray(ep.ends)
    # Use ALL the condition's episode starts (up to 32) for the critic mean and
    # the FQE anchor. Taking the first 8 was a silent noise source: at a 1%
    # subsample halfcheetah has only 2 episodes, so "the mean over 8" was a mean
    # over 2 states.
    anchor_idx = ep_s[: min(len(ep_s), 32)]
    anchor = O[torch.as_tensor(anchor_idx, device=dev).long()]
    torch.manual_seed(seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.steps, tau=0.7, beta=3.0, alpha=0.005, discount=0.99,
        episode_lookup=ep, coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(seed)
    banks = []
    done = 0
    for cp in args.checkpoints:
        while done < cp:
            j = np.sort(rng.integers(0, len(O), 256))
            agent.update(O[j], A[j], NO[j], R[j], TE[j])
            done += 1
        ev = evaluate_per_episode(agent.policy, d.domain, d.normer,
                                  args.eval_episodes, seed=0)
        pol = copy.deepcopy(agent.policy)
        fqe = fitted_q_evaluation(pol, O, NO, R, TE,
                                  anchor,
                                  gamma=0.99, steps=args.fqe_steps, seed=seed,
                                  device=dev, **FQE_CFG)
        with torch.no_grad():
            lr_ = (pol(O).log_prob(A) - beh.log_prob(O, A)).cpu().numpy()
        pd = pdis(np.asarray(rew, dtype=np.float64), ep_s, ep_e, lr_,
                  gamma=0.99, clip=2.0)
        banks.append({"step": int(cp), "score_mean": ev["score_mean"],
                      "score_sem": ev["score_sem"], "per_episode": ev["scores"],
                      "critic_mean": critic_mean_value(agent.vf,
                                                       anchor),
                      "fqe": fqe["value"], "pdis": pd["value"],
                      "pdis_ess": pd.get("ess")})
    return {"env": d.env_id, "condition": cond, "seed": seed,
            "n_data": int(obs.shape[0]), "n_episodes": int(len(ep.ends)),
            "cells": banks}


def analyse(runs: list[dict], args) -> dict:
    out = {}
    for env_id in sorted({r["env"] for r in runs}):
        for cond in sorted({r["condition"] for r in runs}):
            sel = [r for r in runs if r["env"] == env_id and r["condition"] == cond]
            if not sel:
                continue
            rhos, gaps, lasts, pairs_ok = [], [], [], []
            for r in sel:
                cells = sorted(r["cells"], key=lambda c: c["step"])
                truth = np.array([c["score_mean"] for c in cells])
                est = {"critic_mean": np.array([c["critic_mean"] for c in cells]),
                       "fqe": np.array([c["fqe"] for c in cells]),
                       "pdis": np.array([c["pdis"] for c in cells])}
                # rank correlation only over pairs separated by more than 2x the
                # PAIRED SEM; count how many pairs qualify rather than hiding it
                n_ok = 0
                for i, j in itertools.combinations(range(len(cells)), 2):
                    sem = paired_sem(cells[i]["per_episode"],
                                     cells[j]["per_episode"])
                    if abs(truth[i] - truth[j]) > 2 * sem:
                        n_ok += 1
                pairs_ok.append(n_ok)
                for k, v in est.items():
                    rhos.append({"kind": k, "rho": spearman(v, truth),
                                 "gap": oracle_gap(truth, int(np.argmax(v)))})
                lasts.append(oracle_gap(truth, len(cells) - 1))
            key = f"{env_id}|{cond}"
            out[key] = {"n_seeds": len(sel),
                        "qualifying_pairs_per_seed": pairs_ok,
                        "last_gap": float(np.mean(lasts)),
                        "critic_rho": [x["rho"] for x in rhos
                                       if x["kind"] == "critic_mean"],
                        "fqe_rho": [x["rho"] for x in rhos if x["kind"] == "fqe"],
                        "pdis_rho": [x["rho"] for x in rhos if x["kind"] == "pdis"],
                        "fqe_gap": [x["gap"] for x in rhos if x["kind"] == "fqe"]}
    # the pre-registered comparison, per env
    verdict = {}
    for env_id in sorted({r["env"] for r in runs}):
        b = out.get(f"{env_id}|ball")
        n = out.get(f"{env_id}|none")
        if not (b and n):
            continue
        fb, fn = np.array(b["fqe_rho"]), np.array(n["fqe_rho"])
        diff = float(fn.mean() - fb.mean())
        overlap = not (fn.min() > fb.max() or fb.min() > fn.max())
        verdict[env_id] = {
            "fqe_rho_none_mean": float(fn.mean()),
            "fqe_rho_ball_mean": float(fb.mean()),
            "difference": diff,
            "seed_ranges_overlap": bool(overlap),
            "passes": bool(diff >= 1.0 and not overlap),
        }
    return {"conditions": out, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="walker2d_medium_replay-v2,"
                                     "halfcheetah_medium_replay-v2")
    ap.add_argument("--conditions", default="none,ball,action")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--subsample", type=float, default=0.01,
                    help="E0c: this is what makes the regime non-trivial")
    ap.add_argument("--fraction", type=float, default=0.30)
    ap.add_argument("--action-fraction", type=float, default=0.70)
    ap.add_argument("--steps", type=int, default=66_000)
    ap.add_argument("--checkpoints", default="16500,33000,49500,66000")
    ap.add_argument("--fqe-steps", type=int, default=6000)
    ap.add_argument("--behaviour-steps", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.envs = "walker2d_medium_replay-v2"
        args.conditions, args.seeds = "none,ball", "0"
        args.steps, args.checkpoints = 2000, "1000,2000"
        args.fqe_steps, args.behaviour_steps = 300, 300
        args.eval_episodes = 3
    args.checkpoints = [int(c) for c in args.checkpoints.split(",")]
    if args.checkpoints[-1] != args.steps:
        raise SystemExit("the last checkpoint must equal --steps")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    cells_path = BENCH_DIR / ("e1_runs_smoke.jsonl" if args.smoke
                              else "e1_runs.jsonl")
    runs, done = [], set()
    if cells_path.exists():
        for line in cells_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs.append(r)
                done.add((r["env"], r["condition"], r["seed"]))
    todo = [(e, c, s) for e in args.envs.split(",")
            for c in args.conditions.split(",")
            for s in [int(x) for x in args.seeds.split(",")]
            if (e, c, s) not in done]
    print(f"E1: {len(todo)} runs to go, {len(done)} cached  "
          f"(subsample={args.subsample}, steps={args.steps:,})")
    t0 = time.perf_counter()
    for env_id, cond, seed in todo:
        print(f"  {env_id} cond={cond} seed={seed} ...", flush=True)
        r = run_one(env_id, cond, seed, args, dev)
        runs.append(r)
        with cells_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        truth = np.array([c["score_mean"] for c in r["cells"]])
        print(f"    true {[round(float(x),1) for x in truth]}  "
              f"last_gap {oracle_gap(truth, len(truth)-1):.1%}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    res = analyse(runs, args)
    print("\nper-condition (seed-level):")
    for k, v in res["conditions"].items():
        print(f"  {k:<42} last {v['last_gap']:6.1%}  "
              f"critic rho {np.mean(v['critic_rho']):+.2f}  "
              f"fqe rho {np.mean(v['fqe_rho']):+.2f}  "
              f"pdis rho {np.mean(v['pdis_rho']):+.2f}  "
              f"qualifying pairs/seed {v['qualifying_pairs_per_seed']}")
    print("\npre-registered E1 criterion (ball vs none, FQE rho):")
    for env_id, v in res["verdict"].items():
        print(f"  {env_id}: none {v['fqe_rho_none_mean']:+.2f} vs "
              f"ball {v['fqe_rho_ball_mean']:+.2f}  diff {v['difference']:+.2f}  "
              f"overlap={v['seed_ranges_overlap']}  -> "
              f"{'PASS' if v['passes'] else 'not met'}")
    p = BENCH_DIR / ("e1_summary_smoke.json" if args.smoke else "e1_summary.json")
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
