"""T0 acceptance smoke for path A: the selection chain end to end.

v3 defines T0's deliverable as "the evaluation chain smokes end to end". This
script is that chain, and nothing more:

    train one policy with checkpoints
      -> at each checkpoint, estimate its value by several OPE estimators
      -> also measure its TRUE return (the oracle, unavailable in a real
         offline setting but exactly what makes path A's claims falsifiable)
      -> select a checkpoint by each estimator
      -> report each estimator's oracle gap and its rank correlation with truth

That last pair is path A's headline quantity: not "which selector wins" but
"under what data conditions does selection stop tracking the truth".

What is deliberately NOT here yet (T1/T2 work): typed gaps (state vs action),
reduced-size datasets, importance-sampling estimators, and any comparison to
published selectors. This script exists to prove the plumbing.

    python experiments/run_selection_chain.py --smoke
    python experiments/run_selection_chain.py --env halfcheetah_medium_replay-v2
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

from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.methods.harness import episode_lookup, evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR, parse_env_id  # noqa: E402
from orl.record.logger import RunGuard  # noqa: E402
from orl.selection.ope import (  # noqa: E402
    critic_mean_value,
    fitted_q_evaluation,
    spearman,
)

sys.stdout.reconfigure(line_buffering=True)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def oracle_gap(true_scores: np.ndarray, selected: int) -> dict:
    """1 - selected/oracle, on the D4RL normalised score."""
    best = int(np.argmax(true_scores))
    denom = abs(float(true_scores[best]))
    if denom < 1e-9:
        return {"oracle_index": best, "gap": float("nan")}
    return {"oracle_index": best,
            "gap": float(1.0 - true_scores[selected] / true_scores[best])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--checkpoints", default="10000,20000,30000,40000")
    ap.add_argument("--fqe-steps", type=int, default=4000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="output name; defaults vary with --smoke")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.checkpoints = 4000, "1000,2000,3000,4000"
        args.fqe_steps, args.eval_episodes = 300, 2
    cps = [int(c) for c in args.checkpoints.split(",")]
    if cps[-1] != args.steps:
        raise SystemExit("the last checkpoint must equal --steps")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep = episode_lookup(d.env_id)
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    print(f"env={d.env_id}  device={dev}  steps={args.steps:,}  cps={cps}")
    print(f"start states = {len(starts)} episodes;  "
          f"reward scaled mean {d.scaled_rew_mean:.4f}")

    O, A = to_t(d.obs, dev), to_t(d.act, dev)
    R, NO, TE = to_t(d.rew, dev), to_t(d.next_obs, dev), to_t(d.terminals, dev)
    start_states = O[to_t(starts, dev).long()]

    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.steps, tau=0.7, beta=3.0, alpha=0.005, discount=0.99,
        episode_lookup=ep, coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(args.seed)
    snaps, done = [], 0
    for cp in cps:
        while done < cp:
            j = np.sort(rng.integers(0, len(O), 256))
            agent.update(O[j], A[j], NO[j], R[j], TE[j])   # verbatim reference
            done += 1
        snaps.append({"step": cp, "policy": copy.deepcopy(agent.policy),
                      "vf": copy.deepcopy(agent.vf)})
        print(f"  checkpoint {cp:,} saved ({time.perf_counter()-t0:.0f}s)")

    rows = []
    for s in snaps:
        true = evaluate(s["policy"], d.domain, d.normer,
                        n_episodes=args.eval_episodes)
        fqe = fitted_q_evaluation(
            s["policy"], O, NO, R, TE, start_states,
            gamma=0.99, steps=args.fqe_steps, seed=args.seed, device=dev)
        cri = critic_mean_value(s["vf"], start_states)
        rows.append({"step": s["step"], "true_raw": true["raw_mean"],
                     "true_score": true["score_mean"],
                     "true_std": true["raw_std"],
                     "ope_fqe": fqe["value"], "ope_fqe_std": fqe["value_std"],
                     "ope_critic": cri,
                     "fqe_loss_trace": fqe["q_loss_trace"]})
        print(f"  step {s['step']:>7,}  true={true['score_mean']:7.2f}  "
              f"fqe={fqe['value']:8.2f}  critic={cri:8.2f}")

    truth = np.array([r["true_score"] for r in rows])
    print(f"\n{'step':>9}{'true':>9}{'fqe':>10}{'critic':>10}")
    for r in rows:
        print(f"{r['step']:>9,}{r['true_score']:>9.2f}"
              f"{r['ope_fqe']:>10.2f}{r['ope_critic']:>10.2f}")

    print("\nselection quality (higher gap = worse selection):")
    summary = {}
    for name, key in (("ope_fqe", "ope_fqe"), ("ope_critic", "ope_critic")):
        est = np.array([r[key] for r in rows])
        picked = int(np.argmax(est))
        g = oracle_gap(truth, picked)
        rho = spearman(est, truth)
        summary[name] = {"picked_step": rows[picked]["step"],
                         "picked_true": float(truth[picked]),
                         "oracle_step": rows[g["oracle_index"]]["step"],
                         "oracle_true": float(truth[g["oracle_index"]]),
                         "gap": g["gap"], "spearman": rho}
        print(f"  {name:<12} picks step {rows[picked]['step']:>7,} "
              f"(true {truth[picked]:6.2f})  oracle {rows[g['oracle_index']]['step']:>7,} "
              f"({truth[g['oracle_index']]:6.2f})  gap {g['gap']:+.3f}  "
              f"spearman {rho:+.2f}")

    domain, tier = parse_env_id(d.env_id)
    with RunGuard(run_id=f"selchain_{d.env_id}_s{args.seed}",
                  method="iql+fixed_n1", dataset=d.env_id, domain=domain,
                  tier=tier, seed=args.seed, stage="T0",
                  score_norm=float(truth[-1]),
                  notes="T0 selection-chain smoke"):
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        name = args.tag or ("selection_chain_smoke" if args.smoke
                            else "selection_chain")
        p = BENCH_DIR / f"{name}.json"
        out = {"env": d.env_id, "steps": args.steps, "checkpoints": cps,
               "fqe_steps": args.fqe_steps, "eval_episodes": args.eval_episodes,
               "rows": rows, "selection": summary,
               "elapsed_min": (time.perf_counter() - t0) / 60.0}
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
