"""Recompute every number the plan and the gate board assert, from raw runs.

The documents have been edited many times, and the run files have grown under
them (the tau axis went from 2 levels to 3 on some domains but not others), so
the only way to check the claims is to recompute them from `results/benchmarks`
and compare.

The grids are asymmetric and that has to be stated rather than averaged over:

  IQL   seeds 0,1,2 have tau in {0.5, 0.7, 0.9}   (9 configs per domain)
        seeds 3,4   have tau in {0.5, 0.9}        (6 configs per domain)
  CQL   seeds 0,1,2 have alpha in {5, 10, 20} on hopper only
        everything else has alpha in {5, 10}

Averaging a 9-config selection problem with a 6-config one is not a mean of
like things -- the uniform-pick null differs -- so every block below states the
grid it used.

Usage:
    python experiments/audit_plan_claims.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orl.selection.report import oracle_gap  # noqa: E402

BENCH = ROOT / "results" / "benchmarks"


def load(pattern: str) -> list[dict]:
    runs, seen = [], set()
    for name in sorted(glob.glob(str(BENCH / pattern))):
        if "smoke" in pathlib.Path(name).name:
            continue
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["env"], r["seed"], r.get("lr"), r.get("tau"),
                   r.get("min_q_weight"))
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def cross_config(runs: list[dict], env: str, label: str) -> None:
    """Selection over whole configurations, the way `analyse` does it."""
    sub = [r for r in runs if r["env"] == env]
    seeds = sorted({r["seed"] for r in sub})
    sizes = {s: len([r for r in sub if r["seed"] == s]) for s in seeds}
    if len(set(sizes.values())) > 1:
        print(f"    !! MIXED GRID: per-seed config counts {sizes} -- a mean over "
              f"these is not a mean of like things")
    for key in ("critic_mean", "fqe"):
        gaps, rnds = [], []
        for s in seeds:
            sel = [r for r in sub if r["seed"] == s]
            truth = np.array([float(r["true_score"]) for r in sel])
            pick = int(np.argmax([float(r[key]) for r in sel]))
            gaps.append(oracle_gap(truth, pick))
            rnds.append(float(np.mean([oracle_gap(truth, i) for i in range(len(truth))])))
        gaps, rnds = np.array(gaps), np.array(rnds)
        sem = gaps.std(ddof=1) / np.sqrt(len(gaps))
        beats = rnds.mean() - gaps.mean()
        print(f"    {label:<28} {key:<11} gap {gaps.mean():6.1%}  random "
              f"{rnds.mean():6.1%}  beats {beats:+6.1%}  SEM {sem:5.1%}  "
              f"{beats/sem:+5.2f} SEM  useful={bool(beats > sem)}")


def axis_selection(runs: list[dict], env: str, axis: str, within: str) -> None:
    """Rank along one axis, holding the other fixed, >=3 candidates per slice."""
    sub = [r for r in runs if r["env"] == env]
    slices: dict[tuple, list[dict]] = {}
    for r in sub:
        slices.setdefault((r["seed"], r[within]), []).append(r)
    gaps, rnds, used = [], [], 0
    for _, sel in sorted(slices.items()):
        if len({r[axis] for r in sel}) < 3:
            continue
        sel = sorted(sel, key=lambda r: r[axis])
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r["critic_mean"]) for r in sel]))
        gaps.append(oracle_gap(truth, pick))
        rnds.append(float(np.mean([oracle_gap(truth, i) for i in range(len(truth))])))
        used += 1
    if not gaps:
        print(f"    {env:<30} along {axis:<14} no slice has 3 candidates")
        return
    gaps, rnds = np.array(gaps), np.array(rnds)
    sem = gaps.std(ddof=1) / np.sqrt(len(gaps)) if len(gaps) > 1 else float("nan")
    beats = rnds.mean() - gaps.mean()
    print(f"    {env:<30} along {axis:<14} gap {gaps.mean():6.1%}  random "
          f"{rnds.mean():6.1%}  beats {beats:+6.1%}  SEM {sem:5.1%}  "
          f"{beats/sem:+5.2f} SEM  useful={bool(beats > sem)}  (n={used} slices)")


def corr_table(runs: list[dict], axis: str) -> None:
    for env in sorted({r["env"] for r in runs}):
        sub = [r for r in runs if r["env"] == env]
        a = np.array([float(r[axis]) for r in sub])
        t = np.array([float(r["true_score"]) for r in sub])
        c = np.array([float(r["critic_mean"]) for r in sub])
        levels = sorted({float(r[axis]) for r in sub})
        means = "  ".join(f"{lv:g}={np.mean([float(r['true_score']) for r in sub if float(r[axis])==lv]):.1f}"
                          for lv in levels)
        print(f"    {env:<30} corr(critic,{axis})={np.corrcoef(c,a)[0,1]:+.2f}  "
              f"corr(critic,true)={np.corrcoef(c,t)[0,1]:+.2f}  truth {means}")


def main() -> int:
    iql = load("e2b_runs.jsonl")
    cql = load("e2b_cql_runs*.jsonl")
    print(f"IQL records {len(iql)}   CQL records {len(cql)}")

    print("\n[1] IQL, 6-config grid (tau in {0.5,0.9}), all seeds -- what the plan quotes")
    for env in sorted({r["env"] for r in iql}):
        sub = [r for r in iql if r["env"] == env and r["tau"] in (0.5, 0.9)]
        cross_config(sub, env, "tau 0.5/0.9")

    print("\n[2] IQL, 9-config grid (tau 0.5/0.7/0.9), seeds 0-2 only")
    for env in sorted({r["env"] for r in iql}):
        sub = [r for r in iql if r["env"] == env and r["seed"] in (0, 1, 2)]
        cross_config(sub, env, "tau 0.5/0.7/0.9")

    print("\n[3] IQL, the grid the summary file actually contains (mixed)")
    for env in sorted({r["env"] for r in iql}):
        cross_config(iql, env, "ALL records")

    print("\n[4] IQL axis-level, along tau (holding lr fixed)")
    for env in sorted({r["env"] for r in iql}):
        axis_selection(iql, env, "tau", "lr")

    print("\n[5] IQL axis-level, along lr (holding tau fixed)")
    for env in sorted({r["env"] for r in iql}):
        axis_selection(iql, env, "lr", "tau")

    print("\n[6] IQL corr table by tau")
    corr_table(iql, "tau")

    print("\n[7] CQL, 6-config grid (alpha 5/10), all seeds -- what the plan quotes")
    for env in sorted({r["env"] for r in cql}):
        sub = [r for r in cql if r["env"] == env and r["min_q_weight"] in (5.0, 10.0)]
        cross_config(sub, env, "alpha 5/10")

    print("\n[8] CQL, all records (hopper has alpha 20 on seeds 0-2)")
    for env in sorted({r["env"] for r in cql}):
        cross_config(cql, env, "ALL records")

    print("\n[9] CQL axis-level, along alpha (holding lr fixed)")
    for env in sorted({r["env"] for r in cql}):
        axis_selection(cql, env, "min_q_weight", "lr")

    print("\n[10] CQL corr table by alpha")
    corr_table(cql, "min_q_weight")

    print("\n[11] Simpson check, IQL hopper: corr(critic, truth) within tau strata")
    for env in ("hopper_medium_replay-v2", "walker2d_medium_replay-v2",
                "halfcheetah_medium_replay-v2"):
        sub = [r for r in iql if r["env"] == env and r["tau"] in (0.5, 0.9)]
        pooled = np.corrcoef([float(r["critic_mean"]) for r in sub],
                             [float(r["true_score"]) for r in sub])[0, 1]
        parts = []
        for lv in (0.5, 0.9):
            st = [r for r in sub if r["tau"] == lv]
            parts.append(np.corrcoef([float(r["critic_mean"]) for r in st],
                                     [float(r["true_score"]) for r in st])[0, 1])
        print(f"    {env:<30} pooled {pooled:+.2f}   within tau=0.5 {parts[0]:+.2f}"
              f"   within tau=0.9 {parts[1]:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
