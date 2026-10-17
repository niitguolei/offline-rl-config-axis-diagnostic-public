"""Which estimators inherit the hyperparameter they were trained with?

The paper's earlier central figure was a cross-ARM contrast (IQL/hopper along
tau against CQL/hopper along alpha), and it carries three differences at once:
the algorithm, the knob type, and the policy class. This looks for a contrast
with none of them.

The S5 runs recorded three estimators on the SAME 27 runs, and two of them are
in the same functional family -- `critic_mean_q` is min(Q1,Q2) at the policy's
mean action and `fqe` is a post-hoc fitted Q, both evaluated at the policy. So
comparing those two holds domain, data, policy, slices, axis and functional
family fixed, and varies exactly one thing: whether the compared hyperparameter
appears in the estimator's TRAINING objective.

  critic_mean    trained with tau   -> inherits it
  critic_mean_q  same critic        -> inherits it
  fqe            fitted afterwards  -> does not

Measured as the within-slice rank correlation between the estimator and tau.

Usage:
    python experiments/estimator_axis_dependence.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
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


def within_slice_corr(runs: list[dict], knob: str, key: str) -> list[float]:
    """Rank correlation between the estimator and the axis, per (seed, lr)."""
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        buckets.setdefault((r["seed"], r["lr"]), []).append(r)
    out = []
    for _, sel in sorted(buckets.items()):
        if len({r[knob] for r in sel}) < 3:
            continue
        if any(r.get(key) is None for r in sel):
            continue
        a = stats.rankdata([float(r[knob]) for r in sel])
        e = stats.rankdata([float(r[key]) for r in sel])
        out.append(float(np.corrcoef(e, a)[0, 1]))
    return out


def describe(cors: list[float]) -> str:
    if not cors:
        return "n/a"
    c = np.array(cors)
    return (f"{c.min():+.2f}..{c.max():+.2f}  exactly +/-1: "
            f"{int((np.abs(c) > 0.999).sum())}/{len(c)}")


def main() -> int:
    out: dict = {}

    print("=== matched contrast: same 27 runs, same functional family ===")
    matched = load("e2b_runs_m*.jsonl")
    print(f"  {len(matched)} runs (hopper, seeds "
          f"{sorted({r['seed'] for r in matched})})")
    matched_out = {}
    for key in ("critic_mean", "critic_mean_q", "fqe"):
        cors = within_slice_corr(matched, "tau", key)
        matched_out[key] = cors
        trained = "trained WITH tau" if key != "fqe" else "fitted AFTERWARDS"
        print(f"  {key:<15} ({trained:<17}) {describe(cors)}")
    out["matched_hopper"] = matched_out

    print("\n=== all data (functionals not matched across arms) ===")
    totals = {"critic_mean": [], "fqe": []}
    for label, pattern, knob in (("IQL", "e2b_runs*.jsonl", "tau"),
                                 ("CQL", "e2b_cql_runs*.jsonl", "min_q_weight")):
        runs = load(pattern)
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            row = {}
            for key in ("critic_mean", "fqe"):
                cors = within_slice_corr(sub, knob, key)
                row[key] = cors
                totals[key].extend(cors)
            out[f"{label}|{env}"] = row
            print(f"  {label + ' ' + env.split('_')[0]:<24} "
                  f"critic {describe(row['critic_mean']):<34} "
                  f"fqe {describe(row['fqe'])}")
    print(f"\n  TOTALS  critic {describe(totals['critic_mean'])}")
    print(f"          fqe    {describe(totals['fqe'])}")
    out["totals"] = totals

    print("\nMechanism: an estimator inherits the hyperparameter its training")
    print("objective contains. Being free of the axis is necessary, not")
    print("sufficient -- FQE is free of it and still 0 of 6 significant.")
    path = BENCH / "estimator_axis_dependence.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
