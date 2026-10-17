"""Round 12A: five-level tau grid mechanism check.

The pre-registered design is in docs/第十二轮_分类表.md section 7.  This
script merges the S5 three-level records with the new a0/a1/a2 shards, asserts
the expected 3 x 5 x 3 grid, and computes the within-slice rank correlation
between each estimator and tau.

Usage:
    python experiments/round12a_five_level.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import stats


ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmarks"
ENV = "hopper_medium_replay-v2"
SEEDS = (0, 1, 2)
LRS = (1e-4, 3e-4, 1e-3)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
ESTIMATORS = ("critic_mean", "critic_mean_q", "fqe")


def _same_float(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-12


def load_runs() -> list[dict]:
    paths = [BENCH / f"e2b_runs_m{seed}.jsonl" for seed in SEEDS]
    paths += [BENCH / f"e2b_runs_a{seed}.jsonl" for seed in SEEDS]
    runs: list[dict] = []
    seen: set[tuple] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["env"], row["seed"], float(row["lr"]), float(row["tau"]))
            if key in seen:
                continue
            seen.add(key)
            runs.append(row)

    selected = [
        row for row in runs
        if row["env"] == ENV
        and int(row["seed"]) in SEEDS
        and any(_same_float(row["lr"], lr) for lr in LRS)
        and any(_same_float(row["tau"], tau) for tau in TAUS)
    ]
    if len(selected) != 45:
        raise AssertionError(f"expected 45 runs, found {len(selected)}")

    slices: dict[tuple[int, float], list[dict]] = {}
    for row in selected:
        slices.setdefault((int(row["seed"]), float(row["lr"])), []).append(row)
    if len(slices) != 9:
        raise AssertionError(f"expected 9 slices, found {len(slices)}")
    for key, rows in sorted(slices.items()):
        if len(rows) != 5:
            raise AssertionError(f"slice {key} has {len(rows)} rows, expected 5")
        got = sorted(float(row["tau"]) for row in rows)
        if not all(_same_float(a, b) for a, b in zip(got, TAUS)):
            raise AssertionError(f"slice {key} has taus {got}, expected {TAUS}")
        for estimator in ESTIMATORS:
            if any(row.get(estimator) is None for row in rows):
                raise AssertionError(f"slice {key} is missing {estimator}")
    return selected


def within_slice_correlations(runs: list[dict], estimator: str) -> list[float]:
    slices: dict[tuple[int, float], list[dict]] = {}
    for row in runs:
        slices.setdefault((int(row["seed"]), float(row["lr"])), []).append(row)

    correlations = []
    for key, rows in sorted(slices.items()):
        rows = sorted(rows, key=lambda row: float(row["tau"]))
        tau_rank = stats.rankdata([float(row["tau"]) for row in rows])
        est_rank = stats.rankdata([float(row[estimator]) for row in rows])
        corr = float(np.corrcoef(tau_rank, est_rank)[0, 1])
        correlations.append(corr)
    return correlations


def main() -> int:
    runs = load_runs()
    result: dict = {
        "env": ENV,
        "n_runs": len(runs),
        "n_slices": 9,
        "seeds": list(SEEDS),
        "lrs": list(LRS),
        "taus": list(TAUS),
        "estimators": {},
    }

    for estimator in ESTIMATORS:
        correlations = within_slice_correlations(runs, estimator)
        exact = [abs(value) >= 1.0 - 1e-9 for value in correlations]
        result["estimators"][estimator] = {
            "slice_correlations": correlations,
            "exact_pm1": int(sum(exact)),
            "n_slices": len(correlations),
            "all_positive": bool(all(value > 0 for value in correlations)),
        }

    q1 = result["estimators"]["critic_mean"]["exact_pm1"]
    q2 = result["estimators"]["critic_mean_q"]["exact_pm1"]
    q3 = result["estimators"]["fqe"]["exact_pm1"]
    if q1 == 9 and q2 == 9 and q3 < 9:
        verdict = "REPRODUCES_FIVE_LEVEL"
    elif q1 >= 7 and q2 >= 7 and q3 < 9:
        verdict = "PARTIAL_FIVE_LEVEL"
    else:
        verdict = "COARSE_GRID_ARTIFACT"
    result["q1_critic_mean_exact"] = q1
    result["q2_critic_mean_q_exact"] = q2
    result["q3_fqe_exact"] = q3
    result["verdict"] = verdict

    print(f"{ENV}: {len(runs)} runs, 9 slices, 5 tau levels")
    for estimator in ESTIMATORS:
        row = result["estimators"][estimator]
        values = ", ".join(f"{value:+.6f}" for value in row["slice_correlations"])
        print(
            f"  {estimator:<15} exact +/-1 {row['exact_pm1']}/9  "
            f"all-positive={row['all_positive']}  [{values}]"
        )
    print(
        f"Q1={q1}/9  Q2={q2}/9  Q3={q3}/9  "
        f"verdict={verdict}"
    )

    out = BENCH / "round12a_five_level.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
