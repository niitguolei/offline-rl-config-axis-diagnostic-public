"""Review Z13: what could this design actually have detected?

Every significance claim here is a paired t-test over seeds. With 3 or 5 seeds
the design is weak, and "we found nothing in the IQL arm" is not the same as
"there is nothing there" unless the study could have seen an effect of the size
that matters. This computes the minimum detectable effect (MDE) per cell so the
paper can state which absences are informative.

MDE is approximated the standard way for a one-sample t-test,

    MDE = (t_{1-a/2, n-1} + t_{1-b, n-1}) * SEM

which uses t quantiles rather than the exact noncentral t. The approximation is
conservative for small n and is labelled as an approximation wherever it is
quoted.

Usage:
    python experiments/power_analysis.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orl.selection.report import oracle_gap  # noqa: E402

BENCH = ROOT / "results" / "benchmarks"
ALPHA = 0.05
POWER = 0.80


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


def per_seed_diffs(sub: list[dict], key: str) -> np.ndarray:
    out = []
    for seed in sorted({r["seed"] for r in sub}):
        sel = [r for r in sub if r["seed"] == seed]
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r[key]) for r in sel]))
        gap = oracle_gap(truth, pick)
        rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        out.append(rnd - gap)
    return np.array(out)


def mde(sem: float, n: int) -> float:
    df = n - 1
    return float((stats.t.ppf(1 - ALPHA / 2, df)
                  + stats.t.ppf(POWER, df)) * sem)


def main() -> int:
    # Glob, not the exact filename: the seed top-up wrote to e2b_runs_s3/s4.
    iql = load("e2b_runs*.jsonl")
    cql = load("e2b_cql_runs*.jsonl")
    print(f"MDE at alpha={ALPHA}, power={POWER} (approximation)\n")
    print(f"{'arm / domain':<34} {'n':>2} {'SEM':>7} {'MDE':>8} {'observed':>9} {'verdict':>22}")
    rows = []
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            d = per_seed_diffs(sub, "critic_mean")
            n = len(d)
            sem = d.std(ddof=1) / np.sqrt(n)
            m = mde(sem, n)
            obs = d.mean()
            if abs(obs) >= m:
                verdict = "detectable at this design"
            else:
                verdict = "below MDE -> not testable"
            rows.append((f"{label} {env.split('_')[0]}", n, sem, m, obs, verdict))
            print(f"{label + ' ' + env.split('_')[0]:<34} {n:>2} {sem:7.1%} "
                  f"{m:8.1%} {obs:+9.1%} {verdict:>22}")

    print("\n--- what this means for the absences ---")
    for name, n, sem, m, obs, verdict in rows:
        if abs(obs) < m:
            print(f"  {name}: observed {obs:+.1%} is smaller than the {m:.0%} the "
                  f"design could detect -> report as 'not detected at this power', "
                  f"NOT as 'no effect'")

    # Review Z14: the controlled contrast lives on one domain. Before proposing a
    # second one, ask what it would cost. With the per-seed spread fixed, the
    # required n for the observed effect is (t_a/2 + t_b)^2 * (sd/effect)^2.
    print("\n--- Z14: could a second domain support the same contrast? ---")
    print(f"{'cell':<34} {'observed':>9} {'sd':>7} {'n needed':>9}")
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            buckets: dict[tuple, list[dict]] = {}
            for r in sub:
                buckets.setdefault((r["seed"], r["lr"]), []).append(r)
            per_seed: dict[int, list[float]] = {}
            for (seed, _), sel in buckets.items():
                if len({r[knob] for r in sel}) < 3:
                    continue
                truth = np.array([float(r["true_score"]) for r in sel])
                pick = int(np.argmax([float(r["critic_mean"]) for r in sel]))
                gap = oracle_gap(truth, pick)
                rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
                per_seed.setdefault(seed, []).append(rnd - gap)
            d = np.array([float(np.mean(v)) for v in per_seed.values()])
            if len(d) < 2:
                continue
            sd = d.std(ddof=1)
            effect = abs(d.mean())
            if effect == 0:
                need = float("inf")
            else:
                # iterate: t quantiles depend on n
                need = 3.0
                for _ in range(50):
                    df = max(int(np.ceil(need)) - 1, 1)
                    f = (stats.t.ppf(1 - ALPHA / 2, df)
                         + stats.t.ppf(POWER, df)) ** 2 * (sd / effect) ** 2
                    need = max(f, 2.0)
            shown = "infeasible" if need > 50 else f"{need:.1f}"
            print(f"{label + ' ' + env.split('_')[0]:<34} {d.mean():+9.1%} "
                  f"{sd:7.1%} {shown:>9}")

    out = BENCH / "power_analysis.json"
    out.write_text(json.dumps(
        {"alpha": ALPHA, "power": POWER, "note": "MDE uses t quantiles, not the "
         "exact noncentral t; conservative for small n",
         "cells": [{"cell": r[0], "n_seeds": r[1], "sem": r[2], "mde": r[3],
                    "observed": r[4]} for r in rows]}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
