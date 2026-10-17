"""The axis-level tests, done at the level of independent units.

Review Z13(b): the earlier axis-level analyses treated 9 slices as 9 independent
observations. A slice is a (seed, learning-rate) pair, and slices sharing a seed
share the data subsample and the training seed, so they are not independent. The
slice-level test inflates the degrees of freedom from 2 to 8, which is what
turned S3-1's p=0.0124 into something that looked confirmatory.

This collapses each seed's slices to one number first -- the conservative and
correct treatment -- and then runs the paired t-test on the seeds. It reports
both readings side by side so the inflation is visible rather than described.

It also re-runs the mechanism check: within every slice, the rank correlation
between the estimate and the compared axis is exactly +/-1.00. That is an
enumeration over slices, not an inference, so it needs neither independence nor
a p-value, and it is expected to survive the seed top-up unchanged.

Usage:
    python experiments/clustered_axis_tests.py
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

# The two pre-registered cells (plan section 15.3) and the two registered-rule
# extensions (S4), plus the two cells that survived the full-family correction.
CELLS = (
    ("S3-1 CQL/hopper along alpha", "CQL", "hopper", "min_q_weight", "lr", True),
    ("S3-2 IQL/hopper along tau", "IQL", "hopper", "tau", "lr", True),
    ("S4-a IQL/walker2d along tau", "IQL", "walker2d", "tau", "lr", False),
    ("S4-b IQL/halfcheetah along tau", "IQL", "halfcheetah", "tau", "lr", False),
    ("CQL/walker2d along lr", "CQL", "walker2d", "lr", "min_q_weight", False),
    ("CQL/hopper along lr", "CQL", "hopper", "lr", "min_q_weight", False),
)


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


def slice_diffs(sub: list[dict], within: str) -> list[tuple[int, float]]:
    """One paired difference per (seed, within) slice."""
    buckets: dict[tuple, list[dict]] = {}
    for r in sub:
        buckets.setdefault((r["seed"], r[within]), []).append(r)
    out = []
    for (seed, _), sel in sorted(buckets.items()):
        if len(sel) < 3:
            continue
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r["critic_mean"]) for r in sel]))
        gap = oracle_gap(truth, pick)
        rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        out.append((seed, rnd - gap))
    return out


def test(vals) -> tuple[float, float, int]:
    v = np.array(vals)
    n = len(v)
    sem = v.std(ddof=1) / np.sqrt(n)
    if sem == 0:
        return float(v.mean()), 0.0, n
    t = v.mean() / sem
    return float(v.mean()), float(2 * stats.t.sf(abs(t), df=n - 1)), n


def main() -> int:
    iql = load("e2b_runs*.jsonl")
    cql = load("e2b_cql_runs*.jsonl")
    print(f"merged IQL {len(iql)} runs, CQL {len(cql)} runs")
    print(f"  IQL hopper seeds {sorted({r['seed'] for r in iql if r['env'].startswith('hopper')})}")
    print(f"  CQL hopper seeds {sorted({r['seed'] for r in cql if r['env'].startswith('hopper')})}\n")

    rows = []
    print(f"{'cell':<32} {'n_seed':>6} {'mean':>8} {'slice p':>9} {'clustered p':>12}")
    for name, arm, dom, axis, within, confirmatory in CELLS:
        runs = iql if arm == "IQL" else cql
        sub = [r for r in runs if r["env"].startswith(dom)]
        sd = slice_diffs(sub, within)
        if len(sd) < 2:
            print(f"{name:<32}  (not enough slices)")
            continue
        _, p_slice, _ = test([v for _, v in sd])
        by_seed: dict[int, list[float]] = {}
        for seed, v in sd:
            by_seed.setdefault(seed, []).append(v)
        mean, p_clust, n_seed = test([float(np.mean(v)) for v in by_seed.values()])
        rows.append({"cell": name, "n_seeds": n_seed, "mean": mean,
                     "p_slice": p_slice, "p_clustered": p_clust,
                     "confirmatory": confirmatory})
        print(f"{name:<32} {n_seed:>6} {mean:+8.1%} {p_slice:9.4f} {p_clust:12.4f}")

    print("\n=== against the two correction levels (on clustered p) ===")
    n_conf = sum(1 for r in rows if r["confirmatory"])
    n_full = 22   # the family enumerated in plan section 18 Z12
    print(f"confirmatory family = {n_conf} tests -> alpha = {0.05/n_conf:.4f}")
    print(f"full family         = {n_full} tests -> alpha = {0.05/n_full:.5f}\n")
    print(f"{'cell':<32} {'clustered p':>12} {'conf?':>7} {'full?':>7}")
    for r in sorted(rows, key=lambda x: x["p_clustered"]):
        conf = "PASS" if (r["confirmatory"] and r["p_clustered"] < 0.05 / n_conf) else "-"
        full = "PASS" if r["p_clustered"] < 0.05 / n_full else "-"
        print(f"{r['cell']:<32} {r['p_clustered']:12.4f} {conf:>7} {full:>7}")

    print("\n=== mechanism check: within-slice rank corr(estimate, axis) ===")
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        cors = []
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            buckets: dict[tuple, list[dict]] = {}
            for r in sub:
                buckets.setdefault((r["seed"], r["lr"]), []).append(r)
            for _, sel in buckets.items():
                if len({r[knob] for r in sel}) < 3:
                    continue
                sel = sorted(sel, key=lambda r: r[knob])
                cors.append(float(np.corrcoef(
                    stats.rankdata([float(r["critic_mean"]) for r in sel]),
                    stats.rankdata([float(r[knob]) for r in sel]))[0, 1]))
        cors = np.array(cors)
        print(f"  {label}: {len(cors)} slices, exactly +/-1.00: "
              f"{int((np.abs(cors) > 0.999).sum())}/{len(cors)}")

    out = BENCH / "clustered_axis_tests.json"
    out.write_text(json.dumps(
        {"note": "clustered = one observation per seed, the correct treatment; "
                 "slice-level p is reported only to show the inflation",
         "confirmatory_family_size": n_conf, "full_family_size": n_full,
         "cells": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
