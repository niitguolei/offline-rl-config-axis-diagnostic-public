"""Review Z12: how many of the project's significance claims survive correction?

Every "estimator vs random" comparison in this project is a paired t-test over
seeds (the primary criterion, plan section 16 Z10). The project ran a lot of
them -- per domain, per estimator, pooled and per axis -- and reported the ones
that came out significant. A reviewer is entitled to ask what a family-wise
correction does to those claims.

This enumerates the family, sorts by p, and reports raw, Bonferroni and Holm.
It is deliberately mechanical: the point is to see the damage, not to choose
which tests "count".

Usage:
    python experiments/audit_multiplicity.py
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


def paired_test(sub: list[dict], key: str, axis: str | None = None,
                within: str | None = None) -> dict | None:
    """Paired difference (random null minus estimator gap), clustered by seed.

    Review Z13(b): a slice is a (seed, learning-rate) pair, and slices sharing a
    seed share the data subsample. Treating them as independent inflates the
    degrees of freedom -- on three seeds it lifted df from 2 to 8 and turned a
    p of 0.1078 into 0.0124. Axis-level tests are therefore collapsed to one
    observation per seed before the t-test, and only the pooled test (which is
    already one observation per seed) is used as-is.
    """
    slices: dict = {}
    if axis is None:
        for r in sub:
            slices.setdefault(r["seed"], []).append(r)
    else:
        for r in sub:
            slices.setdefault((r["seed"], r[within]), []).append(r)
    per_unit: dict = {}
    for key_, sel in sorted(slices.items(), key=lambda kv: str(kv[0])):
        if len(sel) < 3:
            continue
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r[key]) for r in sel]))
        gap = oracle_gap(truth, pick)
        rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        # axis-level: collapse to the seed; pooled: the key is already the seed
        seed = key_[0] if isinstance(key_, tuple) else key_
        per_unit.setdefault(seed, []).append(rnd - gap)
    diffs = [float(np.mean(v)) for v in per_unit.values()]
    if len(diffs) < 2:
        return None
    d = np.array(diffs)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    if sem == 0:
        return None
    t = d.mean() / sem
    return {"beats": float(d.mean()),
            "p": float(2 * stats.t.sf(abs(t), df=len(d) - 1)), "n": len(d)}


def family() -> list[tuple[str, dict]]:
    # Glob, not the exact filename: the seed top-up wrote its runs to
    # e2b_runs_s3/s4.jsonl, and reading only the canonical file silently used
    # the old asymmetric grid (seeds 0-2 with three tau levels, seeds 3-4 with
    # two). Same trap the stratified script hit with the smoke file.
    iql = load("e2b_runs*.jsonl")
    cql = load("e2b_cql_runs*.jsonl")
    out = []
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            short = env.split("_")[0]
            for key in ("critic_mean", "fqe"):
                r = paired_test(sub, key)
                if r:
                    out.append((f"{label} pooled {short} {key}", r))
            for axis, within, tag in ((knob, "lr", f"along {knob}"),
                                      ("lr", knob, "along lr")):
                r = paired_test(sub, "critic_mean", axis=axis, within=within)
                if r:
                    out.append((f"{label} {tag} {short}", r))
    return out


def grid_report(label: str, runs: list[dict], knob: str) -> list[str]:
    """Flag per-seed configuration counts that differ within a domain.

    Review Z21: four separate bugs in this project came from an analysis
    silently reading the wrong set of runs -- a smoke file caught by a glob, a
    shard file the runner could not see, a comma-split argument, and finally an
    analysis script naming one file instead of globbing the shards. The symptom
    is always the same: an asymmetric grid whose seeds are not comparable, so
    the mean averages different selection problems. This makes that symptom
    loud instead of silent.
    """
    bad = []
    for env in sorted({r["env"] for r in runs}):
        per_seed = {}
        for r in runs:
            if r["env"] != env:
                continue
            per_seed.setdefault(r["seed"], set()).add((r["lr"], r[knob]))
        counts = {s: len(v) for s, v in per_seed.items()}
        if len(set(counts.values())) > 1:
            bad.append(f"{label} {env}: per-seed config counts {counts} differ")
    return bad


def main() -> int:
    iql_all = load("e2b_runs*.jsonl")
    cql_all = load("e2b_cql_runs*.jsonl")
    print(f"loaded IQL {len(iql_all)} runs, CQL {len(cql_all)} runs")
    problems = (grid_report("IQL", iql_all, "tau")
                + grid_report("CQL", cql_all, "min_q_weight"))
    if problems:
        print("  !! NON-UNIFORM GRID -- means below may average unlike things:")
        for p in problems:
            print(f"     {p}")
    else:
        print("  grid uniform per (arm, domain, seed)")

    tests = sorted(family(), key=lambda kv: kv[1]["p"])
    m = len(tests)
    print(f"family size = {m} primary tests (paired t over seeds)\n")
    print(f"{'test':<44} {'beats':>8} {'p':>9} {'p*N':>9} {'Holm':>6}")
    holm_open = True
    survivors = []
    for i, (name, r) in enumerate(tests):
        bonf = min(1.0, r["p"] * m)
        holm = r["p"] < 0.05 / (m - i) and holm_open
        if not holm:
            holm_open = False
        if bonf < 0.05:
            survivors.append(name)
        print(f"{name:<44} {r['beats']:+8.1%} {r['p']:9.4f} {bonf:9.4f} {str(holm):>6}")
    raw = sum(1 for _, r in tests if r["p"] < 0.05)
    print(f"\nraw p<0.05: {raw}/{m}")
    print(f"surviving Bonferroni: {len(survivors)}/{m} -> {survivors}")
    print("\nIf the confirmatory family is only the pre-registered cells "
          "(S3-1, S3-2, the core contrast), the correction is far gentler; "
          "that split has to be declared, not chosen after seeing this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
