"""Two robustness checks a referee will ask for by name.

1. DIVERGENCE THRESHOLD. Runs are flagged when |critic| / |FQE| exceeds 20, and
   that number was chosen by hand. Does the project's strongest surviving cell
   depend on it? Counts per domain at 10x / 20x / 50x, then the walker2d
   lr-axis test re-run with each exclusion.

2. TRIVIAL CONTROL. Every comparison so far is against a uniform pick, which is
   a weak null. The obvious stronger baseline is "sort by the axis being
   compared" -- and if that reproduces the critic exactly, then "the critic
   works" is the same statement as "this axis happens to align with quality",
   which is the paper's point rather than an embarrassment.

Usage:
    python experiments/robustness_checks.py
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
THRESHOLDS = (10.0, 20.0, 50.0)


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


def ratio(r: dict) -> float:
    return abs(float(r["critic_mean"])) / max(abs(float(r["fqe"])), 1e-9)


def axis_test(runs: list[dict], dom: str, axis: str, within: str,
              exclude_over: float | None = None) -> tuple[float, float, int]:
    sub = [r for r in runs if r["env"].startswith(dom)]
    if exclude_over is not None:
        sub = [r for r in sub if ratio(r) < exclude_over]
    buckets: dict[tuple, list[dict]] = {}
    for r in sub:
        buckets.setdefault((r["seed"], r[within]), []).append(r)
    per_seed: dict[int, list[float]] = {}
    for (seed, _), sel in buckets.items():
        if len({r[axis] for r in sel}) < 3:
            continue
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r["critic_mean"]) for r in sel]))
        gap = oracle_gap(truth, pick)
        rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        per_seed.setdefault(seed, []).append(rnd - gap)
    d = np.array([float(np.mean(v)) for v in per_seed.values()])
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem
    return float(d.mean()), float(2 * stats.t.sf(abs(t), df=len(d) - 1)), len(d)


def trivial_control(runs: list[dict], knob: str) -> tuple[int, int]:
    """Does 'sort by the axis, sign-matched' reproduce the critic's pick?"""
    same = total = 0
    for env in sorted({r["env"] for r in runs}):
        sub = [r for r in runs if r["env"] == env]
        buckets: dict[tuple, list[dict]] = {}
        for r in sub:
            buckets.setdefault((r["seed"], r["lr"]), []).append(r)
        for _, sel in buckets.items():
            if len({r[knob] for r in sel}) < 3:
                continue
            c = np.array([float(r["critic_mean"]) for r in sel])
            k = np.array([float(r[knob]) for r in sel])
            sign = 1.0 if np.corrcoef(c, k)[0, 1] >= 0 else -1.0
            total += 1
            same += int(int(np.argmax(sign * k)) == int(np.argmax(c)))
    return same, total


def main() -> int:
    cql = load("e2b_cql_runs*.jsonl")
    iql = load("e2b_runs*.jsonl")

    print("=== 1. divergence threshold: flagged counts ===")
    print(f"{'domain':<32} {'runs':>5} "
          + " ".join(f"{int(t):>3}x" for t in THRESHOLDS) + f" {'max':>8}")
    for env in sorted({r["env"] for r in cql}):
        sub = [r for r in cql if r["env"] == env]
        rr = np.array([ratio(r) for r in sub])
        print(f"{env:<32} {len(sub):>5} "
              + " ".join(f"{int((rr >= t).sum()):>4}" for t in THRESHOLDS)
              + f" {rr.max():>8.1f}")

    print("\n  does the strongest surviving cell depend on it?")
    print(f"  {'exclusion':<22} {'mean':>8} {'clustered p':>12} {'seeds':>6}")
    rows = []
    for thr in (None, *THRESHOLDS):
        m, p, n = axis_test(cql, "walker2d", "lr", "min_q_weight", exclude_over=thr)
        label = "none" if thr is None else f"exclude ratio >= {int(thr)}"
        rows.append({"exclusion": label, "mean": m, "p": p, "n_seeds": n})
        print(f"  {label:<22} {m:+8.1%} {p:12.4f} {n:>6}")

    print("\n=== 2. trivial control: sort by the compared axis ===")
    out_trivial = {}
    for label, runs, knob in (("CQL", cql, "min_q_weight"), ("IQL", iql, "tau")):
        same, total = trivial_control(runs, knob)
        out_trivial[label] = {"same": same, "total": total}
        print(f"  {label}: the axis-extreme control reproduces the critic in "
              f"{same}/{total} slices")

    out = BENCH / "robustness_checks.json"
    out.write_text(json.dumps(
        {"divergence_thresholds": list(THRESHOLDS),
         "walker2d_lr_axis_vs_exclusion": rows,
         "trivial_control": out_trivial}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
