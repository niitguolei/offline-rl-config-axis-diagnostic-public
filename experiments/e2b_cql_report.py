"""Merge the CQL arm's shards and run the same analysis as the IQL arm.

The CQL step is launch-bound: ~578 kernels per step, 86% of the wall clock on
the CPU, GPU at 19% and 774 MB of 8 GB while 24 cores sit at 4%. So the arm is
run as several processes with disjoint seeds, each writing its own
`e2b_cql_runs_<tag>.jsonl`, and merged here.

`analyse` is imported from the IQL arm rather than reimplemented, so the
criterion for "useful" is literally the same code in both arms.

Usage:
    python experiments/e2b_cql_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from e2b_crossconfig import analyse  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def load_shards() -> tuple[list[dict], dict]:
    runs, seen, shards = [], {}, {}
    for path in sorted(BENCH_DIR.glob("e2b_cql_runs*.jsonl")):
        if "smoke" in path.name:
            continue
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["env"], r["lr"], r["min_q_weight"], r["seed"])
            if key in seen:
                continue          # a re-run of the same cell replaces nothing
            seen[key] = path.name
            runs.append(r)
            kept += 1
        shards[path.name] = kept
    return runs, shards


def main() -> int:
    runs, shards = load_shards()
    print(f"merged {len(runs)} runs")
    for name, count in shards.items():
        print(f"  {name:<34} {count:>4} new")

    envs = sorted({r["env"] for r in runs})
    configs = {(r["lr"], r["min_q_weight"]) for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    expected = len(configs) * len(seeds)
    print(f"\ngrid: {len(configs)} configs x {len(seeds)} seeds = {expected} per env")
    for env in envs:
        n = sum(1 for r in runs if r["env"] == env)
        diverged = [r for r in runs if r["env"] == env and r.get("diverged")]
        print(f"  {env:<34} {n:>4}/{expected} runs, "
              f"{len(diverged)} flagged diverged")

    res = analyse(runs)
    print("\ncross-configuration selection (higher gap = worse):")
    for env, s in res.items():
        print(f"  {env}")
        for key in ("critic_mean", "fqe", "pdis"):
            if key not in s:
                continue
            v = s[key]
            print(f"    {key:<12} gap {v['mean_gap']:6.1%}  "
                  f"random {v['mean_random_gap']:6.1%}  "
                  f"beats by {v['beats_random_by']:+6.1%}  "
                  f"SEM {v['seed_gap_sem']:5.1%}  "
                  f"-> {'USEFUL' if v['useful'] else 'no better than random'}")

    print("\nleak probe (corr of the estimator with the compared hyperparameter):")
    for env in envs:
        sub = [r for r in runs if r["env"] == env]
        w = np.array([float(r["min_q_weight"]) for r in sub])
        lr = np.array([float(r["lr"]) for r in sub])
        c = np.array([float(r["critic_mean"]) for r in sub])
        t = np.array([float(r["true_score"]) for r in sub])
        if len(sub) < 3 or c.std() == 0:
            print(f"  {env:<34} too few runs to correlate")
            continue
        print(f"  {env:<34} corr(critic,w)={np.corrcoef(c, w)[0, 1]:+.2f}  "
              f"corr(critic,lr)={np.corrcoef(c, lr)[0, 1]:+.2f}  "
              f"corr(critic,true)={np.corrcoef(c, t)[0, 1]:+.2f}")

    out = BENCH_DIR / "e2b_cql_summary.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
