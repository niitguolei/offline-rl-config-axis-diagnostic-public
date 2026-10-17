"""B: the matched contrast on a second domain.

Round 11's central figure is a contrast with no confound: on the SAME runs, in
the SAME functional family, the algorithm's own value estimate is a
deterministic function of the compared axis while an independently fitted
estimator is not. It rested on 27 hopper runs. This checks whether it reproduces
on walker2d.

Hopper baseline (from the S5 runs): critic_mean 9/9, critic_mean_q 9/9,
fqe 5/9 -- all at exactly +/-1.00 or not.

The expected outcome is that the two critic functionals stay at 9/9 and FQE does
not. A critic below 9/9 is a counter-example and would restrict the claim to
hopper.

Usage:
    python experiments/walker2d_matched_contrast.py
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
EXPECTED = 27


def load(pattern: str) -> list[dict]:
    runs, seen = [], set()
    for name in sorted(glob.glob(str(BENCH / pattern))):
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["env"], r["seed"], r["lr"], r["tau"])
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def within_slice(runs: list[dict], key: str) -> list[float]:
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        buckets.setdefault((r["seed"], r["lr"]), []).append(r)
    out = []
    for _, sel in sorted(buckets.items()):
        if len({r["tau"] for r in sel}) < 3 or any(r.get(key) is None for r in sel):
            continue
        sel = sorted(sel, key=lambda r: r["tau"])
        a = stats.rankdata([float(r["tau"]) for r in sel])
        e = stats.rankdata([float(r[key]) for r in sel])
        out.append(float(np.corrcoef(e, a)[0, 1]))
    return out


def main() -> int:
    runs = load("e2b_runs_mw*.jsonl")
    print(f"merged {len(runs)} walker2d runs (expected {EXPECTED})")
    assert len(runs) == EXPECTED, (
        f"expected {EXPECTED}, got {len(runs)} -- check the glob "
        f"(e2b_runs_mw*, not e2b_iql_mw*) before trusting anything below")
    print(f"  envs {sorted({r['env'] for r in runs})}, "
          f"seeds {sorted({r['seed'] for r in runs})}\n")

    hopper = {"critic_mean": (9, 9), "critic_mean_q": (9, 9), "fqe": (5, 9)}
    print(f"{'estimator':<15} {'walker2d':>22} {'hopper (baseline)':>20}")
    results = {}
    for key in ("critic_mean", "critic_mean_q", "fqe"):
        cors = np.array(within_slice(runs, key))
        exact = int((np.abs(cors) > 0.999).sum())
        results[key] = {"exact": exact, "n": len(cors),
                        "range": [float(cors.min()), float(cors.max())]}
        hx, hn = hopper[key]
        print(f"{key:<15} {f'{cors.min():+.2f}..{cors.max():+.2f}  {exact}/{len(cors)}':>22} "
              f"{f'{hx}/{hn}':>20}")

    print("\n=== does the boundary reproduce? ===")
    critic_ok = all(results[k]["exact"] == results[k]["n"]
                    for k in ("critic_mean", "critic_mean_q"))
    fqe_ok = results["fqe"]["exact"] < results["fqe"]["n"]
    if critic_ok and fqe_ok:
        verdict = ("REPRODUCES: both critic functionals stay at every slice, FQE "
                   "does not. The central figure now spans two domains.")
    elif not critic_ok:
        verdict = ("COUNTER-EXAMPLE: a critic functional is below its slice count "
                   "on walker2d. The inheritance claim must be restricted to "
                   "hopper and this must be recorded as such.")
    else:
        verdict = ("PARTIAL: the critics hold but FQE also reached every slice, so "
                   "the contrast does not separate the two on this domain.")
    print(f"  {verdict}")

    out = BENCH / "walker2d_matched_contrast.json"
    out.write_text(json.dumps({"n_runs": len(runs), "results": results,
                               "hopper_baseline": hopper,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
