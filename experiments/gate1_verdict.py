"""T5 — the five-branch GATE 1 verdict.

Reads the calibration run (n=5) plus the trained runs, computes per-bucket
bias-vs-n, and applies the five pre-registered branches. The branches were
fixed BEFORE any result was seen, precisely so the outcome cannot be chosen
after the fact.

    python experiments/gate1_verdict.py
"""
from __future__ import annotations

import json
import argparse
import sys

import numpy as np

from orl.paths import BENCH_DIR

sys.stdout.reconfigure(line_buffering=True)

MIN_PER_BUCKET = 1000


def slope_endpoints(records: list[dict], bucket: str) -> dict | None:
    """Per-bucket change from n=1 to n=10, paired across seeds."""
    by = {}
    for r in records:
        by.setdefault(r["n_step"], {})[r["seed"]] = r["buckets"][bucket]
    lo = [b["signed_bias_mean"] for _, b in sorted(by.get(1, {}).items())]
    hi = [b["signed_bias_mean"] for _, b in sorted(by.get(10, {}).items())]
    if not lo or not hi:
        return None
    d = [h - l for l, h in zip(lo, hi)]
    return {"n1_mean": float(np.mean(lo)), "n10_mean": float(np.mean(hi)),
            "delta": float(np.mean(d)), "n1": lo, "n10": hi}


def endpoint_seed_agreement(records: list[dict], bucket: str) -> dict:
    """How many seeds move the same way from n=1 to n=10."""
    by = {}
    for r in records:
        by.setdefault(r["n_step"], {})[r["seed"]] = r["buckets"][bucket]["signed_bias_mean"]
    seeds = sorted(set(by.get(1, {})) & set(by.get(10, {})))
    deltas = [by[10][s] - by[1][s] for s in seeds]
    return {"seeds": seeds, "deltas": [float(d) for d in deltas],
            "n_positive": sum(d > 0 for d in deltas),
            "n_negative": sum(d < 0 for d in deltas)}
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="gate1_runs.json",
                    help="records file to judge (gate1_smoke.json to rehearse)")
    args = ap.parse_args()
    cal_p = BENCH_DIR / "calibration.json"
    run_p = BENCH_DIR / args.runs
    if not run_p.exists():
        print(f"{args.runs} missing -- run experiments/gate1.py first")
        return 2

    # A rehearsal on smoke data must not overwrite the real verdict.
    out_name = ("gate1_verdict.json" if run_p.name == "gate1_runs.json"
                else run_p.stem + "_verdict.json")
    if args.runs != "gate1_runs.json":
        cal_p = BENCH_DIR / "no_such_calibration.json"   # skip the reuse line

    meta = json.loads(run_p.read_text(encoding="utf-8"))
    runs = list(meta["records"])
    if cal_p.exists():
        cal = json.loads(cal_p.read_text(encoding="utf-8"))
        last = cal["records"][-1]
        runs.append({"n_step": cal["n_step"], "seed": cal["seed"],
                     "buckets": last["buckets"], "policy_raw": last["policy_raw"],
                     "from": "calibration"})

    print(f"runs: {len(runs)}   steps: {meta['steps']:,}")
    ns = sorted({r["n_step"] for r in runs})
    print("\nper-bucket signed bias by n:")
    print(f"{'n':>4}" + "".join(f"{'bucket ' + b:>24}" for b in ("0", "1", "2")))
    for n in ns:
        sel = [r for r in runs if r["n_step"] == n]
        row = f"{n:>4}"
        for b in ("0", "1", "2"):
            v = np.mean([r["buckets"][b]["signed_bias_mean"] for r in sel])
            s = np.mean([r["buckets"][b]["signed_bias_sem"] for r in sel])
            row += f"{v:>+12.3f}+-{s:.3f}".rjust(24)
        print(row)

    slopes = {b: slope_endpoints(runs, b) for b in ("0", "1", "2")}
    print("\nslope n=1 -> n=10 (signed bias):")
    for b, v in slopes.items():
        if v:
            print(f"  bucket {b}: {v['n1_mean']:+.3f} -> {v['n10_mean']:+.3f}"
                  f"   delta {v['delta']:+.3f}")

    agree = {b: endpoint_seed_agreement(runs, b) for b in ("0", "1", "2")}
    print("\nendpoint seed agreement:")
    for b, a in agree.items():
        print(f"  bucket {b}: {a['n_positive']} up / {a['n_negative']} down"
              f"   deltas={[round(d, 3) for d in a['deltas']]}")

    thin = sum(1 for r in runs for b in ("0", "1", "2")
               if r["buckets"][b]["n_used"] < MIN_PER_BUCKET)
    d0 = slopes["0"]["delta"] if slopes["0"] else None
    d1 = slopes["1"]["delta"] if slopes["1"] else None

    print("\n" + "=" * 62)
    if thin:
        verdict = f"INSUFFICIENT DATA ({thin} bucket-runs below {MIN_PER_BUCKET})"
    elif d0 is None or agree["0"]["n_positive"] < 2:
        verdict = "INSUFFICIENT EVIDENCE -- endpoint slopes disagree across seeds"
    elif d0 > 0 and (d1 is None or d1 <= 0):
        verdict = "BRANCH 1 PASS: low-coverage rises, others flat"
    elif d0 > 0 and d1 > 0 and d0 > d1:
        verdict = "BRANCH 2 PARTIAL: all rise, low-coverage rises more"
    elif d0 > 0 and d1 > 0 and d0 <= d1:
        verdict = "BRANCH 3 COVERAGE NOT THE MODERATOR (equal rises)"
    elif d0 < 0:
        verdict = "BRANCH 5 REVERSED: critic under-estimates at low coverage"
    else:
        verdict = "BRANCH 4 n-dependence absent"
    print(f"VERDICT: {verdict}")
    print("=" * 62)

    p = BENCH_DIR / out_name
    p.write_text(json.dumps(
        {"verdict": verdict, "steps": meta["steps"], "n_runs": len(runs),
         "slopes": {b: (v["delta"] if v else None) for b, v in slopes.items()},
         "seed_agreement": agree}, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
