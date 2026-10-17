"""The two inputs to the confound predictor, from runs already on disk.

The revised claim is that a hyperparameter confound is fatal exactly when the
estimator moves along the compared axis in the direction OPPOSITE to quality:

    fatal  <=>  corr(estimate, axis) * d(mean truth)/d(axis) < 0

Both inputs are computable from runs that already exist, with no new training,
which is what makes the predictor testable at all. This prints them per domain
and axis so the registration in the plan is written from numbers rather than
from memory.

Usage:
    python experiments/axis_direction.py --runs "results/benchmarks/e2b_runs.jsonl" --axis tau
    python experiments/axis_direction.py --runs "results/benchmarks/e2b_cql_runs*.jsonl" --axis min_q_weight
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.stdout.reconfigure(line_buffering=True)


def load(pattern: str) -> list[dict]:
    runs, seen = [], set()
    for name in sorted(glob.glob(str(ROOT / pattern))):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--axis", required=True, help="e.g. tau, min_q_weight, lr")
    ap.add_argument("--estimators", default="critic_mean,fqe")
    args = ap.parse_args()

    runs = load(args.runs)
    print(f"{len(runs)} runs, axis={args.axis}")
    for env in sorted({r["env"] for r in runs}):
        sub = [r for r in runs if r["env"] == env]
        levels = sorted({float(r[args.axis]) for r in sub})
        truth = np.array([float(r["true_score"]) for r in sub])
        axis = np.array([float(r[args.axis]) for r in sub])
        means = {lv: float(np.mean([float(r["true_score"]) for r in sub
                                    if float(r[args.axis]) == lv])) for lv in levels}
        # The direction of quality along the axis, as a least-squares slope.
        slope = float(np.polyfit(axis, truth, 1)[0]) if len(levels) > 1 else float("nan")
        print(f"\n  {env}   levels={levels}")
        print("    mean truth by level: " + "  ".join(
            f"{lv:g}={means[lv]:.1f}" for lv in levels))
        print(f"    d(truth)/d({args.axis}) = {slope:+.2f}  "
              f"({'quality falls as the axis rises' if slope < 0 else 'quality rises'})")
        for key in args.estimators.split(","):
            est = np.array([float(r[key]) for r in sub])
            if est.std() == 0 or axis.std() == 0:
                continue
            c_axis = float(np.corrcoef(est, axis)[0, 1])
            c_true = float(np.corrcoef(est, truth)[0, 1])
            # sign of the product decides the prediction
            product = c_axis * slope
            verdict = ("FATAL" if product < 0 else "survives")
            print(f"    {key:<11} corr(est,axis)={c_axis:+.2f}  "
                  f"corr(est,truth)={c_true:+.2f}  "
                  f"product={product:+.3f}  -> predicted {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
