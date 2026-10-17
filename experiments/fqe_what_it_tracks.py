"""If FQE does not inherit the axis, what does it track?

The surviving boundary from round 11 is: the algorithm's OWN value estimate is a
deterministic function of the configuration it was trained with (57 of 57 slices
at exactly +/-1.00), while an independently fitted estimator is not (FQE, 24 of
48). That raises the obvious follow-up -- if FQE is free of the axis, why is it
not useful either (0 of 6 significant)?

Three candidate answers, all measurable without new runs:

  1. it tracks a DIFFERENT axis of the configuration,
  2. it tracks nothing in the configuration (it is noise across configs),
  3. it does not respond to the policy at all, so it cannot distinguish
     configurations whose policies differ.

The third is the one the earlier within-run work suggested (FQE returned about
the same value for a policy scoring -0.5 and one scoring 65.1). This quantifies
it: the spread of FQE across configurations, divided by the spread of the truth
across the same configurations. A ratio near zero means FQE barely moves when
the policy changes.

Usage:
    python experiments/fqe_what_it_tracks.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np

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
                   r.get("min_q_weight"), r.get("alpha"))
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def design(values: list, levels: list) -> np.ndarray:
    cols = [np.ones(len(values))]
    for lv in levels[1:]:
        cols.append(np.array([1.0 if v == lv else 0.0 for v in values]))
    return np.column_stack(cols)


def r2(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")


def main() -> int:
    out = {}
    print("What each estimator tracks, per cell")
    print(f"{'arm / domain':<30} {'estimator':<12} {'knob R2':>8} {'lr R2':>7} "
          f"{'seed R2':>8} {'corr true':>10} {'spread/truth':>13}")
    for arm, pattern, knob in (
            ("IQL", "e2b_runs*.jsonl", "tau"),
            ("CQL", "e2b_cql_runs*.jsonl", "min_q_weight"),
            ("TD3+BC", "e2b_td3bc_runs_t*.jsonl", "alpha")):
        runs = load(pattern)
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            truth = np.array([float(r["true_score"]) for r in sub])
            for key in ("critic_mean", "fqe"):
                y = np.array([float(r[key]) for r in sub])
                kn = [r[knob] for r in sub]
                lr = [r["lr"] for r in sub]
                sd = [r["seed"] for r in sub]
                rk = r2(y, design(kn, sorted(set(kn))))
                rl = r2(y, design(lr, sorted(set(lr))))
                rs = r2(y, design(sd, sorted(set(sd))))
                ct = float(np.corrcoef(y, truth)[0, 1]) if y.std() else float("nan")
                spread = float(y.std() / truth.std()) if truth.std() else float("nan")
                out[f"{arm}|{env}|{key}"] = {
                    "knob_r2": rk, "lr_r2": rl, "seed_r2": rs,
                    "corr_true": ct, "spread_over_truth": spread, "n": len(sub)}
                print(f"{arm + ' ' + env.split('_')[0]:<30} {key:<12} {rk:>8.2f} "
                      f"{rl:>7.2f} {rs:>8.2f} {ct:>10.2f} {spread:>13.2f}")

    print("\nReading:")
    print("  knob R2 high  -> the estimator is the axis")
    print("  spread/truth near 0 -> the estimator barely moves when the policy does")
    path = BENCH / "fqe_what_it_tracks.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
