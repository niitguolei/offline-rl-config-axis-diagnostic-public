"""How much of the critic's variation is the conservatism knob?

The strongest objection to the mechanism is that it is definitional: of course
the critic is monotone in the coefficient that defines it. Monotonicity is not
the empirical claim. The empirical claim is DOMINANCE -- that the knob moves the
critic so much more than anything else in the grid that comparing across
configurations is, in effect, comparing across the knob.

This quantifies that. For each arm and domain, the critic mean is regressed on
each factor alone (the conservatism knob, the learning rate, the seed) and on
all three, with categorical dummies, and the R-squared of each is reported.

If the knob alone explains most of the variance, "ranking configurations by the
critic" really is "ranking them by the knob", which is the paper's point. If it
explains a minority, the point is much weaker than the paper claims and should
be stated that way.

Usage:
    python experiments/variance_decomposition.py
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
                   r.get("min_q_weight"))
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def design(values: list, levels: list) -> np.ndarray:
    """One-hot with the first level dropped, plus an intercept column."""
    cols = [np.ones(len(values))]
    for lv in levels[1:]:
        cols.append(np.array([1.0 if v == lv else 0.0 for v in values]))
    return np.column_stack(cols)


def r2(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> int:
    out = {}
    for label, pattern, knob in (("IQL", "e2b_runs*.jsonl", "tau"),
                                 ("CQL", "e2b_cql_runs*.jsonl", "min_q_weight")):
        runs = load(pattern)
        print(f"\n=== {label} (knob = {knob}) ===")
        print(f"{'domain':<32} {'n':>4} {'knob':>8} {'lr':>8} {'seed':>8} {'all':>8}")
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            y = np.array([float(r["critic_mean"]) for r in sub])
            kn = [r[knob] for r in sub]
            lr = [r["lr"] for r in sub]
            sd = [r["seed"] for r in sub]
            lv_kn, lv_lr, lv_sd = sorted(set(kn)), sorted(set(lr)), sorted(set(sd))
            r_kn = r2(y, design(kn, lv_kn))
            r_lr = r2(y, design(lr, lv_lr))
            r_sd = r2(y, design(sd, lv_sd))
            X = np.column_stack([design(kn, lv_kn), design(lr, lv_lr)[:, 1:],
                                 design(sd, lv_sd)[:, 1:]])
            r_all = r2(y, X)
            out[f"{label}|{env}"] = {"n": len(sub), "knob": r_kn, "lr": r_lr,
                                     "seed": r_sd, "all": r_all}
            print(f"{env:<32} {len(sub):>4} {r_kn:>8.2f} {r_lr:>8.2f} "
                  f"{r_sd:>8.2f} {r_all:>8.2f}")

    print("\nnote: single-factor R-squared values overlap when factors are")
    print("correlated, so they need not sum to the joint value.")
    path = BENCH / "variance_decomposition.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
