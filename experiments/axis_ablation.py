"""The mechanism's ablation: delete the axis, and see what is left.

If the critic's usefulness is INHERITED from the configuration axis it tracks,
then removing that axis's component from the critic should destroy the
usefulness. If the usefulness came from tracking quality, removing the axis
should not.

Implementation: within each seed, regress `critic_mean` on the conservatism knob
and the learning rate, then rank configurations by the RESIDUAL. This uses only
the critic and the hyperparameters -- never the truth -- so it is a legitimate
selector, not an oracle.

Result (see plan section 21, Z28-C): on CQL/hopper the critic scores +34.7%
(p=0.0001) and the residualized version collapses to +3.1% (p=0.88). The
usefulness was the axis. On CQL/walker2d, where there is signal beyond the
axes, the same operation improves it (+44.8% to +63.0%).

So this is NOT offered as a method -- you cannot know in advance which case you
are in, and on one cell it destroys the signal. It is offered as the ablation a
mechanism claim needs.

Usage:
    python experiments/axis_ablation.py
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


def paired(diffs) -> tuple[float, float, int]:
    d = np.array(diffs)
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    if sem == 0:
        return float(d.mean()), 0.0, len(d)
    t = d.mean() / sem
    return float(d.mean()), float(2 * stats.t.sf(abs(t), df=len(d) - 1)), len(d)


def main() -> int:
    out = {}
    for arm, pattern, knob in (("IQL", "e2b_runs*.jsonl", "tau"),
                               ("CQL", "e2b_cql_runs*.jsonl", "min_q_weight")):
        runs = load(pattern)
        print(f"\n=== {arm}: critic vs axis-residualised critic ===")
        print(f"{'domain':<30} {'plain':>9} {'p':>8}   {'residual':>9} {'p':>8}")
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            plain, resid = [], []
            for seed in sorted({r["seed"] for r in sub}):
                sel = [r for r in sub if r["seed"] == seed]
                truth = np.array([float(r["true_score"]) for r in sel])
                rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
                c = np.array([float(r["critic_mean"]) for r in sel])
                X = np.column_stack([np.ones(len(c)),
                                     [float(r[knob]) for r in sel],
                                     [float(r["lr"]) for r in sel]])
                beta, *_ = np.linalg.lstsq(X, c, rcond=None)
                plain.append(rnd - oracle_gap(truth, int(np.argmax(c))))
                resid.append(rnd - oracle_gap(truth, int(np.argmax(c - X @ beta))))
            m1, p1, n = paired(plain)
            m2, p2, _ = paired(resid)
            out[f"{arm}|{env}"] = {"plain": m1, "plain_p": p1,
                                   "residual": m2, "residual_p": p2, "n_seeds": n}
            print(f"{env:<30} {m1:+9.1%} {p1:8.4f}   {m2:+9.1%} {p2:8.4f}")

    print("\nInterpretation: where the residualised critic collapses, the plain")
    print("critic's usefulness was the axis. Where it improves, there was signal")
    print("beyond the axes. Not a method -- an ablation.")
    path = BENCH / "axis_ablation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
