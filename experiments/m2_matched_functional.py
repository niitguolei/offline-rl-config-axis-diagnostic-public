"""Audit M2: does the free baseline's verdict depend on which functional it is?

The IQL arm read its free baseline off `V(s)` -- an expectile of Q over DATASET
actions. The CQL arm read `min(Q1,Q2)(s, pi(s).mean)`, Q at the POLICY's action.
The paper's central figure is a cross-arm contrast, so it was comparing two
different objects, and no checkpoints were saved, so it had to be re-measured.

`e2b_crossconfig.py` now records both on the same runs. This is the analysis the
plan registered in section 17.3 BEFORE those runs existed:

  - same verdict under both functionals  -> M2 is empirically immaterial, the
    cross-arm contrast stands and these runs are the evidence for saying so;
  - different verdict                    -> the contrast is restricted to
    `critic_mean_q` and the V-based numbers become a within-arm robustness check.

It also re-tests the mechanism claim -- within a fixed (seed, lr) slice, the
critic's rank correlation with tau is exactly +/-1.00 -- for the matched
functional. If that holds too, the mechanism is independent of the functional
form, which is the weakest joint in the argument.

Usage:
    python experiments/m2_matched_functional.py
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
FUNCTIONALS = ("critic_mean", "critic_mean_q")


def load_matched() -> list[dict]:
    """The S5 shards only. Explicitly not the glob the other scripts use, whose
    dedup prefers e2b_runs.jsonl (which lacks critic_mean_q)."""
    runs, seen = [], set()
    for name in sorted(glob.glob(str(BENCH / "e2b_runs_m*.jsonl"))):
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


def cross_config(runs: list[dict], key: str) -> dict:
    diffs, gaps, rnds = [], [], []
    for seed in sorted({r["seed"] for r in runs}):
        sel = [r for r in runs if r["seed"] == seed]
        truth = np.array([float(r["true_score"]) for r in sel])
        pick = int(np.argmax([float(r[key]) for r in sel]))
        gap = oracle_gap(truth, pick)
        rnd = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        gaps.append(gap)
        rnds.append(rnd)
        diffs.append(rnd - gap)
    d = np.array(diffs)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem
    p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
    return {"mean_gap": float(np.mean(gaps)), "mean_random_gap": float(np.mean(rnds)),
            "beats": float(d.mean()), "sem": float(sem), "t": float(t), "p": p,
            "n_seeds": len(d), "useful": bool(p < 0.05 and d.mean() > 0),
            "per_seed_gaps": gaps, "per_seed_random": rnds}


def slice_rank_corr(runs: list[dict], key: str) -> list[float]:
    """Within (seed, lr), the rank correlation of the estimator against tau."""
    slices: dict[tuple, list[dict]] = {}
    for r in runs:
        slices.setdefault((r["seed"], r["lr"]), []).append(r)
    out = []
    for _, sel in sorted(slices.items()):
        if len({r["tau"] for r in sel}) < 3:
            continue
        sel = sorted(sel, key=lambda r: r["tau"])
        est = np.array([float(r[key]) for r in sel])
        tau = np.array([float(r["tau"]) for r in sel])
        out.append(float(np.corrcoef(stats.rankdata(est), stats.rankdata(tau))[0, 1]))
    return out


def main() -> int:
    runs = load_matched()
    envs = sorted({r["env"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    print(f"matched runs {len(runs)}  envs={envs}  seeds={seeds}")
    if len(envs) != 1:
        print("!! expected exactly one env (hopper) -- the registered design is "
              "hopper-only; stopping rather than mixing domains")
        return 1
    env = envs[0]
    print(f"  {env}: {len(runs)} runs, "
          f"{len({r['lr'] for r in runs})} lr x {len({r['tau'] for r in runs})} tau "
          f"x {len(seeds)} seeds\n")

    print("=== cross-configuration selection, one functional at a time ===")
    res = {}
    for key in FUNCTIONALS:
        s = cross_config(runs, key)
        res[key] = s
        print(f"  {key:<15} gap {s['mean_gap']:6.1%}  random {s['mean_random_gap']:6.1%}"
              f"  beats {s['beats']:+6.1%}  paired SEM {s['sem']:5.1%}  "
              f"t={s['t']:+6.2f}  p={s['p']:.4f}  -> "
              f"{'USEFUL' if s['useful'] else 'no better than random'}")

    print("\n=== the registered decision (plan section 17.3) ===")
    agree = res["critic_mean"]["useful"] == res["critic_mean_q"]["useful"]
    if agree:
        print("  SAME verdict under both functionals -> M2 is empirically")
        print("  immaterial: the cross-arm contrast stands, and these runs are")
        print("  the evidence for saying so.")
    else:
        print("  DIFFERENT verdict -> the cross-arm contrast is restricted to")
        print("  critic_mean_q; the V(s) numbers become a within-arm robustness")
        print("  check and must not be used across arms.")
    print(f"  (V(s) useful={res['critic_mean']['useful']}, "
          f"Q(s,pi) useful={res['critic_mean_q']['useful']})")

    print("\n=== mechanism claim: within-slice rank corr(estimate, tau) ===")
    for key in FUNCTIONALS:
        cors = np.array(slice_rank_corr(runs, key))
        exact = int((np.abs(cors) > 0.999).sum())
        print(f"  {key:<15} n={len(cors)} slices  min={cors.min():+.2f} "
              f"max={cors.max():+.2f}  exactly +/-1.00: {exact}/{len(cors)}")

    out = BENCH / "m2_matched_functional.json"
    out.write_text(json.dumps(
        {"env": env, "n_runs": len(runs), "seeds": seeds,
         "cross_config": res,
         "slice_rank_corr": {k: slice_rank_corr(runs, k) for k in FUNCTIONALS},
         "same_verdict": agree}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
