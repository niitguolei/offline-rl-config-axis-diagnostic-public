"""Judge the registered TD3+BC predictions P1 and P2 (plan/doc section 7.4).

The prediction was written before the runs existed:

  P1  the share of slices whose critic-to-alpha rank correlation is exactly
      +/-1.00 will be < 7/9, against 48/48 for IQL/CQL and 24/48 for FQE
  P2  alpha's share of the critic's variance will be < 0.5, against 0.98 for
      IQL/hopper and 0.80 for CQL/hopper

and three dispositions were pre-committed, including that failing both means the
inheritance claim is falsified and the mechanism must fall back to the weaker
"the critic is a function of the configuration".

The glob/filename trap has bitten four times in this project, so the merge
asserts the expected record count rather than trusting it.

Usage:
    python experiments/td3bc_judgement.py
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
EXPECTED = 27          # 9 configs x 3 seeds
CONTRAST = {"IQL/CQL critic": 48 / 48, "FQE": 24 / 48}
VAR_CONTRAST = {"IQL/hopper": 0.98, "CQL/hopper": 0.80}


def load() -> list[dict]:
    runs, seen = [], set()
    for name in sorted(glob.glob(str(BENCH / "e2b_td3bc_runs_t*.jsonl"))):
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["env"], r["seed"], r["lr"], r["alpha"])
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
    return 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())


def main() -> int:
    runs = load()
    print(f"merged {len(runs)} TD3+BC runs (expected {EXPECTED})")
    assert len(runs) == EXPECTED, (
        f"expected {EXPECTED} runs, got {len(runs)} -- check the glob before "
        f"trusting anything below")
    seeds = sorted({r["seed"] for r in runs})
    print(f"  seeds {seeds}, alphas {sorted({r['alpha'] for r in runs})}, "
          f"lrs {sorted({r['lr'] for r in runs})}\n")

    # --- P1 -----------------------------------------------------------------
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        buckets.setdefault((r["seed"], r["lr"]), []).append(r)
    cors = []
    for key, sel in sorted(buckets.items()):
        sel = sorted(sel, key=lambda r: r["alpha"])
        a = stats.rankdata([float(r["alpha"]) for r in sel])
        c = stats.rankdata([float(r["critic_mean"]) for r in sel])
        cors.append(float(np.corrcoef(c, a)[0, 1]))
    cors_arr = np.array(cors)
    exact = int((np.abs(cors_arr) > 0.999).sum())
    p1_holds = exact < 7
    print("=== P1: within-slice rank corr(critic, alpha) ===")
    print(f"  {len(cors)} slices, range {cors_arr.min():+.2f}..{cors_arr.max():+.2f}")
    print(f"  exactly +/-1.00: {exact}/{len(cors)}   (predicted < 7)")
    for label, v in CONTRAST.items():
        print(f"    contrast {label:<14} {v:.0%}")
    print(f"  -> P1 {'HOLDS' if p1_holds else 'FAILS'}\n")

    # --- P2 -----------------------------------------------------------------
    y = np.array([float(r["critic_mean"]) for r in runs])
    al = [r["alpha"] for r in runs]
    lr = [r["lr"] for r in runs]
    sd = [r["seed"] for r in runs]
    r_alpha = r2(y, design(al, sorted(set(al))))
    r_lr = r2(y, design(lr, sorted(set(lr))))
    r_seed = r2(y, design(sd, sorted(set(sd))))
    p2_holds = r_alpha < 0.5
    print("=== P2: variance decomposition of the critic ===")
    print(f"  alpha R2 = {r_alpha:.2f}   lr R2 = {r_lr:.2f}   seed R2 = {r_seed:.2f}")
    for label, v in VAR_CONTRAST.items():
        print(f"    contrast {label:<14} {v:.2f}")
    print(f"  -> P2 {'HOLDS' if p2_holds else 'FAILS'}\n")

    # --- disposition --------------------------------------------------------
    print("=== disposition (pre-committed in section 7.4) ===")
    if p1_holds and p2_holds:
        verdict = ("BOTH HOLD: the inheritance claim gains a predicted-negative "
                   "case; algorithm coverage goes from 2 to 3")
    elif p1_holds or p2_holds:
        which = "P1" if p1_holds else "P2"
        verdict = (f"ONE HOLDS ({which}): report which, and qualify the mechanism "
                   f"with the fact that alpha reaches the critic indirectly")
    else:
        verdict = ("NEITHER HOLDS: the inheritance claim is FALSIFIED -- the "
                   "indirect path is enough for the critic to inherit the knob. "
                   "The mechanism must fall back to 'the critic is a function of "
                   "the configuration' and the 'decided by the training "
                   "objective' wording must be removed")
    print(f"  {verdict}")

    out = BENCH / "td3bc_judgement.json"
    out.write_text(json.dumps(
        {"n_runs": len(runs), "p1": {"exact": exact, "slices": len(cors),
                                     "correlations": cors, "holds": p1_holds},
         "p2": {"alpha_r2": r_alpha, "lr_r2": r_lr, "seed_r2": r_seed,
                "holds": p2_holds},
         "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
