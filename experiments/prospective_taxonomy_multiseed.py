"""Guard-aware prospective taxonomy judge for TD3+BC/walker2d seeds 0/1/2."""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orl.selection.report import oracle_gap  # noqa: E402

BENCH = ROOT / "results" / "benchmarks"
ENV = "walker2d_medium_replay-v2"
ALPHAS = (0.5, 2.5, 10.0)
SEEDS = (0, 1, 2)


def load() -> list[dict]:
    runs: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(str(BENCH / "e2b_td3bc_runs_*.jsonl"))):
        if "smoke" in pathlib.Path(name).name:
            continue
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["env"], row["seed"], float(row["lr"]), float(row["alpha"]))
            if key in seen:
                continue
            seen.add(key)
            if row["env"] == ENV and int(row["seed"]) in SEEDS:
                runs.append(row)
    for seed in SEEDS:
        rows = [r for r in runs if int(r["seed"]) == seed]
        if len(rows) != 9:
            raise AssertionError(f"seed {seed}: expected 9 runs, found {len(rows)}")
        for lr in sorted({float(r["lr"]) for r in rows}):
            sub = [r for r in rows if float(r["lr"]) == lr]
            if sorted(float(r["alpha"]) for r in sub) != list(ALPHAS):
                raise AssertionError(f"seed {seed}, lr {lr}: bad alpha levels")
    return runs


def one_hot_r2(values, y) -> float:
    vals = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    levels = np.unique(vals)
    if len(levels) < 2:
        return 0.0
    x = np.column_stack(
        [np.ones(len(vals))]
        + [(vals == level).astype(np.float64) for level in levels[1:]]
    )
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def judge_seed(rows: list[dict], seed: int) -> dict:
    rows = sorted(rows, key=lambda r: (float(r["lr"]), float(r["alpha"])))
    clean = [r for r in rows if not bool(r.get("diverged", False))]
    alpha_r2 = one_hot_r2(
        [r["alpha"] for r in clean],
        [float(r["critic_mean"]) for r in clean],
    )
    lr_r2 = one_hot_r2(
        [r["lr"] for r in clean],
        [float(r["critic_mean"]) for r in clean],
    )
    slices = []
    for lr in sorted({float(r["lr"]) for r in clean}):
        sel = sorted(
            [r for r in clean if float(r["lr"]) == lr],
            key=lambda r: float(r["alpha"]),
        )
        if len(sel) < 2:
            continue
        alpha = np.asarray([float(r["alpha"]) for r in sel])
        truth = np.asarray([float(r["true_score"]) for r in sel])
        critic = np.asarray([float(r["critic_mean"]) for r in sel])
        corr_critic_alpha = float(stats.spearmanr(alpha, critic).statistic)
        corr_alpha_truth = float(stats.spearmanr(alpha, truth).statistic)
        picked = int(np.argmax(critic))
        selected_gap = float(oracle_gap(truth, picked))
        random_gap = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        aligned = (corr_critic_alpha * corr_alpha_truth) > 0
        direction_ok = (selected_gap < random_gap) if aligned else (selected_gap > random_gap)
        slices.append({
            "lr": lr,
            "n_alpha_levels": len(sel),
            "corr_critic_alpha": corr_critic_alpha,
            "corr_alpha_truth": corr_alpha_truth,
            "aligned": bool(aligned),
            "selected_gap": selected_gap,
            "random_expected_gap": random_gap,
            "direction_ok": bool(direction_ok),
        })
    complete = [s for s in slices if s["n_alpha_levels"] == 3]
    exact = int(sum(abs(s["corr_critic_alpha"]) >= 1.0 - 1e-9 for s in complete))
    divergence_rate = (len(rows) - len(clean)) / len(rows)
    insufficient_complete = len(complete) < 2
    q1 = alpha_r2 > 0.5
    q2 = len(complete) >= 2 and exact == len(complete)
    q3 = sum(s["direction_ok"] for s in slices) >= 2
    if q1 and q2 and q3:
        verdict = "GUARD_PASS"
    else:
        verdict = "GUARD_FAIL"
    return {
        "seed": seed,
        "n_runs": len(rows),
        "n_diverged": len(rows) - len(clean),
        "divergence_rate": divergence_rate,
        "insufficient_complete_slices": insufficient_complete,
        "alpha_r2_guarded": alpha_r2,
        "lr_r2_guarded": lr_r2,
        "slices": slices,
        "complete_slices": len(complete),
        "exact_pm1_complete": exact,
        "Q1_axis_dominant": bool(q1),
        "Q2_order_equals_axis": bool(q2),
        "Q3_direction_predicts": bool(q3),
        "verdict": verdict,
    }


def main() -> int:
    rows = load()
    per_seed = {
        str(seed): judge_seed([r for r in rows if int(r["seed"]) == seed], seed)
        for seed in SEEDS
    }
    passing = [
        seed for seed, row in per_seed.items() if row["verdict"] == "GUARD_PASS"
    ]
    if len(passing) == len(SEEDS):
        verdict = "GUARDED_TAXONOMY_REPRODUCES"
    elif passing:
        verdict = "PARTIAL_GUARDED_TAXONOMY"
    else:
        verdict = "GUARDED_TAXONOMY_FAILS"
    result = {
        "env": ENV,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "passing_seeds": passing,
        "verdict": verdict,
    }
    print(f"{ENV}: seeds={list(SEEDS)}")
    for seed, row in per_seed.items():
        print(
            f"  seed {seed}: diverged={row['n_diverged']} "
            f"rate={row['divergence_rate']:.2f} "
            f"alphaR2={row['alpha_r2_guarded']:.3f} "
            f"lrR2={row['lr_r2_guarded']:.3f} "
            f"complete={row['complete_slices']} exact={row['exact_pm1_complete']} "
            f"insufficient={row['insufficient_complete_slices']} "
            f"Q1={row['Q1_axis_dominant']} Q2={row['Q2_order_equals_axis']} "
            f"Q3={row['Q3_direction_predicts']} -> {row['verdict']}"
        )
        for s in row["slices"]:
            print(
                f"    lr={s['lr']:.0e} levels={s['n_alpha_levels']} "
                f"corr(critic,a)={s['corr_critic_alpha']:+.2f} "
                f"corr(a,truth)={s['corr_alpha_truth']:+.2f} "
                f"gap={s['selected_gap']:.3f} random={s['random_expected_gap']:.3f} "
                f"direction_ok={s['direction_ok']}"
            )
    print(f"  passing seeds={passing} -> {verdict}")
    out = BENCH / "prospective_taxonomy_multiseed.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
