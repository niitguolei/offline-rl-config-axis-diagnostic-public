"""Judge the pre-registered TD3+BC/walker2d prospective taxonomy pilot."""

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


def load() -> list[dict]:
    runs: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(str(BENCH / "e2b_td3bc_runs_p*.jsonl"))):
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
            runs.append(row)
    selected = [
        r for r in runs
        if r["env"] == ENV and int(r["seed"]) == 0
        and float(r["alpha"]) in ALPHAS
    ]
    if len(selected) != 9:
        raise AssertionError(f"expected 9 runs, found {len(selected)}")
    for lr in sorted({float(r["lr"]) for r in selected}):
        rows = [r for r in selected if float(r["lr"]) == lr]
        if sorted(float(r["alpha"]) for r in rows) != list(ALPHAS):
            raise AssertionError(f"lr {lr} does not have three alpha levels")
    return sorted(selected, key=lambda r: (float(r["lr"]), float(r["alpha"])))


def one_hot_r2(values, y) -> float:
    vals = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    levels = np.unique(vals)
    x = np.column_stack(
        [np.ones(len(vals))]
        + [(vals == level).astype(np.float64) for level in levels[1:]]
    )
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> int:
    rows = load()
    critic = np.asarray([float(r["critic_mean"]) for r in rows])
    alpha_r2 = one_hot_r2([r["alpha"] for r in rows], critic)
    lr_r2 = one_hot_r2([r["lr"] for r in rows], critic)
    p1 = max(alpha_r2, lr_r2) > 0.5

    slices = []
    for lr in sorted({float(r["lr"]) for r in rows}):
        sel = sorted(
            [r for r in rows if float(r["lr"]) == lr],
            key=lambda r: float(r["alpha"]),
        )
        alpha = np.asarray([float(r["alpha"]) for r in sel])
        truth = np.asarray([float(r["true_score"]) for r in sel])
        crit = np.asarray([float(r["critic_mean"]) for r in sel])
        corr_critic_alpha = float(stats.spearmanr(alpha, crit).statistic)
        corr_alpha_truth = float(stats.spearmanr(alpha, truth).statistic)
        picked = int(np.argmax(crit))
        selected_gap = float(oracle_gap(truth, picked))
        random_gap = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
        aligned = (corr_critic_alpha * corr_alpha_truth) > 0
        direction_ok = (selected_gap < random_gap) if aligned else (selected_gap > random_gap)
        slices.append({
            "lr": lr,
            "alpha": alpha.tolist(),
            "true": truth.tolist(),
            "critic": crit.tolist(),
            "corr_critic_alpha": corr_critic_alpha,
            "corr_alpha_truth": corr_alpha_truth,
            "aligned": bool(aligned),
            "selected_index": picked,
            "selected_gap": selected_gap,
            "random_expected_gap": random_gap,
            "direction_ok": bool(direction_ok),
        })
    exact = int(sum(abs(s["corr_critic_alpha"]) >= 1.0 - 1e-9 for s in slices))
    p2 = exact == 3
    p3 = sum(s["direction_ok"] for s in slices) >= 2
    clean = [r for r in rows if not bool(r.get("diverged", False))]
    clean_alpha_r2 = one_hot_r2(
        [r["alpha"] for r in clean],
        [float(r["critic_mean"]) for r in clean],
    )
    clean_lr_r2 = one_hot_r2(
        [r["lr"] for r in clean],
        [float(r["critic_mean"]) for r in clean],
    )
    if p1 and p2 and p3:
        verdict = "GO_EXTEND"
    elif p1 and p2:
        verdict = "DIRECTION_LIMIT"
    else:
        verdict = "FAIL"
    result = {
        "env": ENV,
        "seed": 0,
        "n_runs": len(rows),
        "alphas": list(ALPHAS),
        "alpha_r2": alpha_r2,
        "lr_r2": lr_r2,
        "diverged_runs": len(rows) - len(clean),
        "sensitivity_excluding_diverged": {
            "n_runs": len(clean),
            "alpha_r2": clean_alpha_r2,
            "lr_r2": clean_lr_r2,
        },
        "exact_pm1": exact,
        "slices": slices,
        "P1_axis_dependence": bool(p1),
        "P2_order_equals_axis": bool(p2),
        "P3_direction_predicts": bool(p3),
        "verdict": verdict,
    }
    print(f"{ENV} seed 0: {len(rows)} runs")
    print(f"  alphaR2={alpha_r2:.3f} lrR2={lr_r2:.3f} exact={exact}/3")
    print(
        f"  excluding diverged: n={len(clean)} "
        f"alphaR2={clean_alpha_r2:.3f} lrR2={clean_lr_r2:.3f}"
    )
    for s in slices:
        print(
            f"  lr={s['lr']:.0e} corr(critic,a)={s['corr_critic_alpha']:+.2f} "
            f"corr(a,truth)={s['corr_alpha_truth']:+.2f} "
            f"aligned={s['aligned']} gap={s['selected_gap']:.3f} "
            f"random={s['random_expected_gap']:.3f} ok={s['direction_ok']}"
        )
    print(f"  P1={p1} P2={p2} P3={p3} -> {verdict}")
    out = BENCH / "prospective_taxonomy.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
