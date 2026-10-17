"""Judge the pre-registered CQL alpha five-level extension."""

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
ENV = "hopper_medium_replay-v2"
WEIGHTS = (5.0, 10.0, 15.0, 20.0, 25.0)


def load() -> list[dict]:
    runs: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(str(BENCH / "e2b_cql_runs*.jsonl"))):
        if "smoke" in pathlib.Path(name).name:
            continue
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["env"], row["seed"], float(row["lr"]),
                float(row["min_q_weight"]),
            )
            if key in seen:
                continue
            seen.add(key)
            runs.append(row)
    selected = [
        r for r in runs
        if r["env"] == ENV and int(r["seed"]) in (0, 1, 2)
        and float(r["min_q_weight"]) in WEIGHTS
    ]
    if len(selected) != 45:
        raise AssertionError(f"expected 45 runs, found {len(selected)}")
    return sorted(
        selected,
        key=lambda r: (int(r["seed"]), float(r["lr"]), float(r["min_q_weight"])),
    )


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


def judge_seed(rows: list[dict], seed: int) -> dict:
    sel = sorted(rows, key=lambda r: (float(r["lr"]), float(r["min_q_weight"])))
    if len(sel) != 15:
        raise AssertionError(f"seed {seed}: expected 15 runs, found {len(sel)}")
    for lr in sorted({float(r["lr"]) for r in sel}):
        rows_lr = [r for r in sel if float(r["lr"]) == lr]
        if sorted(float(r["min_q_weight"]) for r in rows_lr) != list(WEIGHTS):
            raise AssertionError(f"seed {seed}, lr {lr}: bad alpha levels")

    corrs = []
    for lr in sorted({float(r["lr"]) for r in sel}):
        rows_lr = sorted(
            [r for r in sel if float(r["lr"]) == lr],
            key=lambda r: float(r["min_q_weight"]),
        )
        corrs.append(float(stats.spearmanr(
            [float(r["min_q_weight"]) for r in rows_lr],
            [float(r["critic_mean"]) for r in rows_lr],
        ).statistic))
    critic = np.asarray([float(r["critic_mean"]) for r in sel])
    truth = np.asarray([float(r["true_score"]) for r in sel])
    alpha_r2 = one_hot_r2([r["min_q_weight"] for r in sel], critic)
    lr_r2 = one_hot_r2([r["lr"] for r in sel], critic)
    exact = int(sum(abs(x) >= 1.0 - 1e-9 for x in corrs))
    direction_negative = bool(all(x < 0 for x in corrs))
    picked = int(np.argmax(critic))
    random_gap = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
    selected_gap = float(oracle_gap(truth, picked))
    q1 = exact == 3
    q2 = alpha_r2 > 0.5
    q3 = direction_negative
    if q1 and q2 and q3:
        verdict = "REPRODUCES_CQL_ALPHA_FIVE_LEVEL"
    elif q1 and q3:
        verdict = "ORDER_EQUALS_AXIS_BUT_NOT_DOMINANT"
    else:
        verdict = "DOES_NOT_REPRODUCE"
    return {
        "seed": seed,
        "n_runs": len(sel),
        "weights": list(WEIGHTS),
        "slice_corr_critic_alpha": corrs,
        "exact_pm1": exact,
        "alpha_r2": alpha_r2,
        "lr_r2": lr_r2,
        "direction_negative": direction_negative,
        "selected_index": picked,
        "selected_true_rank": int(
            stats.rankdata(-truth, method="min")[picked]
        ),
        "selected_gap": selected_gap,
        "random_expected_gap": random_gap,
        "Q1_order_equals_axis": bool(q1),
        "Q2_axis_dominant": bool(q2),
        "Q3_direction_negative": bool(q3),
        "verdict": verdict,
    }


def main() -> int:
    rows = load()
    seeds = sorted({int(r["seed"]) for r in rows})
    if seeds != [0, 1, 2]:
        raise AssertionError(f"expected seeds [0, 1, 2], found {seeds}")
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = judge_seed(
            [r for r in rows if int(r["seed"]) == seed], seed
        )
    pass_seeds = [
        seed for seed, row in per_seed.items()
        if row["verdict"] == "REPRODUCES_CQL_ALPHA_FIVE_LEVEL"
    ]
    if len(pass_seeds) == 3:
        verdict = "REPRODUCES_CQL_ALPHA_FIVE_LEVEL_MULTISEED"
    elif pass_seeds:
        verdict = "PARTIAL_MULTISEED"
    else:
        verdict = "DOES_NOT_REPRODUCE"
    result = {
        "env": ENV,
        "seeds": seeds,
        "n_runs": len(rows),
        "weights": list(WEIGHTS),
        "per_seed": per_seed,
        "passing_seeds": pass_seeds,
        "verdict": verdict,
    }
    print(f"{ENV}: seeds={seeds}, {len(rows)} unique runs, alpha={WEIGHTS}")
    for seed, row in per_seed.items():
        print(
            f"  seed {seed}: corr={row['slice_corr_critic_alpha']} "
            f"exact={row['exact_pm1']}/3 alphaR2={row['alpha_r2']:.3f} "
            f"lrR2={row['lr_r2']:.3f} rank={row['selected_true_rank']}/15 "
            f"gap={row['selected_gap']:.3f} random={row['random_expected_gap']:.3f} "
            f"-> {row['verdict']}"
        )
    print(f"  passing seeds={pass_seeds} -> {verdict}")
    out = BENCH / "cql_alpha_five_level.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
