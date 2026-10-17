"""Judge the registered AM probe in docs/第十三轮_AM.md."""

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
TAUS = (0.5, 0.7, 0.9)


def load(pattern: str) -> list[dict]:
    runs: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(str(BENCH / pattern))):
        if "smoke" in pathlib.Path(name).name:
            continue
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["env"], row["seed"], float(row["lr"]), float(row["tau"]))
            if key in seen:
                continue
            seen.add(key)
            runs.append(row)
    return runs


def one_hot_r2(values: list[float], y: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
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


def slice_corr(runs: list[dict], estimator: str) -> list[float]:
    out = []
    for lr in sorted({float(r["lr"]) for r in runs}):
        sel = sorted(
            [r for r in runs if float(r["lr"]) == lr],
            key=lambda r: float(r["tau"]),
        )
        if len(sel) < 3:
            continue
        corr = stats.spearmanr(
            [float(r["tau"]) for r in sel],
            [float(r[estimator]) for r in sel],
        ).statistic
        out.append(float(corr))
    return out


def judge_seed(runs: list[dict]) -> dict:
    sel = sorted(runs, key=lambda r: (float(r["lr"]), float(r["tau"])))
    if len(sel) != 9:
        raise AssertionError(f"expected 9 runs per seed, found {len(sel)}")
    if {float(r["tau"]) for r in sel} != set(TAUS):
        raise AssertionError("tau grid is not {0.5,0.7,0.9}")
    for lr in sorted({float(r["lr"]) for r in sel}):
        if sum(float(r["lr"]) == lr for r in sel) != 3:
            raise AssertionError(f"lr {lr} does not have three tau levels")

    truth = np.asarray([float(r["true_score"]) for r in sel])
    am = np.asarray([float(r["am"]) for r in sel])
    critic = np.asarray([float(r["critic_mean"]) for r in sel])
    corr_am_truth = float(stats.spearmanr(am, truth).statistic)
    corr_critic_truth = float(stats.spearmanr(critic, truth).statistic)

    picked = int(np.argmax(am))
    random_gap = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
    selected_gap = float(oracle_gap(truth, picked))
    true_rank = int(stats.rankdata(-truth, method="min")[picked])

    slice_am = slice_corr(sel, "am")
    slice_critic = slice_corr(sel, "critic_mean")
    max_r2_am = max(
        one_hot_r2([float(r["lr"]) for r in sel], am),
        one_hot_r2([float(r["tau"]) for r in sel], am),
    )
    exact_am = int(sum(abs(x) >= 1.0 - 1e-9 for x in slice_am))
    p1 = exact_am == 0 and max_r2_am < 0.3
    p2 = corr_am_truth >= 0.4
    p3 = true_rank <= len(sel) - 3 and selected_gap < random_gap
    if p1 and p2 and p3:
        verdict = "GO_B2"
    elif p1:
        verdict = "STOP_GAP3"
    else:
        verdict = "STOP_INHERITS_AXIS"
    return {
        "n_runs": len(sel),
        "am_slice_corr_tau": slice_am,
        "critic_slice_corr_tau": slice_critic,
        "am_exact_pm1": exact_am,
        "am_max_axis_r2": max_r2_am,
        "am_spearman_truth": corr_am_truth,
        "critic_spearman_truth": corr_critic_truth,
        "am_selected_index": picked,
        "am_selected_true_rank": true_rank,
        "am_selected_gap": selected_gap,
        "random_expected_gap": random_gap,
        "P1_config_independent": bool(p1),
        "P2_tracks_quality": bool(p2),
        "P3_selection_not_reversed": bool(p3),
        "verdict": verdict,
    }


def main() -> int:
    runs = load("e2b_am_runs_b*.jsonl")
    runs = [r for r in runs if r["env"] == ENV]
    if not runs:
        raise AssertionError("no AM shards found")
    seeds = sorted({int(r["seed"]) for r in runs})
    per_seed = {}
    for seed in seeds:
        sel = [r for r in runs if int(r["seed"]) == seed]
        per_seed[str(seed)] = judge_seed(sel)
    b1 = per_seed[str(seeds[0])]
    result = {
        "env": ENV,
        "seeds": seeds,
        "per_seed": per_seed,
        "B1_seed0": b1,
        "B1_verdict": b1["verdict"],
    }
    print(f"{ENV}: seeds={seeds}")
    for seed in seeds:
        row = per_seed[str(seed)]
        print(
            f"  seed {seed}: AM slice corr={row['am_slice_corr_tau']} "
            f"exact={row['am_exact_pm1']} R2max={row['am_max_axis_r2']:.3f} "
            f"rho(AM,truth)={row['am_spearman_truth']:+.3f} "
            f"rank={row['am_selected_true_rank']}/9 "
            f"gap={row['am_selected_gap']:.3f} random={row['random_expected_gap']:.3f} "
            f"P1={row['P1_config_independent']} P2={row['P2_tracks_quality']} "
            f"P3={row['P3_selection_not_reversed']} -> {row['verdict']}"
        )
    out = BENCH / "am_judgement.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
