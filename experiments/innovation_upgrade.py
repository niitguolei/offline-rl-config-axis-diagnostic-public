"""Zero-run analyses for the Round 14 innovation upgrade.

Three upgrades are computed from existing records:

1. the core same-batch contrast is tested with a continuous paired endpoint
   (seed-clustered paired difference in within-slice rank correlations), not
   only with the binary "exactly +/-1" count;
2. FQE's apparent configuration independence is re-audited per seed instead of
   being inferred from a pooled R2 that can hide seed effects;
3. the C setting sensitivity and B1 AM negative result are folded into the
   three-condition taxonomy.

Usage:
    python experiments/innovation_upgrade.py
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
            key = (
                row["env"], row["seed"], row.get("lr"), row.get("tau"),
                row.get("min_q_weight"),
            )
            if key in seen:
                continue
            seen.add(key)
            runs.append(row)
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


def spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).statistic)


def matched_contrast(rows: list[dict], axis: str = "tau") -> dict:
    slices = {}
    for row in rows:
        slices.setdefault((int(row["seed"]), float(row["lr"])), []).append(row)
    per_seed = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        diffs = []
        critic_corrs = []
        fqe_corrs = []
        for key, sel in sorted(slices.items()):
            if key[0] != seed or len(sel) != 3:
                continue
            sel = sorted(sel, key=lambda r: float(r[axis]))
            critic_corrs.append(spearman(
                [float(r[axis]) for r in sel],
                [float(r["critic_mean"]) for r in sel],
            ))
            fqe_corrs.append(spearman(
                [float(r[axis]) for r in sel],
                [float(r["fqe"]) for r in sel],
            ))
            diffs.append(critic_corrs[-1] - fqe_corrs[-1])
        per_seed[str(seed)] = {
            "critic_corr": critic_corrs,
            "fqe_corr": fqe_corrs,
            "differences": diffs,
            "mean_difference": float(np.mean(diffs)) if diffs else float("nan"),
        }
    seed_means = [v["mean_difference"] for v in per_seed.values()]
    slice_diffs = [d for v in per_seed.values() for d in v["differences"]]
    return {
        "axis": axis,
        "per_seed": per_seed,
        "slice_level_mean_difference": float(np.mean(slice_diffs)),
        "slice_level_paired_p": float(stats.ttest_rel(
            [d for v in per_seed.values() for d in v["critic_corr"]],
            [d for v in per_seed.values() for d in v["fqe_corr"]],
        ).pvalue),
        "seed_clustered_mean_difference": float(np.mean(seed_means)),
        "seed_clustered_p": float(stats.ttest_1samp(seed_means, 0).pvalue),
        "seed_clustered_n": len(seed_means),
    }


def per_seed_variance(rows: list[dict], estimator: str) -> dict:
    out = {"pooled": {}, "per_seed": {}}
    y = np.asarray([float(r[estimator]) for r in rows])
    for axis in ("tau", "lr", "seed"):
        out["pooled"][axis] = one_hot_r2([r[axis] for r in rows], y)
    for seed in sorted({int(r["seed"]) for r in rows}):
        sel = [r for r in rows if int(r["seed"]) == seed]
        yy = np.asarray([float(r[estimator]) for r in sel])
        out["per_seed"][str(seed)] = {
            "tau": one_hot_r2([r["tau"] for r in sel], yy),
            "lr": one_hot_r2([r["lr"] for r in sel], yy),
        }
    return out


def c_settings() -> dict:
    rows = load("e2b_runs_fc*.jsonl")
    rows = [r for r in rows if r["env"] == "hopper_medium_replay-v2"]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 C runs, found {len(rows)}")
    out = {}
    for setting in ("base", "mse", "long_slow"):
        vals = [float(r["fqe_sensitivity"][setting]["value"]) for r in rows]
        corrs = []
        for lr in sorted({float(r["lr"]) for r in rows}):
            sel = sorted(
                [r for r in rows if float(r["lr"]) == lr],
                key=lambda r: float(r["tau"]),
            )
            corrs.append(spearman(
                [float(r["tau"]) for r in sel],
                [float(r["fqe_sensitivity"][setting]["value"]) for r in sel],
            ))
        picked = int(np.argmax(vals))
        out[setting] = {
            "tau_r2": one_hot_r2([r["tau"] for r in rows], vals),
            "lr_r2": one_hot_r2([r["lr"] for r in rows], vals),
            "slice_corr_tau": corrs,
            "selected_index": picked,
            "selected_true_rank": int(
                stats.rankdata(
                    -np.asarray([float(r["true_score"]) for r in rows]),
                    method="min",
                )[picked]
            ),
        }
    return out


def am_probe() -> dict:
    rows = load("e2b_am_runs_bm*.jsonl")
    rows = [r for r in rows if r["env"] == "hopper_medium_replay-v2"]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 AM runs, found {len(rows)}")
    vals = [float(r["am"]) for r in rows]
    truth = np.asarray([float(r["true_score"]) for r in rows])
    return {
        "tau_r2": one_hot_r2([r["tau"] for r in rows], vals),
        "lr_r2": one_hot_r2([r["lr"] for r in rows], vals),
        "spearman_truth": spearman(vals, truth),
        "selected_true_rank": int(
            stats.rankdata(-truth, method="min")[int(np.argmax(vals))]
        ),
    }


def main() -> int:
    hopper = load("e2b_runs_m*.jsonl")
    walker = load("e2b_runs_mw*.jsonl")
    hopper = [r for r in hopper if r["env"] == "hopper_medium_replay-v2"]
    walker = [r for r in walker if r["env"] == "walker2d_medium_replay-v2"]

    result = {
        "matched_contrast": {
            "hopper": matched_contrast(hopper),
            "walker2d": matched_contrast(walker),
        },
        "variance_decomposition": {
            "hopper_critic": per_seed_variance(hopper, "critic_mean"),
            "hopper_fqe": per_seed_variance(hopper, "fqe"),
        },
        "c_settings": c_settings(),
        "am_probe": am_probe(),
        "taxonomy_upgrade": {
            "conditions": [
                "independent of the compared axis",
                "tracks quality within the run",
                "stable across the estimator's own hyperparameters",
            ],
            "critic": "fails condition 1; may satisfy 2/3 depending on arm",
            "fqe": "condition 1 is seed/setting dependent; condition 3 fails",
            "am": "fails condition 1 and condition 2 in the registered design",
            "gap_four": "still empty",
        },
    }

    print("=== matched same-batch contrast (continuous paired endpoint) ===")
    for label in ("hopper", "walker2d"):
        row = result["matched_contrast"][label]
        print(
            f"  {label:<9} seed means={row['seed_clustered_mean_difference']:+.3f} "
            f"p_seed={row['seed_clustered_p']:.5f} "
            f"p_slice={row['slice_level_paired_p']:.5f}"
        )
    print("=== FQE seed heterogeneity (tau R2) ===")
    for seed, row in result["variance_decomposition"]["hopper_fqe"]["per_seed"].items():
        print(f"  seed {seed}: tau={row['tau']:.3f} lr={row['lr']:.3f}")
    print(
        "  pooled: "
        + ", ".join(
            f"{k}={v:.3f}"
            for k, v in result["variance_decomposition"]["hopper_fqe"]["pooled"].items()
        )
    )
    print("=== C settings ===")
    for setting, row in result["c_settings"].items():
        print(
            f"  {setting:<9} tauR2={row['tau_r2']:.3f} "
            f"slice_corr={row['slice_corr_tau']} rank={row['selected_true_rank']}/9"
        )
    print("=== AM probe ===")
    print(
        f"  tauR2={result['am_probe']['tau_r2']:.3f} "
        f"rho_truth={result['am_probe']['spearman_truth']:+.3f} "
        f"rank={result['am_probe']['selected_true_rank']}/9"
    )

    out = BENCH / "innovation_upgrade.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
