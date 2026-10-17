"""Judge the Round 13/C FQE sensitivity check.

This is a descriptive check, not a new significance test.  It reports whether
the ranking of the nine hopper configurations changes when FQE's loss, horizon,
or learning rate changes.
"""

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
SETTINGS = ("base", "mse", "long_slow")


def load() -> list[dict]:
    runs: list[dict] = []
    seen: set[tuple] = set()
    for name in sorted(glob.glob(str(BENCH / "e2b_runs_fc*.jsonl"))):
        for line in pathlib.Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["env"], row["seed"], float(row["lr"]), float(row["tau"]))
            if key in seen:
                continue
            seen.add(key)
            runs.append(row)
    selected = [
        r for r in runs
        if r["env"] == ENV and int(r["seed"]) == 0
        and float(r["tau"]) in TAUS
    ]
    if len(selected) != 9:
        raise AssertionError(f"expected 9 runs, found {len(selected)}")
    for lr in sorted({float(r["lr"]) for r in selected}):
        if sum(float(r["lr"]) == lr for r in selected) != 3:
            raise AssertionError(f"lr {lr} does not have three tau levels")
    return sorted(selected, key=lambda r: (float(r["lr"]), float(r["tau"])))


def values(runs: list[dict], setting: str) -> np.ndarray:
    out = []
    for row in runs:
        payload = row.get("fqe_sensitivity")
        if not payload or setting not in payload:
            raise AssertionError(f"missing FQE setting {setting}")
        out.append(float(payload[setting]["value"]))
    return np.asarray(out)


def slice_corr(runs: list[dict], setting: str) -> list[float]:
    out = []
    for lr in sorted({float(r["lr"]) for r in runs}):
        sel = [r for r in runs if float(r["lr"]) == lr]
        corr = stats.spearmanr(
            [float(r["tau"]) for r in sel],
            [float(r["fqe_sensitivity"][setting]["value"]) for r in sel],
        ).statistic
        out.append(float(corr))
    return out


def main() -> int:
    runs = load()
    truth = np.asarray([float(r["true_score"]) for r in runs])
    payloads = {name: values(runs, name) for name in SETTINGS}
    if not np.allclose(payloads["base"], [float(r["fqe"]) for r in runs]):
        raise AssertionError("base sensitivity entry differs from recorded fqe")

    pairwise = {}
    for i, a in enumerate(SETTINGS):
        for b in SETTINGS[i + 1:]:
            pairwise[f"{a}|{b}"] = float(
                stats.spearmanr(payloads[a], payloads[b]).statistic
            )

    random_gap = float(np.mean([oracle_gap(truth, i) for i in range(len(truth))]))
    per_setting = {}
    selected = {}
    for name in SETTINGS:
        picked = int(np.argmax(payloads[name]))
        selected[name] = picked
        per_setting[name] = {
            "slice_corr_tau": slice_corr(runs, name),
            "selected_index": picked,
            "selected_true_rank": int(stats.rankdata(-truth, method="min")[picked]),
            "selected_gap": float(oracle_gap(truth, picked)),
            "random_expected_gap": random_gap,
        }
    unstable = bool(
        any(abs(value) < 0.5 for value in pairwise.values())
        or len(set(selected.values())) > 1
    )
    result = {
        "env": ENV,
        "n_runs": len(runs),
        "settings": list(SETTINGS),
        "pairwise_spearman": pairwise,
        "per_setting": per_setting,
        "selected_configs": selected,
        "fqe_sensitivity_unstable": unstable,
        "disposition": (
            "limit the FQE claim to the primary setting"
            if unstable else
            "the FQE ranking is stable across the three registered settings"
        ),
    }
    print(f"{ENV}: {len(runs)} runs")
    for name in SETTINGS:
        row = per_setting[name]
        print(
            f"  {name:<9} slice corr={row['slice_corr_tau']} "
            f"selected={row['selected_index']} rank={row['selected_true_rank']}/9 "
            f"gap={row['selected_gap']:.3f} random={row['random_expected_gap']:.3f}"
        )
    print(f"  pairwise={pairwise}")
    print(f"  unstable={unstable} -> {result['disposition']}")
    out = BENCH / "fqe_sensitivity_judgement.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
