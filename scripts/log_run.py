"""CLI for appending experiment records to results/runs/*.jsonl.

Thin wrapper over orl.record.logger so the CLI and the training loop cannot
drift apart.

    python scripts/log_run.py --run-id iql_hc_mr_s0 --stage S5 --method iql \
        --dataset halfcheetah_medium_replay-v2 --seed 0 --score-norm 72.4

    python scripts/log_run.py --summarise
    python scripts/log_run.py --summarise --out results/tables/main.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

from orl.paths import RESULTS_DIR, parse_env_id
from orl.record.logger import RUNS_DIR, log_run

TABLES = RESULTS_DIR / "tables"
VALID_DOMAINS = ("halfcheetah", "walker2d", "hopper")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(RUNS_DIR.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def summarise(out: pathlib.Path | None) -> None:
    rows = load_rows()
    if not rows:
        print("no runs recorded yet")
        return
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.get("status") != "done" or r.get("score_norm") is None:
            continue
        agg.setdefault((r.get("method", ""), r.get("dataset", "")), []).append(
            float(r["score_norm"])
        )
    print(f"{'method':<16}{'dataset':<34}{'n':>3}{'mean':>9}{'std':>8}")
    print("-" * 70)
    lines = ["method,dataset,n_seeds,mean_norm,std_norm"]
    for (method, dataset), vals in sorted(agg.items()):
        n = len(vals)
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        print(f"{method:<16}{dataset:<34}{n:>3}{mean:>9.1f}{std:>8.1f}")
        lines.append(f"{method},{dataset},{n},{mean:.2f},{std:.2f}")
    if out:
        TABLES.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(lines)
        print(f"\nwrote {out}")
    print(f"\n{len(rows)} records in {RUNS_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--stage", default="S0")
    ap.add_argument("--method")
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--status", default="done",
                    choices=["done", "failed", "running"])
    ap.add_argument("--score-norm", type=float, dest="score_norm")
    ap.add_argument("--score-raw", type=float, dest="score_raw")
    ap.add_argument("--wall-clock-min", type=float, dest="wall_clock_min")
    ap.add_argument("--peak-vram-gb", type=float, dest="peak_vram_gb")
    ap.add_argument("--config")
    ap.add_argument("--notes", default="")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.summarise:
        summarise(pathlib.Path(a.out) if a.out else None)
        return 0

    missing = [k for k in ("run_id", "method", "dataset", "seed")
               if getattr(a, k) is None]
    if missing:
        ap.error("missing required: " + ", ".join(missing))

    domain, tier = parse_env_id(a.dataset)
    if domain not in VALID_DOMAINS:
        print(f"WARNING: domain '{domain}' not in approved set {VALID_DOMAINS} "
              "(ant and antmaze are permanently excluded)")

    path = log_run(
        run_id=a.run_id, method=a.method, dataset=a.dataset, domain=domain,
        tier=tier, seed=a.seed, stage=a.stage, status=a.status,
        score_norm=a.score_norm, score_raw=a.score_raw,
        wall_clock_min=a.wall_clock_min, peak_vram_gb=a.peak_vram_gb,
        config=a.config, notes=a.notes,
    )
    print(f"recorded -> {path.relative_to(RESULTS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

