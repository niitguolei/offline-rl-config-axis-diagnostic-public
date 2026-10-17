"""Re-rank a cross-configuration arm along ONE axis at a time (no new training).

The cross-configuration arm crossed lr with tau, so a rank there mixes a clean
axis with a polluted one. This splits it: the same 90 runs are re-ranked along
lr at fixed tau, and along tau at fixed lr. The leak probe is printed next to
the gaps, because the claim under test is that the leak predicts the gap --
computed before the gap is looked at.

Works for the CQL arm too (`--runs ... --axis lr --within min_q_weight`), which
matters because that arm's pooled ranking says the critic mean is a GOOD
selector even though it is strongly correlated with both of its hyperparameters.
The within-stratum correlations are what tell those two situations apart.

Usage:
    python experiments/e2b_stratified.py
    python experiments/e2b_stratified.py --runs results/benchmarks/e2b_cql_runs*.jsonl \
        --axis lr --within min_q_weight --out results/benchmarks/e2b_cql_stratified.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import glob
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orl.selection.facets import facet_gaps, leak_correlation  # noqa: E402

def load(pattern: str) -> list[dict]:
    """Read one file or a glob of shards, deduplicating by (env, seed, config)."""
    runs, seen = [], set()
    for name in sorted(glob.glob(str(ROOT / pattern))):
        # A glob of `e2b_cql_runs*.jsonl` also matches the smoke file, whose
        # 1500-step runs carry a third min_q_weight and would invent an extra
        # level on that axis. Caught by the run count reading 57 instead of 54.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/benchmarks/e2b_runs.jsonl")
    ap.add_argument("--axis", default="lr")
    ap.add_argument("--within", default="tau",
                    help="the axis held fixed while ranking along --axis")
    ap.add_argument("--out", default="results/benchmarks/e2b_stratified.json")
    args = ap.parse_args()
    runs = load(args.runs)
    print(f"{len(runs)} runs")

    report: dict = {"n_runs": len(runs), "axes": {}}
    for axis, within in ((args.axis, [args.within]),
                         (args.within, [args.axis])):
        n_levels = len({r[axis] for r in runs})
        if n_levels < 3:
            print(f"\n=== selection along {axis} === SKIPPED: only {n_levels} "
                  f"levels, so a uniform pick already succeeds half the time "
                  f"and no slice can exercise selection. Reported as a "
                  f"limitation, not a result.")
            report["axes"][axis] = {"within": within, "skipped": f"{n_levels} levels"}
            continue
        by_env = facet_gaps(runs=runs, axis=axis, within=within)
        print(f"\n=== selection along {axis}, holding {within[0]} fixed ===")
        axes = {"within": within, "per_env": {}}
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            leak = {key: leak_correlation(runs=sub, axis=axis, key=key)
                    for key in ("critic_mean", "fqe")}
            summary = by_env.get(env, {}).get("summary", {})
            print(f"  {env}")
            print(f"    corr(critic,{axis})={leak['critic_mean']:+.2f}  "
                  f"corr(fqe,{axis})={leak['fqe']:+.2f}")
            for key, s in summary.items():
                verdict = "USEFUL" if s["useful"] else "no better than random"
                sem_pct = s["seed_gap_sem"] * 100
                ratio = s["beats_random_by"] / s["seed_gap_sem"] if s["seed_gap_sem"] else float("nan")
                print(f"    {key:<11} gap {s['mean_gap']*100:5.1f}%  "
                      f"random {s['mean_random_gap']*100:5.1f}%  "
                      f"beats by {s['beats_random_by']*100:+5.1f}% "
                      f"(SEM {sem_pct:4.1f}%, {ratio:+.2f} SEM)  -> {verdict}")
            axes["per_env"][env] = {"leak": leak, "summary": summary,
                                    "per_slice": by_env.get(env, {}).get("per_slice", {})}
        report["axes"][axis] = axes

    out = ROOT / args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
