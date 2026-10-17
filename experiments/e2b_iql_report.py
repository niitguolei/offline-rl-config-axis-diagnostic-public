"""Merge the IQL shards and summarise **one grid at a time**.

The IQL arm is asymmetric: seeds 0-2 have tau in {0.5, 0.7, 0.9} and seeds 3-4
have tau in {0.5, 0.9}. Running `analyse` over everything averages a 9-way
selection problem with a 6-way one -- the uniform-pick null differs, so that
mean is not a mean of like things. The audit found the committed summary doing
exactly this, and the numbers had drifted with it (hopper read -5.1% against the
plan's -3.7%).

So this emits both grids separately and refuses to blend them.

Usage:
    python experiments/e2b_iql_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from e2b_crossconfig import analyse  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

# (label, seed filter, tau filter)
GRIDS = (
    ("tau{0.5,0.9} x 5 seeds", (0, 1, 2, 3, 4), (0.5, 0.9)),
    ("tau{0.5,0.7,0.9} x 3 seeds", (0, 1, 2), (0.5, 0.7, 0.9)),
)


def load() -> tuple[list[dict], dict]:
    runs, seen, shards = [], {}, {}
    for path in sorted(BENCH_DIR.glob("e2b_runs*.jsonl")):
        if "smoke" in path.name:
            continue
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["env"], r["seed"], r["lr"], r["tau"])
            if key in seen:
                continue
            seen[key] = path.name
            runs.append(r)
            kept += 1
        shards[path.name] = kept
    return runs, shards


def main() -> int:
    runs, shards = load()
    print(f"merged {len(runs)} IQL runs")
    for name, count in shards.items():
        print(f"  {name:<30} {count:>4} new")

    out: dict = {"_note": "one block per grid; the two grids must NOT be blended "
                          "(their uniform-pick nulls differ)"}
    for label, seeds, taus in GRIDS:
        subset = [r for r in runs if r["seed"] in seeds and r["tau"] in taus]
        if not subset:
            continue
        print(f"\n=== {label} ({len(subset)} runs) ===")
        res = analyse(subset)
        out[label] = res
        for env, s in res.items():
            if s.get("mixed_grid"):
                print(f"  !! {env} is still mixed -- not summarising")
                continue
            print(f"  {env}")
            for key in ("critic_mean", "fqe"):
                if key not in s:
                    continue
                v = s[key]
                print(f"    {key:<12} gap {v['mean_gap']:6.1%}  random "
                      f"{v['mean_random_gap']:6.1%}  beats "
                      f"{v['beats_random_by']:+6.1%}  SEM {v['seed_gap_sem']:5.1%}  "
                      f"-> {'USEFUL' if v['useful'] else 'no better than random'}")

    path = BENCH_DIR / "e2b_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
