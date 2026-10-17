"""T3b — does a second concurrent training process actually help?

The S5 schedule rests on "2 concurrent processes compress 732 h to ~370 h",
but that was never measured. Memory allows it (2 x 1.65 GB against 4.2 GB
free); whether it HELPS is a separate question. IQL at 7.10 ms/step is
launch-bound rather than FLOP-bound, so idle GPU gaps should be fillable --
but that is a hypothesis, and the plan requires verifying it.

Must be measured ALONE: run beside other training and it measures the wrong
thing entirely.

    python experiments/bench_concurrency.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

WORKER = "experiments/_conc_worker.py"
OUT = "results/benchmarks/concurrency.json"


def run_group(n_proc: int, steps: int) -> tuple[float, float]:
    """Spawn n_proc workers; return (wall_clock_s, aggregate_steps_per_s)."""
    procs = [
        subprocess.Popen([sys.executable, WORKER, str(steps), f"p{i}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
        for i in range(n_proc)
    ]
    t0 = time.perf_counter()
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            print(out, flush=True)
            raise RuntimeError(f"worker failed rc={p.returncode}")
    wall = time.perf_counter() - t0
    return wall, n_proc * steps / wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    print(f"steps per process: {a.steps:,}")
    print("NOTE: nothing else may be using the GPU during this run.\n")

    one_wall, one_tp = run_group(1, a.steps)
    print(f"1 process : {one_wall:7.1f} s   {one_tp:8.1f} steps/s")

    two_wall, two_tp = run_group(2, a.steps)
    print(f"2 processes: {two_wall:7.1f} s   {two_tp:8.1f} steps/s")

    speedup = two_tp / one_tp
    print(f"\naggregate speedup: {speedup:.2f}x")
    verdict = ("2 CONCURRENT IS WORTH IT" if speedup >= 1.3
               else "NOT WORTH IT -- keep serial" if speedup >= 1.05
               else "CONCURRENCY HURTS")
    print(f"verdict: {verdict}")

    out = {"steps_per_proc": a.steps, "one_wall_s": one_wall, "one_steps_per_s": one_tp,
           "two_wall_s": two_wall, "two_steps_per_s": two_tp,
           "aggregate_speedup": speedup, "verdict": verdict}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
