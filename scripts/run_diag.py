"""Wait for the calibration, then run both divergence variants CONCURRENTLY.

Two questions, one run each:
  onestep     plain 1-step IQL, no clipping
              -> if it also diverges, n-step is not the culprit
  nstep_clip  n=5 IQL with gradient clipping (norm 10)
              -> if it stays stable, clipping is the fix

They run concurrently because this diagnostic measures DIVERGENCE, not timing,
so overlapping processes are harmless -- and it halves the wall clock. This is
the opposite of the T3b throughput probe, which must run alone.

    python scripts/run_diag.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results" / "benchmarks"
MAX_WAIT_H = 12


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    cal = LOG / "calibration.json"
    print(f"[diag] waiting for {cal.name}", flush=True)
    waited = 0.0
    while not cal.exists():
        if waited >= MAX_WAIT_H * 3600:
            print("[diag] FATAL: calibration never finished")
            return 1
        time.sleep(60)
        waited += 60
    print("[diag] calibration present; launching both variants", flush=True)

    procs = []
    for variant in ("onestep", "nstep_clip"):
        log = open(LOG / f"diag_{variant}.log", "a", encoding="utf-8")
        cmd = [PY, "experiments/diag_divergence.py", "--variant", variant,
               "--steps", "1000000"]
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        procs.append((variant, p, log))
        print(f"[diag] launched {variant} pid={p.pid}", flush=True)

    rcs = {}
    for variant, p, log in procs:
        rcs[variant] = p.wait()
        log.close()
        print(f"[diag] {variant} exit={rcs[variant]}", flush=True)

    print("[diag] both done; STOP -- read the logs before T4", flush=True)
    return 0 if all(rc == 0 for rc in rcs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
