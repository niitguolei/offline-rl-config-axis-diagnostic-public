"""Run the remaining S1 stages unattended, in dependency order.

    T3b concurrency probe
    T4  GATE 1 training   (reuses calibration's n=5 run)
    T5  verdict

Start this AFTER experiments/calibrate.py has been launched: it waits for
calibration.json to appear before touching the GPU.

Order is forced by dependencies:
  * T3b must not overlap other training. A concurrent-throughput probe run
    beside another job measures the wrong thing entirely.
  * T4 needs the step budget that T3 produces.
  * T5 needs T4.

    python scripts/run_s1_chain.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results" / "benchmarks"
POLL_S = 60
MAX_WAIT_H = 12
def step(name: str, script: str, log_name: str) -> int:
    print(f"[chain] {name}", flush=True)
    with open(LOG / log_name, "a", encoding="utf-8") as fh:
        rc = subprocess.call([PY, script], cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT)
    print(f"[chain] {name} exit={rc}", flush=True)
    return rc


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    cal = LOG / "calibration.json"
    print(f"[chain] waiting for {cal.name} (max {MAX_WAIT_H} h)", flush=True)
    waited = 0.0
    while not cal.exists():
        if waited >= MAX_WAIT_H * 3600:
            print("[chain] FATAL: calibration never finished; see calibrate.log")
            return 1
        time.sleep(POLL_S)
        waited += POLL_S
    print(f"[chain] calibration done after {waited / 60:.0f} min waiting", flush=True)

    step("T3b concurrency probe", "experiments/bench_concurrency.py", "concurrency.log")
    if step("T4 GATE 1 training", "experiments/gate1.py", "gate1.log") != 0:
        print("[chain] T4 failed; not running the verdict on partial data")
        return 1
    step("T5 verdict", "experiments/gate1_verdict.py", "gate1_verdict.log")
    print("[chain] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
