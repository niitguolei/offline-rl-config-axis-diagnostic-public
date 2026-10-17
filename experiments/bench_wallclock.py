"""T8 — marginal per-step cost `b` of a real IQL run.

Measures `t = a + b*S` at two step counts on one dataset, then repeats a
short run on the LARGEST dataset to check that `b` does not scale with data
size. That scaling check is the point: if `b` grows with data, the I/O path
has become the bottleneck and S0 is not finished.

    python experiments/bench_wallclock.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from orl.algos.iql import IQL, IQLConfig
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.paths import BENCH_DIR

sys.stdout.reconfigure(line_buffering=True)

MEDIUM = "halfcheetah_medium-v2"
EXPERT = "halfcheetah_medium_expert-v2"
BATCH = 256
S_MED = (50_000, 150_000)
S_SCALING = 50_000


def _batch(obs, act, rew, nobs, term, idx):
    return {
        "observations": obs[idx],
        "actions": act[idx],
        "rewards": rew[idx],
        "next_observations": nobs[idx],
        "terminals": term[idx],
    }


def timed_run(env_id: str, steps: int, seed: int = 0) -> dict:
    """Time `steps` IQL updates. Returns wall clock and peak VRAM."""
    t_load0 = time.perf_counter()
    ds = D4RLDataset(env_id, mode="ram")
    for k in ("observations", "next_observations", "actions", "rewards", "terminals"):
        ds.get(k)
    t_load = time.perf_counter() - t_load0

    cfg = IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device="cuda")
    agent = IQL(cfg)
    rng = np.random.default_rng(seed)

    obs, act = ds.get("observations"), ds.get("actions")
    rew, nobs, term = ds.get("rewards"), ds.get("next_observations"), ds.get("terminals")
    n = ds.n

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warmup so CUDA kernel selection is not billed to the first steps
    for _ in range(20):
        agent.update(_batch(obs, act, rew, nobs, term, rng.integers(0, n, BATCH)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        agent.update(_batch(obs, act, rew, nobs, term, rng.integers(0, n, BATCH)))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    out = {
        "env_id": env_id,
        "steps": steps,
        "t_load_s": t_load,
        "t_steps_s": dt,
        "ms_per_step": 1000 * dt / steps,
        "n_transitions": int(n),
        "peak_vram_gb": peak,
    }
    ds.close()
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA unavailable; timings will not represent the real setup")
    print(f"device={device}  batch={BATCH}")
    print("IQL cfg: expectile=0.7 beta=3.0 gamma=0.99 tau=0.005 lr=3e-4 hidden=256x2\n")

    medium_runs = []
    for s in S_MED:
        print(f"running {MEDIUM} for {s:,} steps ...", flush=True)
        r = timed_run(MEDIUM, s)
        medium_runs.append(r)
        print(f"  {r['t_steps_s']:.1f} s  ({r['ms_per_step']:.4f} ms/step)  "
              f"peak_vram={r['peak_vram_gb']:.2f} GB", flush=True)

    s1, s2 = S_MED
    r1, r2 = medium_runs
    b_ms = (r2["t_steps_s"] - r1["t_steps_s"]) / (s2 - s1) * 1000
    a_s = r1["t_steps_s"] - b_ms / 1000 * s1
    print(f"\nfit on {MEDIUM}:")
    print(f"  b = {b_ms:.4f} ms/step")
    print(f"  a = {a_s:.3f} s (extrapolated intercept)")

    print(f"\nrunning {EXPERT} for {S_SCALING:,} steps (scaling check) ...", flush=True)
    big = timed_run(EXPERT, S_SCALING)
    print(f"  {big['t_steps_s']:.1f} s  ({big['ms_per_step']:.4f} ms/step)  "
          f"n={big['n_transitions']:,}  peak_vram={big['peak_vram_gb']:.2f} GB")

    ratio = big["ms_per_step"] / b_ms
    print(f"\n  medium predicted at S={S_SCALING:,}: {b_ms:.4f} ms/step")
    print(f"  expert measured               : {big['ms_per_step']:.4f} ms/step")
    print(f"  ratio                         : {ratio:.2f}x")
    ok = ratio <= 2.0
    print(f"\nT8 criterion (ratio <= 2.0x): {'PASS' if ok else 'FAIL'}")

    proj_min = b_ms * 1e6 / 1000 / 60
    print(f"\nprojection: a 1e6-step run costs {proj_min:.1f} min of stepping "
          f"(+ {a_s:.1f} s setup, + {0.79 * 10:.1f} s if evaluated 10 times)")

    result = {
        "device": device,
        "batch": BATCH,
        "torch": torch.__version__,
        "medium_runs": medium_runs,
        "b_ms_per_step": b_ms,
        "a_s": a_s,
        "expert_run": big,
        "scaling_ratio": ratio,
        "criterion_pass": bool(ok),
        "projected_1m_steps_min": proj_min,
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "wallclock.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

