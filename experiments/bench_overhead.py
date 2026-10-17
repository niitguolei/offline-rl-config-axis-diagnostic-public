"""T6 — algorithm-independent fixed overhead of a training run.

Splits `t = a + b*S` into its pieces so that T8 (which needs IQL) only has to
measure the marginal per-step cost `b`:

    a = t_setup + n_evals * t_eval
    b = IQL gradient step + minibatch sampling

This runs BEFORE any algorithm exists, on purpose: it puts the most important
unknown (wall clock) at the front of week 2 rather than the back.

    python experiments/bench_overhead.py
"""
from __future__ import annotations

import json
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import ZeroLikePolicy, evaluate
from orl.envs.obs_layout import ENV_IDS
from orl.paths import BENCH_DIR

BATCH = 256
S_POINTS = (10_000, 30_000)
# Streaming is ~3 orders of magnitude slower per step, so we measure its
# PER-STEP RATE on a short run and extrapolate. Measuring 30k stream steps
# would take tens of minutes for a number that is linear in steps.
STREAM_STEPS = 200
DOMAIN = "halfcheetah"
EVAL_EPISODES = 10

sys.stdout.reconfigure(line_buffering=True)


def time_setup(env_id: str, mode: str = "ram") -> float:
    t0 = time.perf_counter()
    with D4RLDataset(env_id, mode=mode) as ds:
        for k in ("observations", "next_observations", "actions", "rewards"):
            ds.get(k)
    return time.perf_counter() - t0


def time_sampling(env_id: str, steps: int, mode: str, seed: int = 0) -> float:
    """Cost of drawing `steps` minibatches: the data-access floor of b."""
    with D4RLDataset(env_id, mode=mode) as ds:
        rng = np.random.default_rng(seed)
        ds.get("observations")  # warm the cache
        t0 = time.perf_counter()
        for _ in range(steps):
            ds.sample(BATCH, rng)
        return time.perf_counter() - t0


def time_eval(domain: str, n_episodes: int, seed: int = 0) -> float:
    env = gym.make(ENV_IDS[domain])
    act_dim = int(env.action_space.shape[0])
    env.close()
    t0 = time.perf_counter()
    evaluate(domain, ZeroLikePolicy(act_dim), n_episodes=n_episodes, seed=seed)
    return time.perf_counter() - t0


def time_dummy_step(steps: int, obs_dim: int, act_dim: int, device: str) -> float:
    """A stand-in gradient step, exposing the non-data compute floor."""
    x = torch.randn(BATCH, obs_dim, device=device)
    w = torch.randn(obs_dim, act_dim, device=device, requires_grad=True)
    for _ in range(50):  # warmup
        w.grad = None
        (x @ w).sum().backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        w.grad = None
        (x @ w).sum().backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> int:
    env_id = f"{DOMAIN}_medium-v2"
    print(f"batch={BATCH}  S points={S_POINTS}  eval episodes={EVAL_EPISODES}\n")

    print("setup (open + load into RAM):")
    t_setup = time_setup(env_id)
    print(f"  {t_setup:.2f} s")

    print("\nminibatch sampling cost (the data floor of b):")
    sampling: dict[str, dict[str, float]] = {}
    for mode in ("ram", "stream"):
        sampling[mode] = {}
        points = S_POINTS if mode == "ram" else (STREAM_STEPS,)
        for s in points:
            t = time_sampling(env_id, s, mode)
            sampling[mode][str(s)] = t
            print(f"  mode={mode:<6} S={s:>6}  {t:7.2f} s  ({1000 * t / s:.4f} ms/step)")
    ram_ms = 1000 * sampling["ram"][str(S_POINTS[-1])] / S_POINTS[-1]
    stream_ms = 1000 * sampling["stream"][str(STREAM_STEPS)] / STREAM_STEPS
    ratio = stream_ms / ram_ms
    print(f"  -> per-step rate: ram {ram_ms:.4f} ms, stream {stream_ms:.3f} ms "
          f"-> {ratio:.0f}x")
    print("  (this ratio is why training defaults to mode='ram': streaming")
    print("   would dominate b and masquerade as a data-loading bottleneck)")

    print("\nfull evaluation cost:")
    t_eval = time_eval(DOMAIN, EVAL_EPISODES)
    print(f"  {t_eval:.2f} s for {EVAL_EPISODES} episodes "
          f"({1000 * t_eval / EVAL_EPISODES:.0f} ms/episode)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ndummy gradient step ({device}):")
    t_dummy = time_dummy_step(2000, 17, 6, device)
    print(f"  {t_dummy:.2f} s / 2000 steps = {1000 * t_dummy / 2000:.4f} ms/step")

    result = {
        "batch_size": BATCH,
        "env_id": env_id,
        "domain": DOMAIN,
        "s_points": list(S_POINTS),
        "t_setup_s": t_setup,
        "t_sampling_s": sampling,
        "stream_over_ram_ratio": ratio,
        "t_eval_s": t_eval,
        "eval_episodes": EVAL_EPISODES,
        "t_dummy_step_s": t_dummy,
        "device": device,
        "torch": torch.__version__,
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "overhead.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
