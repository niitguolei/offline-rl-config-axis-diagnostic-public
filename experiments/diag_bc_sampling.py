"""DEC-3 evidence: is hopper's low BC ratio caused by mean-action averaging?

Hypothesis (docs/GATE0报告.md section 6): hopper terminates on a hard
healthy-range threshold and the expert policy sits near that boundary making
small corrective adjustments. BC learns the conditional MEAN of those
adjustments, and the mean of a boundary-hugging corrective distribution is a
"lazy" action that lets the state drift out of the healthy set.

Falsifiable prediction: if that is the cause, then evaluating the SAME trained
policy with actions SAMPLED from the learned Gaussian (instead of the mean)
should recover a large part of the return.

Control: walker2d, where the mean already achieves ratio 0.93. If sampling
also helps there, the effect is not hopper-specific and the hypothesis is weak.

    python experiments/diag_bc_sampling.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from orl.algos.bc import BC, BCConfig
from orl.data.d4rl_hdf5 import D4RLDataset
from orl.envs.gym_eval import evaluate

sys.stdout.reconfigure(line_buffering=True)

BC_STEPS = 100_000
BATCH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sampling_policy(bc: BC, deterministic: bool):
    """Wrap a trained BC as an evaluation policy."""
    cfg = bc.cfg
    lo, hi = cfg.act_low, cfg.act_high

    def fn(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, log_std = bc.net(obs.to(cfg.device)).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -5.0, 2.0)
            a = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        return a.clamp(lo, hi)

    return fn


def run(domain: str, seed: int = 0) -> dict:
    env_id = f"{domain}_expert-v2"
    with D4RLDataset(env_id, mode="ram") as ds:
        data_ret = float(np.mean(ds.episodes.returns(ds.get("rewards"))))
        bc = BC(BCConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device=DEVICE))
        rng = np.random.default_rng(seed)
        obs, act = ds.get("observations"), ds.get("actions")
        t0 = time.perf_counter()
        for i in range(BC_STEPS):
            idx = rng.integers(0, ds.n, BATCH)
            bc.update({"observations": obs[idx], "actions": act[idx]})
            if (i + 1) % 50_000 == 0:
                print(f"      {i + 1:,}/{BC_STEPS:,}", flush=True)
        train_s = time.perf_counter() - t0
        probe = torch.as_tensor(obs[:4096], dtype=torch.float32, device=DEVICE)
        learned_std = float(
            bc.net(probe).chunk(2, dim=-1)[1].clamp(-5.0, 2.0).exp().mean()
        )

    mean_r = evaluate(domain, sampling_policy(bc, True), n_episodes=10, seed=0)
    samp_means = [
        evaluate(domain, sampling_policy(bc, False), n_episodes=10, seed=100 * k).raw_mean
        for k in range(3)
    ]
    samp_r = float(np.mean(samp_means))
    return {
        "domain": domain, "data_return": data_ret,
        "mean_return": mean_r.raw_mean, "mean_ratio": mean_r.raw_mean / data_ret,
        "sample_return": samp_r, "sample_ratio": samp_r / data_ret,
        "learned_std": learned_std, "train_s": train_s,
    }


def main() -> int:
    print(f"BC_STEPS={BC_STEPS} device={DEVICE}")
    print("(train once per domain, evaluate the SAME policy two ways)\n")
    hdr = ("%-11s %10s %10s %9s %11s %9s %10s"
           % ("domain", "data_ret", "mean_ret", "mean_r", "samp_ret", "samp_r", "learned_sd"))
    print(hdr)
    print("-" * len(hdr))
    for domain in ("hopper", "walker2d"):
        r = run(domain)
        row = ("%-11s %10.1f %10.1f %9.4f %11.1f %9.4f %10.4f"
               % (r["domain"], r["data_return"], r["mean_return"], r["mean_ratio"],
                  r["sample_return"], r["sample_ratio"], r["learned_std"]))
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

