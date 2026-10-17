"""Run the VENDORED reference IQL (gwthomas/IQL-PyTorch) on our data.

This is the "swap the baseline" step. No code from our own `orl.algos.iql`
is involved -- we import ImplicitQLearning / TwinQ / ValueFunction /
GaussianPolicy straight from `vendor/IQL-PyTorch/src/`, and only replace the
d4rl replay buffer with our h5py loader and the gym eval with our own.

It also reproduces the reference's reward rescaling, which our own IQL was
missing entirely:

    dataset['rewards'] /= (max_ret - min_ret)
    dataset['rewards'] *= max_episode_steps

So the correct fixed point here is the SCALED one, not the raw
r/(1-gamma). We print the expected target so it can be checked directly.

    python experiments/run_reference_iql.py --steps 200000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "IQL-PyTorch"))
sys.path.insert(0, str(ROOT / "src"))

from src.iql import ImplicitQLearning  # noqa: E402  (vendored reference)
from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.data.normalize import ObsNormalizer  # noqa: E402

MAX_EPISODE_STEPS = 1000


def evaluate_policy(policy, domain: str, n_episodes: int = 10, seed: int = 0) -> dict:
    """Roll out the learned policy and score it the way the paper does."""
    import gymnasium as gym

    from orl.envs.obs_layout import ENV_IDS
    from orl.envs.ref_scores import normalized_score
    from orl.data.normalize import ObsNormalizer

    norm = _STATE["norm"]
    env = gym.make(ENV_IDS[domain])
    rets = []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            ot = torch.as_tensor(norm(o[None, :]), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                a = policy(ot).mean.cpu().numpy()[0]
            o, r, term, trunc, _ = env.step(np.clip(a, -1, 1))
            total += float(r)
            done = term or trunc
        rets.append(total)
    env.close()
    rets = np.asarray(rets)
    return {
        "raw_mean": float(rets.mean()), "raw_std": float(rets.std()),
        "score_mean": normalized_score(domain, float(rets.mean())),
    }


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_STATE: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--every", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7, help="expectile")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.005, help="EMA coefficient")
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--no-normalize", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    ds = D4RLDataset(a.env, mode="ram")
    obs = ds.get("observations")
    act = ds.get("actions")
    rew = ds.get("rewards").astype(np.float32)
    nobs = ds.get("next_observations")
    term = ds.get("terminals").astype(np.float32)

    # --- the reference's reward rescaling, which our own IQL omitted entirely
    rets = ds.episodes.returns(rew)
    min_ret, max_ret = float(rets.min()), float(rets.max())
    rew_scaled = (rew / (max_ret - min_ret)) * MAX_EPISODE_STEPS
    print(f"dataset returns in [{min_ret:.1f}, {max_ret:.1f}]")
    print(f"reward: raw mean={rew.mean():.4f}  ->  "
          f"scaled mean={rew_scaled.mean():.4f}  "
          f"(x{rew_scaled.mean() / rew.mean():.4f})")
    expected_v = rew_scaled.mean() / (1 - a.discount)
    print(f"expected fixed point (scaled) = {expected_v:.1f}")
    print(f"  (our own IQL was chasing the RAW scale "
          f"{rew.mean() / (1 - a.discount):.1f} -- never reachable for it)\n")

    if a.no_normalize:
        obs = obs.astype(np.float32)
        nobs = nobs.astype(np.float32)
        norm_tag = "off"
        _STATE["norm"] = ObsNormalizer(mean=np.zeros(obs.shape[1]),
                                       std=np.ones(obs.shape[1]))
    else:
        nm = ObsNormalizer.fit(obs)
        _STATE["norm"] = nm
        obs = nm(obs)
        nobs = nm(nobs)
        norm_tag = "on"

    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    O, A, R, NO, TE = T(obs), T(act), T(rew_scaled), T(nobs), T(term)

    obs_dim, act_dim = obs.shape[1], act.shape[1]
    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim),
        vf=ValueFunction(obs_dim),
        policy=GaussianPolicy(obs_dim, act_dim),
 optimizer_factory=lambda p: torch.optim.Adam(p, lr=a.lr),
        max_steps=a.steps, tau=a.tau, beta=a.beta,
        alpha=a.alpha, discount=a.discount,
    )

    probe = np.sort(np.random.default_rng(7).choice(len(O), 4096, replace=False))
    pO = O[probe]
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    print(f"obs normalize={norm_tag}  steps={a.steps:,}")
    print(f"{'step':>10}{'V':>11}{'Q_target':>11}{'V_std':>10}")
    for i in range(1, a.steps + 1):
        j = torch.as_tensor(rng.integers(0, len(O), a.batch_size), device=dev)
        iql.update(O[j], A[j], NO[j], R[j], TE[j])
        if i % a.every == 0 or i == a.steps:
            with torch.no_grad():
                V = iql.vf(pO).squeeze(-1)
                Q = iql.q_target(pO, A[probe]).squeeze(-1)
            print(f"{i:>10,}{V.mean():>11.2f}{Q.mean():>11.2f}{V.std():>10.2f}",
                flush=True)
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min")

    domain = a.env.split("_")[0]
    ev = evaluate_policy(iql.policy, domain, n_episodes=20)
    print("policy evaluation (20 episodes):")
    print(f"  raw return      = {ev['raw_mean']:.1f} +- {ev['raw_std']:.1f}")
    print(f"  normalized score= {ev['score_mean']:.1f}")
    print(f"  (published IQL on {a.env}: ~44-46 normalized)")
    torch.save(iql.state_dict(), "results/benchmarks/ref_iql_final.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
