"""Round 13/B: approximate-model (AM) probe.

AM is the one estimator that can plausibly fill gap (4) of the estimator
taxonomy: it is fitted after training, so it should not inherit the compared
training axis, but it is policy-sensitive because its value comes from rolling
the policy through a learned dynamics model.

This runner trains one small dynamics ensemble on the offline transitions,
then trains the same IQL configurations used by the cross-configuration study
and evaluates AM from the same episode-start anchors.  It is deliberately a
single registered design: if AM does not fill the gap, the project reports the
negative result rather than tuning AM until it does.

Usage:
    python experiments/e2b_am.py --smoke
    python experiments/e2b_am.py --tag am0 --seeds 0 --lrs 1e-4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(ROOT))

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.gaps.construct import full_dataset, subsample_episodes  # noqa: E402
from orl.methods.harness import (  # noqa: E402
    episode_lookup,
    evaluate_per_episode,
    load,
)
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.ope import (  # noqa: E402
    critic_mean_value,
    policy_action,
)

sys.stdout.reconfigure(line_buffering=True)

REGIME = {
    "walker2d_medium_replay-v2": 0.05,
    "halfcheetah_medium_replay-v2": 0.01,
    "hopper_medium_replay-v2": 1.00,
}
DEFAULT_LRS = (1e-4, 3e-4, 1e-3)
DEFAULT_TAUS = (0.5, 0.7, 0.9)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


class DynamicsMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsEnsemble(nn.Module):
    """Mean prediction over a small ensemble of deterministic MLPs."""

    def __init__(self, obs_dim: int, act_dim: int, members: int, hidden: int = 256):
        super().__init__()
        self.members = nn.ModuleList(
            [DynamicsMLP(obs_dim + act_dim, obs_dim + 2, hidden)
             for _ in range(members)]
        )

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.members], dim=0)
        return preds.mean(dim=0)


def train_dynamics(
    obs: torch.Tensor,
    act: torch.Tensor,
    next_obs: torch.Tensor,
    rew: torch.Tensor,
    terminals: torch.Tensor,
    *,
    steps: int,
    members: int,
    batch_size: int,
    lr: float,
    seed: int,
    dev: str,
) -> DynamicsEnsemble:
    in_dim = obs.shape[1] + act.shape[1]
    model = DynamicsEnsemble(obs.shape[1], act.shape[1], members).to(dev)
    x = torch.cat([obs, act], dim=1)
    y_next = next_obs
    y_rew = rew.reshape(-1, 1)
    y_term = terminals.reshape(-1, 1)
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    # Train members sequentially. Each member sees the same data but a different
    # initialisation and minibatch order, which is enough to expose prediction
    # spread without changing the data distribution.
    for member_idx, member in enumerate(model.members):
        torch.manual_seed(seed + member_idx)
        opt = torch.optim.Adam(member.parameters(), lr=lr)
        for _ in range(steps):
            j = torch.as_tensor(rng.integers(0, n, batch_size), device=dev)
            pred = member(x[j])
            p_next = pred[:, :next_obs.shape[1]]
            p_rew = pred[:, next_obs.shape[1]:next_obs.shape[1] + 1]
            p_term = pred[:, next_obs.shape[1] + 1:next_obs.shape[1] + 2]
            loss = (
                F.mse_loss(p_next, y_next[j])
                + F.mse_loss(p_rew, y_rew[j])
                + F.binary_cross_entropy_with_logits(p_term, y_term[j])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def am_value(
    model: DynamicsEnsemble,
    policy,
    start_states: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
) -> dict:
    """Finite-horizon model rollout under the deterministic policy mean."""

    s = start_states
    ret = torch.zeros(s.shape[0], device=s.device)
    discount = 1.0
    alive = torch.ones(s.shape[0], dtype=torch.bool, device=s.device)
    for _ in range(horizon):
        a = policy_action(policy, s)
        pred = model(torch.cat([s, a], dim=1))
        next_s = pred[:, :s.shape[1]]
        r = pred[:, s.shape[1]]
        term_logit = pred[:, s.shape[1] + 1]
        done = torch.sigmoid(term_logit) > 0.5
        step_alive = alive & (~done)
        ret = ret + discount * r * step_alive.float()
        alive = step_alive
        s = next_s
        discount *= gamma
        if not bool(alive.any()):
            break
    return {"value": float(ret.mean()), "std": float(ret.std())}


def prepare_data(env_id: str, seed: int, args, dev: str):
    d = load(env_id, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {
        "obs": d.obs,
        "act": d.act,
        "rew": d.rew,
        "next_obs": d.next_obs,
        "terminals": d.terminals,
        "timeouts": d.timeouts,
        "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
        "ends": ep_base.ends,
    }
    frac = REGIME.get(env_id, args.subsample)
    small = (
        subsample_episodes("sub", base, frac, seed=seed)
        if frac < 1.0
        else full_dataset("full", base)
    )
    ep = small.episode_lookup()
    obs = to_t(small.obs, dev)
    act = to_t(small.act, dev)
    rew = to_t(small.rew, dev)
    next_obs = to_t(small.next_obs, dev)
    terminals = to_t(small.terminals, dev)
    ep_s = np.concatenate(([0], np.asarray(ep.ends)[:-1] + 1))
    anchor_idx = ep_s[: min(len(ep_s), 32)]
    anchor = obs[torch.as_tensor(anchor_idx, device=dev).long()]
    return d, small, ep, obs, act, rew, next_obs, terminals, anchor


def train_policy(env_id: str, lr: float, tau: float, seed: int, args, dev: str,
                 data):
    d, small, ep, obs, act, rew, next_obs, terminals, anchor = data
    torch.manual_seed(seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=lr),
        max_steps=args.steps,
        tau=tau,
        beta=3.0,
        alpha=0.005,
        discount=0.99,
        episode_lookup=ep,
        coverage=None,
        fixed_n=1,
        n_min=1,
        n_max=1,
    )
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        j = np.sort(rng.integers(0, len(obs), 256))
        agent.update(obs[j], act[j], next_obs[j], rew[j], terminals[j])
    ev = evaluate_per_episode(
        agent.policy, d.domain, d.normer, args.eval_episodes, seed=0
    )

    def matched_value(states):
        with torch.no_grad():
            return agent.qf(states, agent.policy(states).mean)

    return agent, ev, {
        "critic_mean": critic_mean_value(agent.vf, anchor),
        "critic_mean_q": critic_mean_value(matched_value, anchor),
    }


def run_config(env_id: str, lr: float, tau: float, seed: int, args, dev: str,
               data, dynamics: DynamicsEnsemble) -> dict:
    d, small, ep, obs, act, rew, next_obs, terminals, anchor = data
    agent, ev, critic = train_policy(
        env_id, lr, tau, seed, args, dev, data
    )
    am = am_value(
        dynamics,
        agent.policy,
        anchor,
        horizon=args.am_horizon,
        gamma=0.99,
    )
    return {
        "env": d.env_id,
        "lr": lr,
        "tau": tau,
        "seed": seed,
        "n_data": int(small.n),
        "n_episodes": int(len(ep.ends)),
        "true_score": ev["score_mean"],
        "score_sem": ev["score_sem"],
        "per_episode": ev["scores"],
        "critic_mean": critic["critic_mean"],
        "critic_mean_q": critic["critic_mean_q"],
        "am": am["value"],
        "am_std": am["std"],
        "am_horizon": args.am_horizon,
        "am_model_steps": args.model_steps,
        "am_ensemble": args.model_ensemble,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="hopper_medium_replay-v2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lrs", default=",".join(f"{x:g}" for x in DEFAULT_LRS))
    ap.add_argument("--taus", default=",".join(f"{x:g}" for x in DEFAULT_TAUS))
    ap.add_argument("--subsample", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=66_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--model-steps", type=int, default=8_000)
    ap.add_argument("--model-ensemble", type=int, default=4)
    ap.add_argument("--model-batch-size", type=int, default=256)
    ap.add_argument("--model-lr", type=float, default=3e-4)
    ap.add_argument("--am-horizon", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.envs = "hopper_medium_replay-v2"
        args.seeds = "0"
        args.lrs = "1e-4"
        args.taus = "0.5"
        args.steps = 150
        args.eval_episodes = 2
        args.model_steps = 50
        args.model_ensemble = 2
        args.am_horizon = 3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [(float(lr), float(tau)) for lr in args.lrs.split(",")
               for tau in args.taus.split(",")]
    envs = args.envs.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    name = "e2b_am_runs_smoke.jsonl" if args.smoke else (
        f"e2b_am_runs_{args.tag}.jsonl" if args.tag else "e2b_am_runs.jsonl"
    )
    path = BENCH_DIR / name
    runs: list[dict] = []
    done: set[tuple] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                runs.append(row)
                done.add((row["env"], row["lr"], row["tau"], row["seed"]))
    todo = [
        (env_id, lr, tau, seed)
        for env_id in envs
        for lr, tau in configs
        for seed in seeds
        if (env_id, lr, tau, seed) not in done
    ]
    print(f"E2b-AM: {len(todo)} config-runs to go, {len(done)} cached")
    t0 = time.perf_counter()

    # One model per env/seed; reuse it across every lr/tau in this worker.
    data_cache = {}
    model_cache = {}
    for env_id in envs:
        for seed in seeds:
            data = prepare_data(env_id, seed, args, dev)
            data_cache[(env_id, seed)] = data
            d, small, ep, obs, act, rew, next_obs, terminals, anchor = data
            print(
                f"  training dynamics {env_id} seed={seed} "
                f"({small.n} transitions, {args.model_steps} steps x "
                f"{args.model_ensemble})",
                flush=True,
            )
            model_cache[(env_id, seed)] = train_dynamics(
                obs, act, next_obs, rew, terminals,
                steps=args.model_steps,
                members=args.model_ensemble,
                batch_size=args.model_batch_size,
                lr=args.model_lr,
                seed=seed,
                dev=dev,
            )

    for env_id, lr, tau, seed in todo:
        row = run_config(
            env_id, lr, tau, seed, args, dev,
            data_cache[(env_id, seed)],
            model_cache[(env_id, seed)],
        )
        runs.append(row)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"  {env_id.split('-v2')[0]:<12} lr={lr:.0e} tau={tau} seed={seed}"
            f"  true {row['true_score']:6.2f}  am {row['am']:7.2f}  "
            f"critic {row['critic_mean']:7.2f}  "
            f"({(time.perf_counter()-t0)/60:.1f} min)",
            flush=True,
        )
    print(f"\nAM shard '{args.tag or name}' done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
