"""Where do the 18 ms per CQL step go?

The CQL arm at 66k steps takes ~20 min per run while the GPU sits at ~19% and
uses 774 MB of 8 GB. That is the signature of a launch-bound loop, not a
compute-bound one, so the question is which of the ~40 small ops per step is
worth merging.

This times the step as it is, then with the per-step GPU synchronisations
removed and with the three conservative-term Q evaluations merged into one. It
does not change any algorithm code; the variants are built here so the numbers
are comparable inside one process.

Usage:
    python experiments/bench_cql_step.py --steps 200
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import types

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.value_functions import TwinQ  # noqa: E402

from orl.algos.cql import ConservativeQLearning  # noqa: E402
from orl.algos.tanh_policy import TanhGaussianPolicy  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def merged_cql_term(self, obs, q_preds, action_sets):
    """`_cql_term` with the three Q evaluations fused into one batch.

    Same values, same order (rand, next, curr as the source's cat), one obs
    repeat and two MLP calls instead of three of each.
    """
    n = self.num_random
    acts = torch.cat([action_sets["random"], action_sets["next"],
                      action_sets["curr"]], dim=1)
    q1, q2 = self._q_values(obs, acts)
    q1_rand, q1_next, q1_curr = q1[:, :n], q1[:, n:2 * n], q1[:, 2 * n:]
    q2_rand, q2_next, q2_curr = q2[:, :n], q2[:, n:2 * n], q2[:, 2 * n:]

    action_dim = self.policy.log_std.shape[0]
    if self.min_q_version == 3:
        random_density = float(np.log(0.5 ** action_dim))
        curr_log = action_sets["curr_log_pi"]
        next_log = action_sets["next_log_pi"]
        cat_q1 = torch.cat([q1_rand - random_density, q1_next - next_log,
                            q1_curr - curr_log], dim=1)
        cat_q2 = torch.cat([q2_rand - random_density, q2_next - next_log,
                            q2_curr - curr_log], dim=1)
    else:
        raise ValueError("benchmark only covers min_q_version=3")

    def penalty(cat_q, q_pred):
        lse = torch.logsumexp(cat_q / self.temp, dim=1).mean() * self.temp
        return (lse - q_pred.mean()) * self.min_q_weight

    return penalty(cat_q1, q_preds[0]), penalty(cat_q2, q_preds[1])


def make_agent(obs_dim: int, act_dim: int, weight: float, dev: str, steps: int):
    torch.manual_seed(0)
    return ConservativeQLearning(
        qf=TwinQ(obs_dim, act_dim),
        policy=TanhGaussianPolicy(obs_dim, act_dim),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=steps,
        min_q_weight=weight,
        device=dev,
    )


def time_steps(agent, obs, act, next_obs, rew, term, steps: int, batch: int,
               seed: int, sync_each: bool = True) -> float:
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    if sync_each:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        j = torch.as_tensor(np.sort(rng.integers(0, n, batch)), device=obs.device)
        agent.update(observations=obs[j], actions=act[j], next_observations=next_obs[j],
                     rewards=rew[j], terminals=term[j])
    if sync_each:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--act-dim", type=int, default=6)
    ap.add_argument("--profile", action="store_true",
                    help="print the top ops by self CPU time instead of timings")
    ap.add_argument("--compile", action="store_true",
                    help="also time a torch.compile(reduce-overhead) variant")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  steps={args.steps}  batch={args.batch}  "
          f"obs_dim={args.obs_dim}  act_dim={args.act_dim}")

    torch.manual_seed(0)
    obs = torch.randn(20000, args.obs_dim, device=dev)
    act = (torch.rand(20000, args.act_dim, device=dev) * 2 - 1)
    next_obs = torch.randn(20000, args.obs_dim, device=dev)
    rew = torch.randn(20000, device=dev)
    term = torch.zeros(20000, device=dev)

    base = make_agent(args.obs_dim, args.act_dim, 5.0, dev, args.steps + args.warmup)
    time_steps(base, obs, act, next_obs, rew, term, args.warmup, args.batch, seed=0)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            time_steps(base, obs, act, next_obs, rew, term, args.steps, args.batch,
                       seed=1, sync_each=False)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
        return 0

    t_base = time_steps(base, obs, act, next_obs, rew, term, args.steps, args.batch, seed=1)
    print(f"  baseline                      {t_base*1000:7.2f} ms/step")

    merged_agent = make_agent(args.obs_dim, args.act_dim, 5.0, dev, args.steps + args.warmup)
    merged_agent._cql_term = types.MethodType(merged_cql_term, merged_agent)
    time_steps(merged_agent, obs, act, next_obs, rew, term, args.warmup, args.batch, seed=0)
    t_merged = time_steps(merged_agent, obs, act, next_obs, rew, term, args.steps,
                          args.batch, seed=1)
    print(f"  merged conservative term       {t_merged*1000:7.2f} ms/step  "
          f"({t_base/t_merged:.2f}x)")

    if args.compile:
        # 578 kernel launches per step at 13 us each is the bottleneck, so the
        # lever is fusion, not batch size. `both` is a separate method from
        # `forward`, so compiling the module alone would leave the hot path
        # (which calls `both`) uncompiled.
        comp = make_agent(args.obs_dim, args.act_dim, 5.0, dev, args.steps + args.warmup)
        comp._cql_term = types.MethodType(merged_cql_term, comp)
        for mod in (comp.qf, comp.q_target):
            mod.both = torch.compile(mod.both, mode="reduce-overhead")
            mod.forward = torch.compile(mod.forward, mode="reduce-overhead")
        comp.policy.net = torch.compile(comp.policy.net, mode="reduce-overhead")
        time_steps(comp, obs, act, next_obs, rew, term, args.warmup, args.batch, seed=0)
        t_comp = time_steps(comp, obs, act, next_obs, rew, term, args.steps,
                            args.batch, seed=1)
        print(f"  merged + torch.compile        {t_comp*1000:7.2f} ms/step  "
              f"({t_base/t_comp:.2f}x)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
