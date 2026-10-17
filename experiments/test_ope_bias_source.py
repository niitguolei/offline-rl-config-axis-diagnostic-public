"""Why is FQE biased by 38% here? Test the action-extrapolation hypothesis.

Measured: after the divergence fix, three independent FQE configurations all
converge to 40.0-40.1 against an MC-anchored truth of 64.905. Reproducible across
target/loss settings means a systematic bias, not optimisation lag.

Hypothesis: FQE fits Q(s, a_beta) on the dataset's actions but bootstraps with
Q(s', pi(s')), so when pi differs from the behaviour policy the bootstrap needs
to EXTRAPOLATE across the action dimension and the error is systematic.

Falsifiable test: evaluate a policy that is close to the behaviour policy. A BC
policy is the standard proxy -- it is fit to the dataset's actions directly. If
the relative bias shrinks sharply for BC relative to IQL, the hypothesis holds.
If BC shows the same ~38%, the source is something else and must be found before
FQE enters any failure map.

    python experiments/test_ope_bias_source.py --smoke
    python experiments/test_ope_bias_source.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(ROOT))

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.algos.bc import BC, BCConfig  # noqa: E402
from orl.algos.mc_reference import mc_reference_batched  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.methods.harness import episode_lookup, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.ope import fitted_q_evaluation, policy_action  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


class Adapter(torch.nn.Module):
    """Give a callable policy a parameter, so device detection works."""

    def __init__(self, fn, device):
        super().__init__()
        self.fn = fn
        self.dummy = torch.nn.Parameter(torch.zeros(1, device=device))

    def forward(self, obs):
        return self.fn(obs)


def raw_policy_fn(policy, normer, dev):
    mu = torch.as_tensor(normer.mean, dtype=torch.float64, device=dev)
    sd = torch.as_tensor(normer.std, dtype=torch.float64, device=dev)

    def fn(obs_raw):
        o = ((obs_raw.to(dev).to(torch.float64) - mu) / sd).to(torch.float32)
        return policy_action(policy, o)

    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--train-steps", type=int, default=20_000)
    ap.add_argument("--fqe-steps", type=int, default=20_000)
    ap.add_argument("--mc-episodes", type=int, default=64)
    ap.add_argument("--mc-horizon", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_steps, args.fqe_steps = 1000, 1000
        args.mc_episodes, args.mc_horizon = 8, 100

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep = episode_lookup(d.env_id)
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    with D4RLDataset(d.env_id, mode="stream") as ds:
        raw_obs = ds.get("observations")
    raw_starts = raw_obs[starts][: args.mc_episodes].astype(np.float64)

    O, A = to_t(d.obs, dev), to_t(d.act, dev)
    R, NO, TE = to_t(d.rew, dev), to_t(d.next_obs, dev), to_t(d.terminals, dev)
    ss = O[torch.as_tensor(starts[: args.mc_episodes], device=dev).long()]
    pick = np.sort(np.random.default_rng(0).choice(
        len(O), size=min(4096, len(O)), replace=False))
    probe, probe_act = O[pick], A[pick]

    bc = BC(BCConfig(obs_dim=d.obs.shape[1], act_dim=d.act.shape[1], device=dev))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.train_steps):
        j = np.sort(rng.integers(0, len(O), 256))
        bc.update({"observations": d.obs[j], "actions": d.act[j]})
    bc_pol = Adapter(bc.policy_fn(), dev)

    torch.manual_seed(args.seed)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.train_steps, tau=0.7, beta=3.0, alpha=0.005,
        discount=0.99, episode_lookup=ep,
        coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(args.seed)
    for _ in range(args.train_steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(O[j], A[j], NO[j], R[j], TE[j])
    print(f"trained BC and IQL {args.train_steps:,} steps "
          f"({time.perf_counter()-t0:.0f}s)")

    rows = []
    for name, pol in (("bc(~behaviour)", bc_pol), ("iql", agent.policy)):
        pf = raw_policy_fn(pol, d.normer, dev)
        mc = mc_reference_batched(d.domain, raw_starts, pf, gamma=0.99,
                                  horizon=args.mc_horizon, device=dev,
                                  batch_size=32)
        truth = float((d.reward_scale * mc.values).mean())
        sem = float(d.reward_scale * mc.values.std() / np.sqrt(len(mc.values)))
        res = fitted_q_evaluation(pol, O, NO, R, TE, ss, gamma=0.99,
                                  steps=args.fqe_steps, seed=args.seed,
                                  device=dev, loss="huber", target_agg="mean",
                                  alpha=0.05, grad_clip=10.0, lr=1e-3)
        denom = max(abs(truth), 3.0 * sem, 1e-6)
        bias = abs(res["value"] - truth) / denom
        with torch.no_grad():
            pa = policy_action(pol, probe)
        row = {"policy": name, "mc_truth": truth, "mc_sem": sem,
               "fqe": res["value"], "rel_bias": bias,
               "action_gap_mean": float((pa - probe_act).abs().mean()),
               "action_std_ratio": float(pa.std(0).mean()
                                         / probe_act.std(0).mean()),
               "trace": res["value_trace"]}
        rows.append(row)
        print(f"  {name:<16} truth {truth:7.2f}  fqe {res['value']:7.2f}  "
              f"bias {bias:6.1%}  ||pi-a_beta|| {row['action_gap_mean']:.3f}  "
              f"std ratio {row['action_std_ratio']:.2f}")

    bc_bias, iql_bias = rows[0]["rel_bias"], rows[1]["rel_bias"]
    supported = bc_bias < 0.5 * iql_bias
    verdict = ("HYPOTHESIS SUPPORTED: bias shrinks when the policy is close to "
               "the behaviour policy" if supported else
               "HYPOTHESIS NOT SUPPORTED: BC is biased too -- source is elsewhere")
    print(f"\n{verdict}")
    print(f"  bc bias {bc_bias:.1%}  vs  iql bias {iql_bias:.1%}  "
          f"(ratio {bc_bias/max(iql_bias,1e-9):.2f})")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("ope_bias_source_smoke.json" if args.smoke
                     else "ope_bias_source.json")
    p.write_text(json.dumps({"env": d.env_id, "rows": rows,
                             "verdict": verdict, "supported": bool(supported),
                             "elapsed_min": (time.perf_counter() - t0) / 60.0},
                            indent=2), encoding="utf-8")
    print(f"wrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
