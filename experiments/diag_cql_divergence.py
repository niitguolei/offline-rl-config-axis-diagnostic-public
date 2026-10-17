"""Why does the CQL arm's critic reach 2e3 when FQE on the same data says 30?

The pre-run gate caught this on walker2d-medium-replay at the 5% episode
subsample: the evaluated policy scored -0.24 while the critic mean read 2224.
FQE, fitted to the same data under the same policy, read 30. So the data
supports a value near 30 and something in the CQL training loop is running away.

This separates the two candidates:

    min_q_weight = 0   pure TD bootstrapping, the conservative term switched off
    min_q_weight = w   the real thing

If the Q magnitudes blow up with the weight at zero, the fault is in the backup
(or in the policy that supplies its action); if they only blow up with the
weight on, the fault is in the transcription of the conservative term.

Usage:
    python experiments/diag_cql_divergence.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.value_functions import TwinQ  # noqa: E402

from e2b_cql import REGIME  # noqa: E402
from e2b_crossconfig import to_t  # noqa: E402
from orl.algos.cql import ConservativeQLearning  # noqa: E402
from orl.algos.tanh_policy import TanhGaussianPolicy  # noqa: E402
from orl.gaps.construct import full_dataset, subsample_episodes  # noqa: E402
from orl.methods.harness import episode_lookup, load  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="walker2d_medium_replay-v2")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--report-every", type=int, default=2_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy-lr", type=float, default=None,
                    help="the authors use policy_lr=1e-4 with qf_lr=3e-4; the "
                         "runner passes one lr for both unless this is set")
    ap.add_argument("--policy-eval-start", type=int, default=0)
    ap.add_argument("--weights", default="0,1,10")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(args.env, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    frac = REGIME.get(args.env, 0.05)
    small = (subsample_episodes("sub", base, frac, seed=0)
             if frac < 1.0 else full_dataset("full", base))
    print(f"{args.env}: frac={frac}  n_data={small.n}  n_episodes={len(small.episode_lookup().ends)}")
    print(f"  actions   min={small.act.min():+.3f}  max={small.act.max():+.3f}  "
          f"mean|a|={np.abs(small.act).mean():.3f}")
    print(f"  rewards   mean={small.rew.mean():+.4f}  std={small.rew.std():.4f}  "
          f"min={small.rew.min():+.3f}  max={small.rew.max():+.3f}")
    print(f"  terminals {int(np.asarray(small.terminals).sum())} of {small.n}")

    O, A = to_t(small.obs, dev), to_t(small.act, dev)
    R, NO, TE = to_t(small.rew, dev), to_t(small.next_obs, dev), to_t(small.terminals, dev)

    for w in [float(x) for x in args.weights.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            torch.manual_seed(seed)
            agent = ConservativeQLearning(
                qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
                policy=TanhGaussianPolicy(d.obs.shape[1], d.act.shape[1]),
                optimizer_factory=lambda p, lr=args.lr: torch.optim.Adam(p, lr=lr),
                max_steps=args.steps,
                min_q_weight=w,
                policy_eval_start=args.policy_eval_start,
                device=dev,
            )
            if args.policy_lr is not None:
                for group in agent.policy_optimizer.param_groups:
                    group["lr"] = args.policy_lr
                agent.policy_lr_schedule.base_lrs = [args.policy_lr]
            rng = np.random.default_rng(seed)
            print(f"\nmin_q_weight={w}  policy_lr={args.policy_lr or args.lr}  "
                  f"policy_eval_start={args.policy_eval_start}")
            for step in range(1, args.steps + 1):
                j = np.sort(rng.integers(0, len(O), args.batch))
                agent.update(observations=O[j], actions=A[j], next_observations=NO[j],
                             rewards=R[j], terminals=TE[j])
                if step % args.report_every == 0 or step == args.steps:
                    with torch.no_grad():
                        a_data = A[j]
                        a_rand = torch.empty_like(a_data).uniform_(-1.0, 1.0)
                        a_pol = agent.policy(O[j]).mean
                        q_data = agent.qf(O[j], a_data)
                        q_rand = agent.qf(O[j], a_rand)
                        q_pol = agent.qf(O[j], a_pol)
                        target = R[j] + 0.99 * agent.q_target(NO[j], agent.policy(NO[j]).rsample())
                    alpha = float(agent.log_alpha.exp()) if agent.use_automatic_entropy_tuning else 1.0
                    print(f"   step {step:6d}  |Q| data {q_data.abs().mean():9.2f}  "
                          f"rand {q_rand.abs().mean():9.2f}  pi {q_pol.abs().mean():9.2f}  "
                          f"target {target.abs().mean():9.2f}  "
                          f"log_std {agent.policy.log_std.mean():+.2f}  alpha {alpha:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
