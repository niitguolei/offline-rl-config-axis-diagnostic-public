"""E2b, third algorithm: TD3+BC. The registered prediction's negative case.

IQL puts tau in the critic's objective and CQL puts alpha there; both critics
turn out to be deterministic functions of the axis (48 of 48 slices at exactly
+/-1.00). TD3+BC puts its alpha in the ACTOR loss only, so the mechanism
under test -- an estimator inherits the hyperparameter its training objective
contains -- predicts the opposite here.

The prediction is registered in docs/第十一轮_同批对照.md section 7, before
these runs exist:

  P1  the fraction of slices whose critic-to-alpha rank correlation is exactly
      +/-1.00 will be < 7/9, against 48/48 for IQL/CQL and 24/48 for FQE
  P2  alpha's share of the critic's variance will be < 0.5, against 0.98 for
      IQL/hopper and 0.80 for CQL/hopper

An indirect path is recorded up front: the critic target contains pi'(s'), and
pi' depends on alpha. So the prediction is "substantially weaker", not "no
dependence".

Everything else is shared with the other arms -- same harness, same reward
rescaling, same 30-episode evaluation, same FQE, same anchors -- so the three
algorithms differ only in the algorithm.

Usage:
    python experiments/e2b_td3bc.py --smoke
    python experiments/e2b_td3bc.py --no-is --tag t0 --seeds 0
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
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.policy import DeterministicPolicy  # noqa: E402
from src.value_functions import TwinQ  # noqa: E402

from e2b_crossconfig import REGIME, analyse, to_t  # noqa: E402
from orl.algos.td3bc import TD3BC  # noqa: E402
from orl.gaps.construct import full_dataset, subsample_episodes  # noqa: E402
from orl.methods.harness import (  # noqa: E402
    episode_lookup,
    evaluate_per_episode,
    load,
)
from orl.paths import BENCH_DIR  # noqa: E402
from orl.selection.is_ope import fit_behaviour, pdis  # noqa: E402
from orl.selection.ope import critic_mean_value, fitted_q_evaluation  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

FQE_CFG = dict(loss="huber", target_agg="mean", alpha=0.05, grad_clip=10.0, lr=1e-3)
DEFAULT_LRS = (1e-4, 3e-4, 1e-3)
# The authors use alpha=2.5; the grid keeps the two extremes an order of
# magnitude apart so the axis has room to act if it acts at all.
DEFAULT_ALPHAS = (0.5, 2.5, 10.0)


def run_config(env_id: str, lr: float, alpha: float, seed: int, args,
               dev: str) -> dict:
    d = load(env_id, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    frac = REGIME.get(env_id, args.subsample)
    small = (subsample_episodes("sub", base, frac, seed=seed)
             if frac < 1.0 else full_dataset("full", base))
    ep = small.episode_lookup()
    O, A = to_t(small.obs, dev), to_t(small.act, dev)
    R, NO, TE = to_t(small.rew, dev), to_t(small.next_obs, dev), to_t(small.terminals, dev)
    ep_s = np.concatenate(([0], np.asarray(ep.ends)[:-1] + 1))
    anchor_idx = ep_s[: min(len(ep_s), 32)]
    anchor = O[torch.as_tensor(anchor_idx, device=dev).long()]
    beh = (None if args.no_is else
           fit_behaviour(small.obs, small.act, steps=args.behaviour_steps,
                         device=dev, seed=seed))
    torch.manual_seed(seed)
    agent = TD3BC(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        policy=DeterministicPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=lr),
        max_steps=args.steps,
        alpha=alpha,
        discount=0.99,
        soft_target_tau=5e-3,
        device=dev,
    )
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        j = np.sort(rng.integers(0, len(O), 256))
        agent.update(observations=O[j], actions=A[j], next_observations=NO[j],
                     rewards=R[j], terminals=TE[j])
    ev = evaluate_per_episode(agent.policy, d.domain, d.normer,
                              args.eval_episodes, seed=0)
    pol = agent.policy
    fqe = fitted_q_evaluation(pol, O, NO, R, TE, anchor, gamma=0.99,
                              steps=args.fqe_steps, seed=seed, device=dev,
                              **FQE_CFG)
    pd = {"value": None, "ess": None}
    if beh is not None:
        with torch.no_grad():
            log_r = (pol(O).log_prob(A) - beh.log_prob(O, A)).cpu().numpy()
        pd = pdis(np.asarray(small.rew, dtype=np.float64), ep_s,
                  np.asarray(ep.ends), log_r, gamma=0.99, clip=2.0)

    def own_value(states):
        """TD3+BC has no V, so the free baseline is min(Q1,Q2) at pi(s).

        That is the matched functional -- the same one the CQL arm uses and the
        one the S5 re-run recorded for IQL as critic_mean_q -- so the three
        algorithms are compared on like for like.
        """
        with torch.no_grad():
            return agent.qf(states, agent.policy(states))

    critic = critic_mean_value(own_value, anchor)
    fqe_value = float(fqe["value"])
    ratio = abs(critic) / max(abs(fqe_value), 1e-6)
    return {"env": d.env_id, "lr": lr, "alpha": alpha, "seed": seed,
            "n_data": int(small.n), "n_episodes": int(len(ep.ends)),
            "true_score": ev["score_mean"], "score_sem": ev["score_sem"],
            "per_episode": ev["scores"],
            "critic_mean": critic, "fqe": fqe_value,
            "critic_over_fqe": ratio, "diverged": bool(not np.isfinite(critic)
                                                       or ratio > 20.0),
            "pdis": pd["value"], "pdis_ess": pd.get("ess")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="hopper_medium_replay-v2")
    ap.add_argument("--no-is", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lrs", default=",".join(f"{x:g}" for x in DEFAULT_LRS))
    ap.add_argument("--alphas", default=",".join(f"{x:g}" for x in DEFAULT_ALPHAS))
    ap.add_argument("--subsample", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=66_000)
    ap.add_argument("--fqe-steps", type=int, default=6000)
    ap.add_argument("--behaviour-steps", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.envs, args.seeds = "hopper_medium_replay-v2", "0"
        args.steps, args.fqe_steps, args.behaviour_steps = 1500, 300, 300
        args.eval_episodes = 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [(float(lr), float(a)) for lr in args.lrs.split(",")
               for a in args.alphas.split(",")]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    name = ("e2b_td3bc_runs_smoke.jsonl" if args.smoke
            else f"e2b_td3bc_runs_{args.tag}.jsonl" if args.tag
            else "e2b_td3bc_runs.jsonl")
    p = BENCH_DIR / name
    runs, done = [], set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs.append(r)
                done.add((r["env"], r["lr"], r["alpha"], r["seed"]))
    todo = [(e, lr, a, s) for e in args.envs.split(",")
            for (lr, a) in configs
            for s in [int(x) for x in args.seeds.split(",")]
            if (e, lr, a, s) not in done]
    print(f"E2b-TD3BC: {len(todo)} config-runs to go, {len(done)} cached "
          f"({len(configs)} configs x seeds x envs)")
    t0 = time.perf_counter()
    for env_id, lr, a, seed in todo:
        r = run_config(env_id, lr, a, seed, args, dev)
        runs.append(r)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"  {env_id.split('-v2')[0]:<12} lr={lr:.0e} a={a:5.1f} seed={seed}"
              f"  true {r['true_score']:6.2f} +- {r['score_sem']:4.2f}  "
              f"critic {r['critic_mean']:8.2f}  fqe {r['fqe']:7.2f}  "
              f"{'DIVERGED' if r['diverged'] else 'ok':>8}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    if args.tag:
        print(f"\nshard '{args.tag}' done")
        return 0
    print("\nwrote", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
