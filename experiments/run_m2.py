"""T4 -- M2: support-aware per-sample backup length, on the reference baseline.

Same data pipeline and same reward rescaling as the verified reference run, so
the ONLY difference from `run_reference_iql.py` is which agent class is built.

    python experiments/run_m2.py --env halfcheetah_medium_replay-v2 --steps 100000
    python experiments/run_m2.py --smoke
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

from src.policy import GaussianPolicy  # noqa: E402
from src.value_functions import TwinQ, ValueFunction  # noqa: E402

from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.data.episode import EpisodeIndexLookup  # noqa: E402
from orl.methods.harness import evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402
def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--every", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7, help="expectile")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="m2")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.every, a.eval_episodes = 2000, 1000, 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    d = load(a.env, with_coverage=True, normalize=True)
    print(f"env            = {d.env_id}")
    print(f"returns range  = [{d.ret_range[0]:.1f}, {d.ret_range[1]:.1f}]")
    print(f"reward         = raw {d.raw_rew_mean:.4f} -> scaled {d.scaled_rew_mean:.4f}")
    print(f"fixed point    = {d.scaled_rew_mean / (1 - a.discount):.1f} (scaled)")

    O, A, R = to_t(d.obs, dev), to_t(d.act, dev), to_t(d.rew, dev)
    NO, TE = to_t(d.next_obs, dev), to_t(d.terminals, dev)

    with D4RLDataset(d.env_id, mode="stream") as ds:
        ep = EpisodeIndexLookup(ds.episodes.starts(), ds.episodes.ends, ds.n)

    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=a.lr),
        max_steps=a.steps, tau=a.tau, beta=a.beta, alpha=a.alpha,
        discount=a.discount,
        episode_lookup=ep, coverage=d.coverage,
        n_min=a.n_min, n_max=a.n_max,
    )
    s = agent.requested_n_summary()
    print(f"n-step policy  = edges {s['edges'][0]:.2f}/{s['edges'][1]:.2f}, "
          f"hist {s['requested_n_hist']}")
    print()

    probe = np.sort(np.random.default_rng(7).choice(len(O), 4096, replace=False))
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    print(f"{'step':>10}{'V':>10}{'Q_tgt':>10}{'q_loss':>12}{'v_loss':>12}")
    for i in range(1, a.steps + 1):
        j = np.sort(rng.integers(0, len(O), a.batch_size))
        out = agent.update(O[j], A[j], NO[j], R[j], TE[j],
                           indices=j, dataset_rewards=d.rew,
                           dataset_next_obs=d.next_obs,
                           dataset_terminals=d.terminals)
        if i % a.every == 0 or i == a.steps:
            with torch.no_grad():
                V = agent.vf(O[probe]).squeeze(-1)
                Q = agent.q_target(O[probe], A[probe]).squeeze(-1)
            print(f"{i:>10,}{V.mean():>10.2f}{Q.mean():>10.2f}"
                  f"{out['q_loss']:>12.3f}{out['value_loss']:>12.3f}", flush=True)

    ev = evaluate(agent.policy, d.domain, d.normer, n_episodes=a.eval_episodes)
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min")
    print(f"policy raw={ev['raw_mean']:.1f} +- {ev['raw_std']:.1f}  "
          f"normalized={ev['score_mean']:.1f}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"env": a.env, "steps": a.steps, "tag": a.tag, "n_policy": s,
           "eval": ev, "ret_range": list(d.ret_range),
           "scaled_rew_mean": d.scaled_rew_mean}
    outp = BENCH_DIR / f"{a.tag}_{a.env}_{a.steps}.json"
    outp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"wrote {outp.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
