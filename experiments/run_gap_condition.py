"""Train one condition on a controlled-gap dataset. The T1 integration.

The point of this script is the interface, not the number: a Gapped dataset must
be trained on with `Gapped.episode_lookup()`, never with
`harness.episode_lookup()`. Fragment ends created by a drop carry NO terminal
flag, so a lookup derived from `terminals | timeouts` would let n-step windows
step across the cut -- with every array shape still correct and nothing raised.
That is the failure this file exists to make impossible, by taking the lookup
from the object that knows.

It also writes the gap characterisation next to the training result, so a
condition can never be reported without the evidence that it is the type of gap
it claims to be.

    python experiments/run_gap_condition.py --gap ball --fraction 0.3 --smoke
    python experiments/run_gap_condition.py --gap action --fraction 0.3
"""
from __future__ import annotations

import argparse
import copy
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

from orl.gaps.construct import action_gap, state_gap, subsample_episodes  # noqa: E402
from orl.methods.harness import episode_lookup, evaluate, load  # noqa: E402
from orl.methods.support_aware import SupportAwareIQL  # noqa: E402
from orl.paths import BENCH_DIR, parse_env_id  # noqa: E402
from orl.record.logger import RunGuard  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def to_t(x, dev):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


def build_gap(kind: str, base: dict, fraction: float, seed: int):
    if kind == "none":
        return None
    if kind == "sub":
        return subsample_episodes(kind, base, fraction, seed=seed)
    if kind == "ball":
        return state_gap(kind, base, drop_quantile=fraction, seed=seed,
                         mode="ball")
    if kind == "action":
        return action_gap(kind, base, keep_inner=fraction, seed=seed)
    raise ValueError(f"unknown gap kind {kind!r}")


def gap_characterisation(g, base: dict, dev: str) -> dict:
    """The two quantities that say WHICH type of gap this is."""
    if g is None:
        return {"empty_ball_ratio": 1.0, "action_support_ratio": 1.0,
                "n": int(base["obs"].shape[0]), "n_episodes": int(base["ends"].size),
                "mean_fragment_len": 0.0}
    centre = np.asarray(g.obs, dtype=np.float64).mean(0)
    null_idx = np.random.default_rng(10_000 + g.n).choice(
        base["obs"].shape[0], size=min(g.n, base["obs"].shape[0]), replace=False)
    empty_g = float(np.linalg.norm(
        np.asarray(g.obs, dtype=np.float64) - centre, axis=1).min())
    empty_n = float(np.linalg.norm(
        np.asarray(base["obs"][null_idx], dtype=np.float64) - centre, axis=1).min())
    c = base["act"].mean(0, keepdims=True)
    sup_base = float(np.quantile(np.linalg.norm(base["act"] - c, axis=1), 0.95))
    sup_g = float(np.quantile(np.linalg.norm(g.act - c, axis=1), 0.95))
    return {"empty_ball_ratio": empty_g / max(empty_n, 1e-9),
            "action_support_ratio": sup_g / max(sup_base, 1e-9),
            "n": int(g.n), "n_episodes": int(g.n_episodes),
            "mean_fragment_len": float(g.lengths().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--gap", default="none",
                    choices=("none", "sub", "ball", "action"))
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--gap-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--checkpoints", default="10000,20000,30000,40000")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.checkpoints, args.eval_episodes = 4000, "2000,4000", 2
    cps = [int(c) for c in args.checkpoints.split(",")]
    if cps[-1] != args.steps:
        raise SystemExit("the last checkpoint must equal --steps")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep_base = episode_lookup(d.env_id)
    base = {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": np.concatenate(([0], ep_base.ends[:-1] + 1)),
            "ends": ep_base.ends}
    g = build_gap(args.gap, base, args.fraction, args.gap_seed)
    obs = g.obs if g is not None else base["obs"]
    act = g.act if g is not None else base["act"]
    rew = g.rew if g is not None else base["rew"]
    nobs = g.next_obs if g is not None else base["next_obs"]
    term = g.terminals if g is not None else base["terminals"]
    # THE interface point: the fragment structure comes from the constructor.
    ep = g.episode_lookup() if g is not None else ep_base

    ch = gap_characterisation(g, base, dev)
    print(f"env={d.env_id}  gap={args.gap} fraction={args.fraction} "
          f"seed={args.gap_seed}")
    print(f"  n={ch['n']:,}  episodes={ch['n_episodes']}  "
          f"mean fragment {ch['mean_fragment_len']:.1f}")
    print(f"  empty-ball ratio {ch['empty_ball_ratio']:.2f}  "
          f"action support ratio {ch['action_support_ratio']:.2f}")

    O, A, R = to_t(obs, dev), to_t(act, dev), to_t(rew, dev)
    NO, TE = to_t(nobs, dev), to_t(term, dev)
    agent = SupportAwareIQL(
        qf=TwinQ(d.obs.shape[1], d.act.shape[1]),
        vf=ValueFunction(d.obs.shape[1]),
        policy=GaussianPolicy(d.obs.shape[1], d.act.shape[1]),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=3e-4),
        max_steps=args.steps, tau=0.7, beta=3.0, alpha=0.005, discount=0.99,
        episode_lookup=ep, coverage=None, fixed_n=1, n_min=1, n_max=1,
    )
    rng = np.random.default_rng(args.seed)
    snaps, done = [], 0
    for cp in cps:
        while done < cp:
            j = np.sort(rng.integers(0, len(O), 256))
            agent.update(O[j], A[j], NO[j], R[j], TE[j])
            done += 1
        snaps.append({"step": cp, "policy": copy.deepcopy(agent.policy)})
    rows = []
    for s in snaps:
        ev = evaluate(s["policy"], d.domain, d.normer,
                      n_episodes=args.eval_episodes)
        rows.append({"step": s["step"], "raw": ev["raw_mean"],
                     "score": ev["score_mean"], "std": ev["raw_std"]})
        print(f"  step {s['step']:>7,}  raw {ev['raw_mean']:8.1f}  "
              f"normalised {ev['score_mean']:6.2f}")

    domain, tier = parse_env_id(d.env_id)
    tag = args.tag or f"{args.gap}{args.fraction}"
    with RunGuard(run_id=f"gap_{d.env_id}_{tag}_s{args.seed}",
                  method="iql+fixed_n1", dataset=d.env_id, domain=domain,
                  tier=tier, seed=args.seed, stage="T1",
                  score_norm=rows[-1]["score"], score_raw=rows[-1]["raw"],
                  notes=f"T1 gap condition {tag}; gap_seed={args.gap_seed}"):
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        p = BENCH_DIR / f"gap_condition_{tag}_{d.domain}_s{args.seed}.json"
        p.write_text(json.dumps(
            {"env": d.env_id, "gap": args.gap, "fraction": args.fraction,
             "gap_seed": args.gap_seed, "characterisation": ch, "rows": rows,
             "elapsed_min": (time.perf_counter() - t0) / 60.0}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
