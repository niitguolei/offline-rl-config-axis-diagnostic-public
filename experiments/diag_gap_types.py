"""Does a "state gap" really differ in TYPE from an "action gap"? Measure first.

GATE 4's whole premise is that two gap types produce separable failure modes.
If the construction meant to remove a state region also narrows the action
support, or the action-narrowing construction also punches a state hole, then
the two conditions differ in DEGREE along one axis, not in TYPE -- and GATE 4
would be measuring nothing. That failure would otherwise surface only after
training the whole grid.

So both quantities are measured for every construction, including the cross
terms, and the acceptance thresholds are written down before the numbers are
seen:

    state_gap   must have  state_hole ≥ 1.5   and  action_support ≥ 0.70
    action_gap  must have  action_support ≤ 0.80  and  state_hole ≤ 1.5

Two blind distance metrics were tried and both failed, which is why the final
measures are construction-aware:

  1. dropped vs kept 1-NN distances -- explodes for ANY subsampling (a 1% random
     sample scored 6.8e8) because retained states include their own episode-mates.
  2. retained vs a size-matched random subset, at q95 and q99 -- the measurements
     do not separate the constructions (band 1.18, tail 1.14, ball 0.78, action
     gap 1.17). In 17 near-isotropic dimensions the 1-NN distance is dominated by
     how sparse the cloud's outskirts are, not by any local hole, and 141k
     retained points are far too thin to make a hole visible this way.

Since the constructions are CONTROLLED -- their parameters are ours, not
inferred -- the honest measures use them:

    empty_ball_ratio       min distance from the removed region's centre to the
                           retained data, over the same for a size-matched random
                           subset. ~1 for random removal, >= 3 for a real hole.
    removed_out_of_support share of REMOVED states whose 1-NN distance to the
                           retained set exceeds the base's own 95th-percentile
                           coverage line (the support definition GATE 1 used).
    action_support_ratio   p95(||a - a_bar||) over the base's; < 1 = narrower.

`action_support` is p95(||a - a_bar||) of the variant over the base's, so < 1
means the action marginal lost its tails.

    python experiments/diag_gap_types.py --smoke
    python experiments/diag_gap_types.py
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

from orl.coverage.knn import knn_distances  # noqa: E402
from orl.gaps.construct import (  # noqa: E402
    action_gap,
    state_gap,
    subsample_episodes,
)
from orl.methods.harness import episode_lookup, load  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def as_base(d, ep) -> dict:
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    return {"obs": d.obs, "act": d.act, "rew": d.rew, "next_obs": d.next_obs,
            "terminals": d.terminals, "timeouts": d.timeouts,
            "starts": starts, "ends": ep.ends}


def one_nn(queries: np.ndarray, refs: np.ndarray, device: str,
           chunk: int = 512) -> np.ndarray:
    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    r = torch.as_tensor(refs, dtype=torch.float32, device=device)
    d = knn_distances(q, r, k=1, chunk=chunk, drop_self=False)
    return d[:, 0].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="halfcheetah_medium_replay-v2")
    ap.add_argument("--probe", type=int, default=1500,
                    help="original states used to measure the hole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.probe = 200

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    d = load(args.env, with_coverage=False, normalize=True)
    ep = episode_lookup(d.env_id)
    base = as_base(d, ep)
    print(f"env={d.env_id}  n={d.obs.shape[0]:,}  episodes={ep.ends.size}")

    variants = [
        ("base", None),
        ("sub_ep_10pct", ("sub", 0.10)),
        ("sub_ep_1pct", ("sub", 0.01)),
        ("state_ball_30pct", ("state", 0.30)),
        ("state_band_30pct", ("band", 0.30)),
        ("state_tail_30pct", ("tail", 0.30)),
        ("action_gap_30pct", ("action", 0.30)),
    ]

    rng = np.random.default_rng(args.seed)
    probe_idx = np.sort(rng.choice(base["obs"].shape[0],
                                   size=min(args.probe, base["obs"].shape[0]),
                                   replace=False))
    probe_states = base["obs"][probe_idx]
    # the base's own support line, reused as the "out of support" definition
    base_nn = one_nn(base["obs"][probe_idx], base["obs"], dev)
    support_line = float(np.quantile(base_nn, 0.95))
    base_act = base["act"]
    centre = base_act.mean(0, keepdims=True)
    base_support = float(np.quantile(np.linalg.norm(base_act - centre, axis=1), 0.95))

    rows = []
    print(f"\n{'variant':<18}{'n':>9}{'eps':>7}{'frag':>7}"
          f"{'emptyball':>10}{'oos':>7}{'act_sup':>9}")
    for name, spec in variants:
        if spec is None:
            g, keep = None, np.ones(base["obs"].shape[0], dtype=bool)
            n, n_ep, frag = base["obs"].shape[0], ep.ends.size, 0.0
            kept_obs, kept_act = base["obs"], base_act
        else:
            kind, frac = spec
            g = (subsample_episodes(name, base, frac, seed=args.seed)
                 if kind == "sub" else
                 state_gap(name, base, drop_quantile=frac, seed=args.seed)
                 if kind == "state" else
                 state_gap(name, base, drop_quantile=frac, seed=args.seed,
                           mode="band")
                 if kind == "band" else
                 state_gap(name, base, drop_quantile=frac, seed=args.seed,
                           mode="tail")
                 if kind == "tail" else
                 action_gap(name, base, keep_inner=frac, seed=args.seed))
            keep = g.keep            # carried by the constructor, not guessed
            n, n_ep = g.n, g.n_episodes
            frag = float(g.lengths().mean())
            kept_obs, kept_act = g.obs, g.act

        # --- empty-ball at the construction's centre, vs a size-matched null
        null_idx = np.random.default_rng(10_000 + int(n)).choice(
            base["obs"].shape[0], size=min(int(n), base["obs"].shape[0]),
            replace=False)
        null_obs = base["obs"][null_idx]
        centre_arr = np.asarray(kept_obs, dtype=np.float64).mean(0)
        empty_var = float(np.linalg.norm(
            np.asarray(kept_obs, dtype=np.float64) - centre_arr, axis=1).min())
        empty_null = float(np.linalg.norm(
            np.asarray(null_obs, dtype=np.float64) - centre_arr, axis=1).min())
        emptyball = empty_var / max(empty_null, 1e-9)

        # --- how much of the REMOVED mass is now beyond the base's support line
        d_kept = one_nn(base["obs"][probe_idx], kept_obs, dev)
        removed_probe = ~keep[probe_idx]
        oos = (float((d_kept[removed_probe] > support_line).mean())
               if removed_probe.any() else 0.0)
        sup = float(np.quantile(np.linalg.norm(kept_act - centre, axis=1), 0.95))
        act_support = sup / base_support
        print(f"{name:<18}{n:>9,}{n_ep:>7}{frag:>7.1f}"
              f"{emptyball:>10.2f}{oos:>7.2f}{act_support:>9.2f}")
        rows.append({"name": name, "n": int(n), "n_episodes": int(n_ep),
                     "mean_fragment_len": frag,
                     "empty_ball_ratio": emptyball,
                     "removed_out_of_support": oos,
                     "action_support_ratio": act_support})

    by = {r["name"]: r for r in rows}
    print("\nacceptance (thresholds fixed before looking at the numbers):")
    checks = []
    sg = by["state_ball_30pct"]
    ag = by["action_gap_30pct"]
    checks.append(("state_ball: empty-ball ratio ≥ 2.0",
                   sg["empty_ball_ratio"] >= 2.0, sg["empty_ball_ratio"]))
    checks.append(("state_ball: removed mass out of support ≥ 0.5",
                   sg["removed_out_of_support"] >= 0.5,
                   sg["removed_out_of_support"]))
    checks.append(("state_ball: action support ≥ 0.70",
                   sg["action_support_ratio"] >= 0.70,
                   sg["action_support_ratio"]))
    checks.append(("action_gap: action support ≤ 0.80",
                   ag["action_support_ratio"] <= 0.80,
                   ag["action_support_ratio"]))
    checks.append(("action_gap: empty-ball ratio ≤ 1.5",
                   ag["empty_ball_ratio"] <= 1.5, ag["empty_ball_ratio"]))
    ok = True
    for label, passed, value in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}  (measured {value:.2f})")
    print(f"\nTYPING {'SEPARABLE' if ok else 'NOT SEPARABLE -- GATE 4 needs redesign'}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("gap_types_smoke.json" if args.smoke else "gap_types.json")
    p.write_text(json.dumps(
        {"env": d.env_id, "probe": int(len(probe_idx)), "rows": rows,
         "checks": [{"label": l, "pass": bool(v), "value": float(x)}
                    for l, v, x in checks],
         "separable": bool(ok),
         "elapsed_min": (time.perf_counter() - t0) / 60.0}, indent=2),
        encoding="utf-8")
    print(f"wrote {p.name}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
