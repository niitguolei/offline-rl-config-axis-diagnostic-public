"""M1's premise, turned into a number that can be computed BEFORE any training.

The mechanism
-------------
M1 ties the critic onto the environment's mirror symmetry. That is useful
exactly when the data does NOT already cover both sides of the orbit: if the
mirror partner of a state is missing from the data, the constraint is doing
real extrapolation; if the partner is nearby, the constraint is nearly
redundant. So the premise of M1 is a statement about ORBIT COVERAGE, and it can
be measured without training anything.

    d_self   = distance from s to its 10th nearest OTHER point in the data
    d_mirror = distance from g(s) to its 10th nearest point in the data

Both use the SAME rank ("the 10th other datum"), which matters: for a dataset
state the nearest reference point is s itself, so its rank 1 is a self-match
and must be dropped, whereas g(s) is essentially never in the data and dropping
its nearest would remove a genuine neighbour. Taking column k-1 of the
drop_self=True result for s and of the drop_self=False result for g(s) makes
the two comparable, and makes d_self reproduce the cached coverage score.

Reported quantities
-------------------
  ratio        median d_mirror / median d_self        (>1 means extrapolation)
  outside      share of states whose mirror partner is beyond the cell's own
               95th-percentile support line (the same line `buckets.py` uses)
  moved        mirror dims actually permuted, for context

Two components, because distance alone does not predict M1's usefulness
-----------------------------------------------------------------------
The first version of this script claimed the identity-mirror domains
(halfcheetah, hopper: no permuted dims, only a sign flip on rooty) would show
ratio ~ 1 as a control, on the reasoning that a sign flip on a coordinate the
task ignores cannot move the mirror partner far. **Measured: hopper 1.59-2.42,
halfcheetah 1.19-1.56.** The control failed, and the failure is informative:
those domains' rooty drifts laterally, so flipping it lands on the far side of
an off-centre distribution. The data really is asymmetric there.

So distance measures only HALF the premise:

  A. orbit extrapolation  how far the mirror partner is from the data
                          -> does the constraint supply information at all?
  B. mirror relevance     how much the coordinates the mirror moves matter
                          for the task
                          -> does the supplied information matter?

M1 should help only when BOTH are non-trivial. Hopper has large A and near-zero
B (its rooty does not affect forward progress), so its constraint is close to
vacuous; walker2d has large A and large B (the permuted leg joints determine
the gait). `A x B` is therefore the predictor, and B is measured here as the
ridge-regression coefficient mass of the return-to-go on the mirror-touched
coordinates, relative to the untouched ones.

    python experiments/diag_orbit_coverage.py
    python experiments/diag_orbit_coverage.py --envs walker2d_medium_replay-v2
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

from orl.coverage.buckets import buckets_for  # noqa: E402
from orl.coverage.knn import DEFAULT_K, DEFAULT_N_REF, knn_distances  # noqa: E402
from orl.data.d4rl_hdf5 import D4RLDataset  # noqa: E402
from orl.methods.equivariance import derive_mirror  # noqa: E402
from orl.paths import BENCH_DIR, all_env_ids  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def orbit_coverage(env_id: str, queries: int, seed: int = 0,
                   device: str = "cuda") -> dict:
    if "random" in env_id:
        raise SystemExit("the main table excludes the random tier")
    with D4RLDataset(env_id, mode="ram") as ds:
        obs = ds.get("observations")
        domain, obs_dim, act_dim = ds.domain, ds.obs_dim, ds.act_dim
    spec = derive_mirror(domain, obs_dim=obs_dim, act_dim=act_dim)

    # Rebuild the reference subsample EXACTLY as coverage_scores did, so that
    # d_self here is comparable with the cached coverage score.
    t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    ref_idx = torch.randperm(t.shape[0], generator=gen)[:DEFAULT_N_REF]
    refs = t[ref_idx]

    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(t.shape[0], size=min(queries, t.shape[0]),
                              replace=False))
    s = t[pick]
    g_s = torch.as_tensor(spec.apply_obs(obs[pick]), dtype=torch.float32,
                          device=device)

    # rank k+1 overall: self-match included for s, absent for g(s). Column k-1
    # of each is therefore "the k-th other datum" on both sides.
    k = DEFAULT_K
    d_all_s = knn_distances(s, refs, k=k, drop_self=True)
    d_all_g = knn_distances(g_s, refs, k=k, drop_self=False)
    d_self = d_all_s[:, k - 1].cpu().numpy().astype(np.float64)
    d_mirr = d_all_g[:, k - 1].cpu().numpy().astype(np.float64)

    cached = buckets_for(env_id)
    support = cached.support_threshold
    cached_at_pick = cached.scores[pick]
    return {
        "env_id": env_id, "domain": domain, "obs_dim": obs_dim,
        "n_queries": int(len(pick)), "k": k, "n_ref": int(refs.shape[0]),
        "moved_dims": int((spec.obs_perm != np.arange(obs_dim)).sum()),
        "flipped_dims": int((spec.obs_sign == -1).sum()),
        "d_self_median": float(np.median(d_self)),
        "d_mirror_median": float(np.median(d_mirr)),
        "ratio_median": float(np.median(d_mirr) / np.median(d_self)),
        "ratio_per_query_median": float(np.median(d_mirr / d_self)),
        "outside_support": float((d_mirr > support).mean()),
        "mirror_farther": float((d_mirr > d_self).mean()),
        "support_threshold": float(support),
        # Validation of the rank convention: must reproduce the cache.
        "d_self_vs_cache_max_abs": float(np.abs(d_self - cached_at_pick).max()),
        "d_self_vs_cache_med_abs": float(np.median(np.abs(d_self - cached_at_pick))),
    }


def mirror_relevance(env_id: str, ridge: float = 1e-2, chunk: int = 200_000) -> dict:
    """Component B: do the mirror-touched coordinates matter for the task?

    Fit a ridge regression of the return-to-go on the (standardised)
    observation and compare the coefficient mass on coordinates the mirror
    touches against the untouched ones. A domain whose mirror only flips a
    coordinate that forward progress does not depend on gets a small ratio, and
    M1's constraint there can only be vacuous however far the mirror partner
    sits from the data.

    Deliberately a linear probe: it is a lower bound on relevance and it costs
    seconds per cell, which is what a pre-training predictor has to cost.
    """
    with D4RLDataset(env_id, mode="ram") as ds:
        obs = ds.get("observations")
        rew = ds.get("rewards")
        ep = ds.episodes
        spec = derive_mirror(ds.domain, obs_dim=ds.obs_dim, act_dim=ds.act_dim)

    # Return-to-go, computed within episode boundaries only.
    g = np.zeros(len(rew), dtype=np.float64)
    for s0, ln in zip(ep.starts(), ep.lengths()):
        seg = rew[s0:s0 + ln].astype(np.float64)
        g[s0:s0 + ln] = np.cumsum(seg[::-1])[::-1]

    x = np.asarray(obs, dtype=np.float64)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    y = g - g.mean()
    ysd = y.std() if y.std() > 0 else 1.0
    y = y / ysd

    # Normal equations, accumulated in chunks so the 2M-row cells stay cheap.
    d = x.shape[1]
    a = np.zeros((d, d), dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    for lo in range(0, x.shape[0], chunk):
        z = (x[lo:lo + chunk] - mu) / sd
        yy = y[lo:lo + chunk]
        a += z.T @ z
        b += z.T @ yy
    a = a / x.shape[0] + ridge * np.eye(d)
    beta = np.linalg.solve(a, b / x.shape[0])

    perm, sign = spec.obs_perm, spec.obs_sign
    touched = (perm != np.arange(d)) | (sign == -1)

    # --- B as defined here, after the first attempt failed.
    #
    # The first version averaged |beta| over touched dims and divided by the
    # same over untouched dims. It is not comparable across domains: walker2d
    # leaves only 3 dims untouched and one of them (rootx velocity) dominates
    # the return-to-go, so its ratio came out BELOW 1 for every walker2d cell
    # while the highest value landed on halfcheetah, whose mirror is nearly
    # vacuous. A ratio of means over tiny, arbitrarily-sized groups is the
    # wrong statistic.
    #
    # Fix: use beta as a value surrogate and ask how far the mirror MOVES the
    # predicted value, relative to the value's spread across the data.
    #     delta  = mean |beta.(g(s) - s)|      in standardised coordinates
    #     spread = std  (beta.s)
    # By the environment's symmetry the TRUE value is unchanged by g, so this
    # measures how much the mirror actually displaces a task-relevant
    # direction: near zero when the mirror only touches coordinates the task
    # ignores (hopper's rooty), sizeable when it permutes coordinates that
    # determine the gait (walker2d's left/right leg joints).
    rng = np.random.default_rng(0)
    pick = rng.choice(x.shape[0], size=min(4000, x.shape[0]), replace=False)
    xs = (x[pick] - mu) / sd
    gs = (spec.apply_obs(x[pick]) - mu) / sd
    v_self = xs @ beta
    v_mirr = gs @ beta
    delta = float(np.abs(v_mirr - v_self).mean())
    spread = float(v_self.std())
    return {
        "script": "linear ridge probe on return-to-go",
        "ridge": ridge,
        "n_touched": int(touched.sum()),
        "n_untouched": int((~touched).sum()),
        "B_touched": float(np.abs(beta[touched]).mean()),
        "B_untouched": float(np.abs(beta[~touched]).mean()),
        "B_ratio": float(np.abs(beta[touched]).mean()
                          / np.abs(beta[~touched]).mean()),
        "B2_delta": delta,
        "B2_spread": spread,
        "B2": delta / spread if spread > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None,
                    help="comma separated; default is the 12 main-table cells")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.envs = "walker2d_medium_replay-v2"
        args.queries = 256

    envs = args.envs.split(",") if args.envs else all_env_ids()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"orbit coverage: {len(envs)} cells, {args.queries} queries each, "
          f"n_ref={DEFAULT_N_REF}, k={DEFAULT_K}, device={device}")
    print("'outside' = share of states whose mirror partner sits beyond the "
          "cell's own 95th-percentile support line\n")
    hdr = (f"{'env':<34}{'moved':>6}{'flip':>5}{'d_self':>9}{'d_mirror':>10}"
           f"{'A_ratio':>8}{'outside':>9}{'B2':>9}{'AxB':>8}"
           f"{'cache_err':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    t0 = time.perf_counter()
    for env_id in envs:
        r = orbit_coverage(env_id, args.queries, args.seed, device)
        r.update(mirror_relevance(env_id))
        # Both components must be non-trivial for M1 to have anything to do.
        r["AxB"] = float(r["ratio_median"] * r["B2"])
        rows.append(r)
        print(f"{env_id:<34}{r['moved_dims']:>6}{r['flipped_dims']:>5}"
              f"{r['d_self_median']:>9.3f}{r['d_mirror_median']:>10.3f}"
              f"{r['ratio_median']:>8.3f}{r['outside_support']:>9.3f}"
              f"{r['B2']:>9.3f}{r['AxB']:>8.3f}"
              f"{r['d_self_vs_cache_max_abs']:>11.2e}", flush=True)
    print(f"\n{time.perf_counter() - t0:.0f} s total")

    print("\nA (orbit extrapolation) by cell:")
    for r in sorted(rows, key=lambda z: -z["ratio_median"]):
        print(f"  {r['env_id']:<34} A={r['ratio_median']:6.3f}  "
              f"outside={r['outside_support']:.3f}  touched={r['n_touched']}/"
              f"{r['n_touched'] + r['n_untouched']}")

    print("\nB2 (how far the mirror MOVES the predicted value, relative to "
          "the value spread) by cell:")
    for r in sorted(rows, key=lambda z: -z["B2"]):
        print(f"  {r['env_id']:<34} B2={r['B2']:6.3f}  "
              f"delta={r['B2_delta']:.4f}  spread={r['B2_spread']:.4f}  "
              f"(old per-dim ratio {r['B_ratio']:.3f}, kept for the record)")

    print("\nPREDICTOR A x B (M1 has room to help only where this is large):")
    for r in sorted(rows, key=lambda z: -z["AxB"]):
        print(f"  {r['env_id']:<34} AxB={r['AxB']:7.3f}")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("orbit_coverage_smoke.json" if args.smoke
                     else "orbit_coverage.json")
    p.write_text(json.dumps({"queries": args.queries, "seed": args.seed,
                             "k": DEFAULT_K, "n_ref": DEFAULT_N_REF,
                             "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
