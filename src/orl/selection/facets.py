"""Selection quality measured along ONE configuration axis at a time.

E2b's cross-configuration arm ranked six configs at once: three learning rates
crossed with two conservatism targets (tau). That is not one selection problem,
it is two problems with different amounts of built-in leakage, and the 5-seed
result says so. The critic mean is a near-monotone function of tau
(corr +0.54 / +0.99 / +0.97 on halfcheetah / hopper / walker2d) but almost
independent of lr (+0.15 / +0.08 / +0.20). Ranking along a polluted axis and
ranking along a clean axis are therefore different questions.

The headline claim of the paper makes a directional prediction about them:

    a selection signal is trustworthy to the extent that it is NOT a function
    of the axis being compared, and that leak can be measured before any truth
    is known.

This module makes the prediction falsifiable on data already collected: it
re-ranks the SAME runs along each axis separately, holding the other fixed, and
reports the oracle gap next to the uniform-pick null. No new training. If the
critic mean is useless along tau and useful along lr, the claim predicted that
before the number was computed -- which is what separates a diagnosis from a
post-hoc story.
"""

from __future__ import annotations

import numpy as np

from .report import oracle_gap

ESTIMATORS = ("critic_mean", "fqe", "pdis")


def slice_index(runs: list[dict], axis: str, within: list[str]) -> dict:
    """Group runs into selection problems: one per (env, seed, *within) slice."""
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r["env"], r["seed"], *[r[w] for w in within])
        buckets.setdefault(key, []).append(r)
    return buckets


def facet_gaps(
    runs: list[dict],
    axis: str,
    within: list[str],
    est_keys: tuple[str, ...] = ESTIMATORS,
    truth_key: str = "true_score",
) -> dict:
    """Rank each slice along `axis`; report gaps per env and per estimator.

    A slice with fewer than three distinct `axis` values is skipped: with two
    candidates the null is 0.5 and nothing is being selected.
    """
    by_env: dict[str, dict] = {}
    for (env, seed, *wvals), sel in sorted(
            slice_index(runs, axis, within).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len({r[axis] for r in sel}) < 3:
            continue
        sel = sorted(sel, key=lambda r: r[axis])
        truth = np.array([r[truth_key] for r in sel], dtype=np.float64)
        # The slice key must carry the held-fixed values: two taus with the same
        # seed are two different selection problems, and keying by seed alone
        # silently drops one of them (caught by test_facets.py).
        slice_key = "|".join([f"seed={seed}"] + [f"{w}={v}" for w, v in zip(within, wvals)])
        ent = {
            "n_configs": len(sel),
            "true_scores": truth.tolist(),
            "axis_values": [r[axis] for r in sel],
            # Uniform pick over the candidates: same definition as e2b.
            "random_expected_gap": float(
                np.mean([oracle_gap(truth, i) for i in range(len(truth))])),
            "oracle_index": int(np.argmax(truth)),
        }
        for key in est_keys:
            if any(r.get(key) is None for r in sel):
                continue
            est = np.array([float(r[key]) for r in sel], dtype=np.float64)
            picked = int(np.argmax(est))
            ent[key] = {"picked": picked, "gap": oracle_gap(truth, picked)}
        by_env.setdefault(env, {"per_slice": {}})["per_slice"][slice_key] = ent

    for env, env_ent in by_env.items():
        per_slice = env_ent["per_slice"]
        summary: dict[str, dict] = {}
        for key in est_keys:
            if any(key not in v for v in per_slice.values()):
                continue
            gaps = np.array([v[key]["gap"] for v in per_slice.values()])
            rnd = np.array([v["random_expected_gap"] for v in per_slice.values()])
            sem = (float(gaps.std(ddof=1) / np.sqrt(len(gaps)))
                   if len(gaps) > 1 else float("nan"))
            summary[key] = {
                "mean_gap": float(gaps.mean()),
                "mean_random_gap": float(rnd.mean()),
                "beats_random_by": float(rnd.mean() - gaps.mean()),
                "seed_gap_sem": sem,
                "useful": bool(gaps.mean() < rnd.mean() - (sem if sem == sem else 0.0)),
            }
        env_ent["summary"] = summary
        env_ent["axis"] = axis
        env_ent["within"] = list(within)
    return by_env


def leak_correlation(runs: list[dict], axis: str, key: str) -> float:
    """corr(estimator, axis value) over all runs -- the zero-cost leak probe."""
    est = np.array([float(r[key]) for r in runs], dtype=np.float64)
    ax = np.array([float(r[axis]) for r in runs], dtype=np.float64)
    if est.std() == 0 or ax.std() == 0:
        return float("nan")
    return float(np.corrcoef(est, ax)[0, 1])
