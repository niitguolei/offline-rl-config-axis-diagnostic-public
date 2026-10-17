"""Selection-quality reporting, including the baselines a reviewer will ask for.

Added by review S1/T3/U1. Asking "what does a free rule achieve?" produced the
most damaging finding in the whole review, so the answer can no longer live in a
report paragraph -- it is computed here and every table must carry it.

Three baselines:

  last     pick the final checkpoint. Free, no estimator, and it ties or beats
           every estimator in three of the four measured conditions. If the true
           score rises monotonically with training, this rule is near-optimal and
           the whole selection problem is trivial.
  random   the expected oracle gap when one checkpoint is picked uniformly.
           An estimator that cannot beat this is not selecting anything.
  naive_step   rank checkpoints by training step, i.e. the monotone-progress
           proxy. This is what `rho` is really measuring when the truth happens
           to be monotone, which is why `spearman` is reported for it too.

Non-triviality: a condition only exercises selection if `last`'s gap reaches
`gap_threshold` OR the truth has a SIGNIFICANT peak before the end. The default
threshold is 5%, tied to the measured evaluation noise: the 30-episode SEM is
0.34-1.7% of the score, so 5% is more than 3x the noise and therefore
distinguishable from it.

The word SIGNIFICANT is load-bearing, and a test caught why. The first version
of this criterion treated ANY dip as non-trivial. The base condition's truth is
33.09 / 41.75 / 42.69 / 42.32 -- it dips at the end by 0.87%, which is inside the
evaluation noise, and the criterion would have declared the condition
non-trivial on that basis. A criterion that noise can trigger is worse than no
criterion. So the monotonicity leg requires the dip to exceed 2x the SEM, and
when per-episode returns are unavailable (so no SEM) the leg is recorded as
`cannot_evaluate` and only the gap leg applies -- conservatively.

A condition that fails both legs is TRIVIAL, and its estimator rank correlations
are not evidence about estimator quality.

Both legs now need a PRACTICAL size as well as a statistical one, and E0c's probe
10 showed why. halfcheetah_medium_replay trained to 200k on the full dataset has
a flat truth (5020-5112, SEM ~15) and a dip of 0.37 -- which exceeds 2x SEM and
so fired the monotonicity leg, while being 0.007% of the score. A criterion that
calls a 0.007% wiggle a degradation is measuring nothing. Each leg therefore
requires both `> 2x SEM` (statistical) and `>= min_rel_effect` (practical).
"""

from __future__ import annotations

import numpy as np

from .ope import spearman

DEFAULT_GAP_THRESHOLD = 0.05


def oracle_gap(true_scores, picked: int) -> float:
    """1 - picked/oracle. Uses the raw score as the denominator: normalised
    scores can sit near zero (measured 1.20 in the action condition), and the
    first version of this metric divided by that."""
    t = np.asarray(true_scores, dtype=np.float64)
    o = int(np.argmax(t))
    return float(1.0 - t[picked] / max(abs(t[o]), 1e-9))


def trivial_baselines(true_scores, steps) -> dict:
    """`last`, `random` and the monotone-progress proxy, with their null levels."""
    t = np.asarray(true_scores, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    n = len(t)
    last_gap = oracle_gap(t, n - 1)
    expected_random_gap = float(np.mean([oracle_gap(t, i) for i in range(n)]))
    return {
        "last_gap": last_gap,
        "naive_step_spearman": spearman(s, t),
        "random_expected_gap": expected_random_gap,
        # with four checkpoints the best random pick beats this half the time;
        # the exact number matters less than being on the same row as the
        # estimators, so a reader cannot miss it
        "random_best_of_all_gap": 0.0,
    }


def nontriviality(true_scores, steps, sem: float | None = None,
                  gap_threshold: float = DEFAULT_GAP_THRESHOLD,
                  min_rel_effect: float = 0.02) -> dict:
    """Is there anything for selection to do here?

    `sem` is the SEM of the true score (from per-episode returns). Without it the
    monotonicity leg cannot be evaluated -- a dip of any size would otherwise
    count, and noise alone produces those.

    `min_rel_effect` is the practical floor for both legs, as a fraction of the
    best score: 2%, i.e. 40x the 30-episode SEM. It exists because a statistically
    real but pract. negligible dip must not turn a flat curve into a selection
    problem (probe 10: 0.007%).
    """
    t = np.asarray(true_scores, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    order = np.argsort(s)
    t_sorted = t[order]
    scale = max(float(np.max(np.abs(t_sorted))), 1e-9)
    if sem is None:
        significant_peak = None
        peak_size = float(np.max(t_sorted[:-1] - t_sorted[1:])) \
            if len(t_sorted) > 1 else 0.0
        mono_leg = "cannot_evaluate_without_per_episode_returns"
    else:
        peak_size = float(np.max(t_sorted[:-1] - t_sorted[1:])) \
            if len(t_sorted) > 1 else 0.0
        statistically_real = peak_size > 2.0 * float(sem)
        practically_real = (peak_size / scale) >= min_rel_effect
        significant_peak = bool(statistically_real and practically_real)
        mono_leg = ("significant_peak" if significant_peak else
                    "statistically_real_but_negligible"
                    if statistically_real else "no_significant_peak")
    base = trivial_baselines(t, s)
    gap_leg = bool(base["last_gap"] >= gap_threshold
                   and base["last_gap"] >= min_rel_effect)
    trivial = not (gap_leg or bool(significant_peak))
    return {
        "last_gap": base["last_gap"],
        "random_expected_gap": base["random_expected_gap"],
        "naive_step_spearman": base["naive_step_spearman"],
        "gap_leg": gap_leg,
        "peak_size": peak_size,
        "peak_relative": peak_size / scale,
        "min_rel_effect": float(min_rel_effect),
        "monotonicity_leg": mono_leg,
        "gap_threshold": float(gap_threshold),
        "trivial": trivial,
        "verdict": ("TRIVIAL: a free rule is within the noise"
                    if trivial else
                    "NON-TRIVIAL: selection has something to do"),
    }


def summarise(rows: list[dict], estimator_keys: dict, gap_threshold: float =
              DEFAULT_GAP_THRESHOLD) -> dict:
    """Per-condition summary with the baselines on every row.

    `rows` are the failure-map cells; each has `condition`, `step`, `true_score`
    and one key per estimator. `estimator_keys` maps a display name to its key.
    """
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        sel = sorted([r for r in rows if r["condition"] == cond],
                     key=lambda r: r["step"])
        if len(sel) < 2:
            continue
        truth = np.array([float(r["true_score"]) for r in sel])
        steps = np.array([float(r["step"]) for r in sel])
        # `sem=None` on purpose: map rows carry no per-episode returns, so the
        # monotonicity leg is not evaluable and only the gap leg applies.
        # NOTE the keyword: the first version passed gap_threshold positionally,
        # which landed in the `sem` slot -- so 0.05 acted as an SEM and every dip
        # larger than 0.1 was declared a significant peak. Three of four
        # conditions were mislabelled non-trivial. A test now pins the leg.
        nt = nontriviality(truth, steps, sem=None, gap_threshold=gap_threshold)
        entry = {"n_checkpoints": len(sel),
                 "true_scores": truth.tolist(),
                 "steps": steps.tolist(),
                 "random_expected_gap": nt["random_expected_gap"],
                 "triviality": nt}
        for name, key in estimator_keys.items():
            est = np.array([float(r[key]) for r in sel])
            entry[name] = {
                "spearman": spearman(est, truth),
                "oracle_gap": oracle_gap(truth, int(np.argmax(est))),
            }
            if "mc_det_value" in sel[0]:
                anch = np.array([float(r["mc_det_value"]) for r in sel])
                entry[name]["rel_bias_vs_mc_det"] = float(
                    np.mean(np.abs(est - anch) / np.maximum(np.abs(anch), 1e-9)))
        out[cond] = entry
    return out


def format_table(summary: dict) -> str:
    """One line per condition, baselines first, TRIVIAL marked."""
    lines = []
    head = (f"{'condition':<10}{'trivial':>9}{'last gap':>10}"
            f"{'random gap':>12}  estimators rho (gap)")
    lines.append(head)
    for cond, e in summary.items():
        tr = e["triviality"]
        est = "  ".join(
            f"{k} {v['spearman']:+.2f}({v['oracle_gap']:+.1%})"
            for k, v in e.items() if isinstance(v, dict) and "spearman" in v)
        lines.append(f"{cond:<10}{str(tr['trivial']):>9}{tr['last_gap']:>10.1%}"
                     f"{e.get('random_expected_gap', float('nan')):>12.1%}  {est}")
    return "\n".join(lines)
