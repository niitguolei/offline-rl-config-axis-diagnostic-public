"""T1: controlled datasets with TYPED gaps (state-region vs action-support).

Path A's headline claim is a map of where OPE-based selection fails, and the
map's axis is the *type* of gap. GATE 4 asks whether the two types produce
statistically separable failures, so the constructions here are the instrument
that decides it -- if the "state gap" also narrows the action support, the
typing collapses and there is nothing to separate.

That risk is measured, not assumed: `experiments/diag_gap_types.py` reports both
quantities for both constructions and the cross terms. Verification-first
applies to the instrument as much as to a mechanism.

Reduced-size construction follows Cheng et al. (arXiv:2306.04220), who evaluate
on 1% / 1/50 / 1/100 of D4RL, so the small-data regime is an accepted protocol
rather than something invented here.

THE TEMPORAL RULE
-----------------
Dropping interior transitions silently breaks every n-step window that crosses
the hole: the window would step from a retained transition onto a dropped one
and bootstrap from an unrelated successor. So a drop always CUTS the trajectory:
a kept index becomes an episode end when the original episode ended there, or
when the next index was dropped. `rebuild_boundaries` is the single place that
arithmetic lives and it has direct tests.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Gapped:
    """A controlled dataset derived from a base one.

    Carries the base normaliser / reward scale by reference so every derived
    dataset stays comparable: changing the reward scale or the observations'
    normalisation between conditions would confound the gap with a rescaling.
    """

    name: str
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    kept_fraction: float
    n_base: int
    description: str
    keep: np.ndarray

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(self.ends.shape[0])

    def lengths(self) -> np.ndarray:
        return self.ends - self.starts + 1

    def episode_lookup(self):
        """Lookup built from `starts`/`ends`, NOT from the terminal flags.

        Fragment ends introduced by a drop carry no flag -- the original
        transition was not an episode end, it just had a dropped successor. So
        `harness.episode_lookup()` (which derives boundaries from
        `terminals | timeouts`) would be WRONG here and would let n-step windows
        step across a cut. Use this method.
        """
        from ..data.episode import EpisodeIndexLookup

        return EpisodeIndexLookup(self.starts, self.ends, self.n)


def rebuild_boundaries(keep: np.ndarray, base_ends: np.ndarray) -> tuple:
    """Episode boundaries of the retained subsequence, with drops cutting it.

    A kept index ends an episode if the ORIGINAL episode ended there, if it is
    the last kept index, or if the next original index was dropped. The last
    condition is the one that is easy to forget and silently corrupts n-step
    targets.
    """
    keep = np.asarray(keep, dtype=bool)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("nothing retained")
    base_end_set = set(int(e) for e in base_ends)
    ends = []
    for k, pos in enumerate(idx):
        last = k == idx.size - 1
        nxt_dropped = (not last) and (idx[k + 1] != pos + 1)
        if last or nxt_dropped or int(pos) in base_end_set:
            ends.append(k)                     # position within the NEW arrays
    ends = np.asarray(ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1] + 1))
    return starts, ends


def _pack(name, base, keep, description) -> Gapped:
    keep = np.asarray(keep, dtype=bool)
    starts, ends = rebuild_boundaries(keep, base["ends"])
    j = np.flatnonzero(keep)
    return Gapped(
        name=name,
        obs=base["obs"][j], act=base["act"][j], rew=base["rew"][j],
        next_obs=base["next_obs"][j], terminals=base["terminals"][j],
        timeouts=base["timeouts"][j], starts=starts, ends=ends,
        kept_fraction=float(keep.mean()), n_base=int(keep.size),
        description=description, keep=keep)


def subsample_episodes(name, base, fraction: float, seed: int = 0) -> Gapped:
    """Keep whole episodes -- the cleanest reduced-size construction."""
    rng = np.random.default_rng(seed)
    n_ep = len(base["ends"])
    want = max(1, int(round(fraction * n_ep)))
    chosen = set(rng.choice(n_ep, size=want, replace=False).tolist())
    keep = np.zeros(base["obs"].shape[0], dtype=bool)
    for k, (s, e) in enumerate(zip(base["starts"], base["ends"])):
        if k in chosen:
            keep[s:e + 1] = True
    return _pack(name, base, keep,
                 f"keep {want}/{n_ep} whole episodes ({fraction:.0%})")


def full_dataset(name: str, base: dict) -> Gapped:
    """Wrap the whole base dataset as a `Gapped`, so callers that vary the
    subsample fraction can keep one code path for every fraction."""
    keep = np.ones(base["obs"].shape[0], dtype=bool)
    return _pack(name, base, keep, "full dataset")


def _pca1(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Leading principal direction, so 'a region' is data-driven not hand-picked."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0, keepdims=True)
    z = x - mu
    rng = np.random.default_rng(seed)
    if z.shape[1] > 64:
        z = z[:, rng.choice(z.shape[1], 64, replace=False)]
    # power iteration on the covariance
    v = rng.normal(size=z.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(40):
        v = z.T @ (z @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            break
        v /= nrm
    return v


def state_gap(name, base, drop_quantile: float = 0.3, seed: int = 0,
              direction: np.ndarray | None = None, mode: str = "ball",
              centre: np.ndarray | None = None) -> Gapped:
    """Remove a REGION of state space along a leading axis.

    Three modes, and two of them were measured to be inadequate:

      mode="ball" (default) removes a BALL around a centre state, i.e. a hole in
        every direction at once. This is the only mode that creates a
        measurable hole, for a quantitative reason: after per-dimension
        standardisation the observation cloud is near-isotropic, so with 141k
        retained points in 17 dimensions the typical nearest-neighbour distance
        is about 0.5, while a 30%-quantile SLAB is only about 0.15-0.3 thick.
        A slab thinner than the nearest-neighbour spacing is not a gap at all.
        Measured hole ratios against a size-matched random subset (q99):
        ball ~O(few), band 1.18, tail 1.14, action gap 1.17.
      mode="band" removes an interior slab along the leading direction. Kept for
        the record; it underperforms for the reason above.
      mode="tail" removes the upper tail, which is sparse to begin with.

    Dropped transitions cut their trajectories, so the result is a set of
    fragments.
    """
    x = np.asarray(base["obs"], dtype=np.float64)
    if mode == "ball":
        c = (np.asarray(centre, dtype=np.float64) if centre is not None
             else x.mean(0))
        rad = np.linalg.norm(x - c, axis=1)
        thr = np.quantile(rad, drop_quantile)
        keep = rad > thr
        desc = (f"drop the central {drop_quantile:.0%} ball (r <= {thr:.2f}) "
                f"around the state mean; trajectory fragments")
    elif mode == "band":
        d = direction if direction is not None else _pca1(x, seed)
        proj = x @ np.asarray(d, dtype=np.float64)
        lo = np.quantile(proj, 0.5 - drop_quantile / 2.0)
        hi = np.quantile(proj, 0.5 + drop_quantile / 2.0)
        keep = (proj < lo) | (proj > hi)
        desc = (f"drop the interior {drop_quantile:.0%} band of PC1 "
                f"[{lo:.2f}, {hi:.2f}]; trajectory fragments")
    elif mode == "tail":
        d = direction if direction is not None else _pca1(x, seed)
        proj = x @ np.asarray(d, dtype=np.float64)
        thr = np.quantile(proj, 1.0 - drop_quantile)
        keep = proj <= thr
        desc = (f"drop the top {drop_quantile:.0%} tail of PC1 "
                f"(> {thr:.2f}); trajectory fragments")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if keep.sum() < 16:
        raise ValueError("state gap removed too much")
    return _pack(name, base, keep, desc)


def action_gap(name, base, keep_inner: float = 0.3, seed: int = 0) -> Gapped:
    """Narrow the ACTION support to a band around the mean action.

    Keeps only transitions whose action lies in the central `keep_inner`
    fraction by distance from the mean action, so the action marginal loses its
    tails while the state marginal is only mildly perturbed. The cross-check in
    `diag_gap_types.py` measures whether the state coverage survived, because
    that is the assumption this construction rests on.
    """
    a = np.asarray(base["act"], dtype=np.float64)
    centre = a.mean(0, keepdims=True)
    dist = np.linalg.norm(a - centre, axis=1)
    thr = np.quantile(dist, keep_inner)
    keep = dist <= thr
    if keep.sum() < 16:
        raise ValueError("action gap removed too much")
    return _pack(name, base, keep,
                 f"keep the central {keep_inner:.0%} of actions by ||a-a-bar|| "
                 f"(<= {thr:.3f}); trajectory fragments")
