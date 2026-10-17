"""T1a — coverage bucketing and the support filter for GATE 1.

Both thresholds come from the DATASET's own coverage-score distribution, and
they are deliberately independent of each other (a round-5 audit finding):

  * buckets   : 33 / 67 percentiles -> low / mid / high coverage
  * filter    : 95th percentile      -> "in-support" for MC start states

The two measure DIFFERENT things and must not share a threshold:
  * the bucket of a transition says "how isolated this datum is";
  * the support filter says "how far a policy-visited state is from the data".
Keeping them separate avoids the circularity flagged in the round-5 audit,
where one threshold was doing both jobs and could filter away exactly the
low-coverage states of interest.

Thresholds are computed once, written to the config, and never retuned
per-dataset mid-experiment.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..coverage.knn import DEFAULT_K, DEFAULT_N_REF, cache_path

BUCKET_EDGES = (33.0, 67.0)
SUPPORT_PCTL = 95.0


@dataclasses.dataclass(frozen=True)
class CoverageBuckets:
    """Fixed thresholds plus the per-transition bucket assignment."""

    edges: tuple[float, float]
    support_threshold: float
    scores: np.ndarray

    @property
    def low_edge(self) -> float:
        return self.edges[0]

    @property
    def high_edge(self) -> float:
        return self.edges[1]

    def bucket_of(self, idx: np.ndarray | int) -> np.ndarray | int:
        """0 = low coverage, 1 = mid, 2 = high."""
        return np.digitize(self.scores[idx], self.edges)

    def bucket_sizes(self) -> dict[int, int]:
        b = np.digitize(self.scores, self.edges)
        return {k: int((b == k).sum()) for k in (0, 1, 2)}

    def in_support(self, d_to_data: np.ndarray) -> np.ndarray:
        """Filter policy-visited states by distance to the dataset."""
        return d_to_data <= self.support_threshold

    def summary(self) -> dict:
        return {
            "edges": list(self.edges),
            "support_threshold": self.support_threshold,
            "n": int(self.scores.size),
            "bucket_sizes": self.bucket_sizes(),
            "median": float(np.median(self.scores)),
        }


def load_scores(env_id: str, n_ref: int = DEFAULT_N_REF, k: int = DEFAULT_K) -> np.ndarray:
    """Load cached coverage scores, building the cache if absent."""
    p = cache_path(env_id, n_ref, k)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it first:\n"
            f"  python experiments/build_coverage_cache.py"
        )
    return np.load(p)


def build_buckets(
    scores: np.ndarray,
    edges: tuple[float, float] = BUCKET_EDGES,
    support_pctl: float = SUPPORT_PCTL,
) -> CoverageBuckets:
    """Derive fixed thresholds from the given score distribution."""
    lo, hi = np.percentile(scores, edges)
    sup = float(np.percentile(scores, support_pctl))
    return CoverageBuckets(
        edges=(float(lo), float(hi)),
        support_threshold=sup,
        scores=np.asarray(scores, dtype=np.float64),
    )


def buckets_for(env_id: str, **kw) -> CoverageBuckets:
    """Convenience: load cached scores for `env_id` and derive thresholds."""
    return build_buckets(load_scores(env_id), **kw)

