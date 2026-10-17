"""Observation normalisation -- MISSING FROM THE FIRST IQL IMPLEMENTATION.

Diagnosis (2026-10-05): without it the critic barely learns.

  * halfcheetah_medium_replay observations have per-dimension std from 0.170
    to 9.106 -- a 53.5x spread. Dims 11-16 (joint velocities) dominate the
    first-layer gradient; dim 0 contributes almost nothing.
  * Consequence measured after 150k steps: corr(Q, r) = 0.9986 and
    std(Q)/std(r) = 1.006 -- the critic had collapsed to regressing the
    IMMEDIATE reward with essentially no state dependence, Q/trueV = 0.17.
  * n-step then bootstraps 5 steps further into that broken V and diverges
    (q_loss 60 -> 480 -> 596,813).

The reference IQL implementation normalises observations with a running
mean/std. Omitting it is the single deviation that explains the divergence
diagnostic, so it is fixed here rather than tuned around.

Statistics are computed once over the dataset and stored, not estimated
online, so a run is exactly reproducible from its config.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ObsNormalizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-3

    @classmethod
    def fit(cls, obs: np.ndarray, eps: float = 1e-3) -> "ObsNormalizer":
        o = np.asarray(obs, dtype=np.float64)
        return cls(mean=o.mean(0), std=o.std(0) + eps, eps=eps)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return ((np.asarray(obs, dtype=np.float32) - self.mean) / self.std
                ).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict) -> "ObsNormalizer":
        return cls(mean=np.asarray(d["mean"], dtype=np.float64),
                   std=np.asarray(d["std"], dtype=np.float64),
                   eps=float(d.get("eps", 1e-3)))

