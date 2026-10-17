"""T3 — episode boundary recovery from D4RL HDF5.

D4RL datasets are flat concatenations of fixed-length episodes with no
explicit index. Boundaries must be recovered from `terminals | timeouts`.

Verified on 2026-10-04 (episode counts are the ground truth for the tests):
    halfcheetah_expert-v2            1000 episodes
    halfcheetah_medium_replay-v2      202
    walker2d_medium_replay-v2        1093
    hopper_medium_replay-v2          2041
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class EpisodeIndex:
    """Episode boundaries recovered from `terminals | timeouts`."""

    ends: np.ndarray  # inclusive index of each episode's last transition

    @classmethod
    def from_flags(cls, terminals: np.ndarray, timeouts: np.ndarray) -> "EpisodeIndex":
        ends = np.flatnonzero(terminals.astype(bool) | timeouts.astype(bool))
        if ends.size == 0:
            raise ValueError("no episode boundaries found")
        return cls(ends=ends)

    @property
    def n_episodes(self) -> int:
        return int(self.ends.shape[0])

    def starts(self) -> np.ndarray:
        return np.concatenate(([0], self.ends[:-1] + 1))

    def lengths(self) -> np.ndarray:
        return self.ends - self.starts() + 1

    def returns(self, rewards: np.ndarray) -> np.ndarray:
        """Per-episode undiscounted return."""
        cs = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)))
        return cs[self.ends + 1] - cs[self.starts()]


class EpisodeIndexLookup:
    """`steps remaining in this episode` lookup over flat transition arrays.

    Lives in the data layer (verified) rather than with any one algorithm: both
    M2's n-step window and any future sequence model need it, and D4RL stores a
    flat concatenation with no index of its own.
    """

    def __init__(self, starts: np.ndarray, ends: np.ndarray, n: int) -> None:
        self.ends = np.asarray(ends)
        eid = np.zeros(n, dtype=np.int64)
        for k, (s, e) in enumerate(zip(np.asarray(starts), self.ends)):
            eid[s : e + 1] = k
        self.episode_of = eid

    def steps_remaining(self, idx: np.ndarray) -> np.ndarray:
        """Steps from each index to its episode end, inclusive."""
        return self.ends[self.episode_of[idx]] - idx + 1
