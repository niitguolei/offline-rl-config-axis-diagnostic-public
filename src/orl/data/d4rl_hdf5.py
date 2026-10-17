"""I1 — D4RL HDF5 reader.

Design note (this mattered for the S0 wall-clock benchmark):
HDF5 with scattered hyperslab reads is 10-100x slower than RAM. If the
training path read random minibatches straight off disk, the measured
per-step cost `b` would be dominated by I/O and we would wrongly conclude
"data loading is the bottleneck". So:

  mode="ram"    (default) load the transition arrays into RAM once.
                The largest dataset (halfcheetah_medium_expert-v2, 2M
                transitions) needs ~330 MB, well inside the 1.5 GB budget.
  mode="stream" read hyperslabs on demand. Slower, but memory-flat. Used
                for inspection and for validating the RAM path.

Both modes are byte-identical; `tests/test_data.py` asserts that.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np

from ..paths import dataset_path, parse_env_id
from .episode import EpisodeIndex
from .normalize import ObsNormalizer

REQUIRED_KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")


class D4RLDataset:
    """One D4RL locomotion dataset."""

    def __init__(
        self,
        env_id: str,
        mode: str = "ram",
        normalize: bool = False,
        path: pathlib.Path | None = None,
        load_next_obs: bool = True,
    ) -> None:
        if mode not in ("ram", "stream"):
            raise ValueError(f"mode must be 'ram' or 'stream', got {mode!r}")
        self.env_id = env_id
        self.mode = mode
        self.path = pathlib.Path(path) if path else dataset_path(env_id)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. D4RL locomotion must be downloaded to "
                "ORL_DATA_DIR (see docs/ENVIRONMENT.md)."
            )
        self.domain, self.tier = parse_env_id(env_id)
        self.load_next_obs = load_next_obs
        self.normalize = normalize
        self._normer: ObsNormalizer | None = None
        self._f: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._open()

    # ------------------------------------------------------------------ open
    def _open(self) -> None:
        self._f = h5py.File(self.path, "r")
        missing = [k for k in REQUIRED_KEYS if k not in self._f]
        if missing:
            self._f.close()
            raise KeyError(f"{self.path.name}: missing keys {missing}")
        # Vintage guard. The old `gym_mujoco/` tree served files WITHOUT
        # next_observations and with half the transitions (halfcheetah
        # medium-replay: 101,000 vs the canonical 202,000). Mixing them
        # silently invalidates comparison with published -v2 numbers.
        self.has_next_obs = "next_observations" in self._f
        if not self.has_next_obs:
            self._f.close()
            raise ValueError(
                f"{self.path.name} has no next_observations -> WRONG VINTAGE. "
                "Use the gym_mujoco_v2/ files, not the old gym_mujoco/ tree."
            )
        self.n = int(self._f["observations"].shape[0])
        self.obs_dim = int(self._f["observations"].shape[1])
        self.act_dim = int(self._f["actions"].shape[1])
        self.episodes = EpisodeIndex.from_flags(
            self._f["terminals"][:], self._f["timeouts"][:]
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "D4RLDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ read
    def _read(self, key: str, sl: slice | np.ndarray | None = None) -> np.ndarray:
        assert self._f is not None
        if key == "next_observations" and not self.load_next_obs:
            raise KeyError("next_observations not loaded (load_next_obs=False)")
        arr = self._f[key][sl] if sl is not None else self._f[key][:]
        # next_observations MUST be float. Leaving it out of this branch made
        # it fall through to the bool cast below, so V(s') was evaluated on an
        # all-True array. That single bug produced Q ~= reward + const
        # (corr 0.9986) and made n-step diverge.
        if key in ("observations", "next_observations", "actions"):
            out = np.asarray(arr, dtype=np.float32)
            if key == "observations" and self.normalize:
                return self._normalizer()(out)
            return out
        if key == "rewards":
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(arr, dtype=bool)

    def _normalizer(self) -> ObsNormalizer:
        """Fit once on the FULL observation array (never on a slice)."""
        if self._normer is None:
            assert self._f is not None
            self._normer = ObsNormalizer.fit(self._f["observations"][:])
        return self._normer

    def get(self, key: str, sl: slice | np.ndarray | None = None) -> np.ndarray:
        """Read `key`, using the RAM cache when mode='ram'."""
        if self.mode == "ram":
            if key not in self._cache:
                self._cache[key] = self._read(key)
            return self._cache[key] if sl is None else self._cache[key][sl]
        return self._read(key, sl)

    def sample(
        self, batch_size: int, rng: np.random.Generator, keys: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        """Uniform minibatch over transitions."""
        if keys is None:
            keys = ("observations", "actions", "rewards", "next_observations")
        idx = rng.integers(0, self.n, size=batch_size)
        out = {}
        for k in keys:
            v = self.get(k)
            out[k] = v[idx]
        return out

    # ------------------------------------------------------------- summaries
    def summary(self) -> dict:
        r = self.get("rewards")
        rets = self.episodes.returns(r)
        a = self.get("actions")
        return {
            "env_id": self.env_id,
            "domain": self.domain,
            "tier": self.tier,
            "n_transitions": self.n,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "n_episodes": self.episodes.n_episodes,
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std()),
            "return_rel_std": float(100 * rets.std() / abs(rets.mean())),
            "action_min": float(a.min()),
            "action_max": float(a.max()),
        }


def open_dataset(env_id: str, mode: str = "ram") -> D4RLDataset:
    return D4RLDataset(env_id, mode=mode)
