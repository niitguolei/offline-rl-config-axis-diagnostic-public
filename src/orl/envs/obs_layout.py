"""Observation / action layout metadata, derived from the live MuJoCo model.

Why this exists: S0 must NOT hand-write an observation index map, and it must
NOT guess a mirror transformation. Both are easy to get subtly wrong
(halfcheetah and walker2d differ in this respect), and a wrong map would
silently corrupt the whole M1 line of work.

So S0 only derives and records the *structure* from gymnasium's own model:
how the observation vector splits into its qpos/qvel parts, plus joint and
actuator names. Constructing the actual mirror operator is stage S2
(transform correctness check) and is deliberately out of scope here.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym

ENV_IDS = {
    "halfcheetah": "HalfCheetah-v5",
    "walker2d": "Walker2d-v5",
    "hopper": "Hopper-v5",
}


@dataclasses.dataclass(frozen=True)
class ObsLayout:
    domain: str
    obs_dim: int
    nq_minus_one: int  # qpos[1:] contributes this many dims (global x excluded)
    nv: int
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]

    @property
    def qpos_slice(self) -> slice:
        return slice(0, self.nq_minus_one)

    @property
    def qvel_slice(self) -> slice:
        return slice(self.nq_minus_one, self.nq_minus_one + self.nv)


def describe(domain: str) -> ObsLayout:
    """Introspect the gymnasium model for `domain`."""
    env = gym.make(ENV_IDS[domain])
    try:
        u = env.unwrapped
        nq, nv = int(u.model.nq), int(u.model.nv)
        obs_dim = int(u.observation_space.shape[0])
        joint_names = tuple(u.model.joint(j).name for j in range(int(u.model.njnt)))
        actuator_names = tuple(
            u.model.actuator(a).name for a in range(int(u.model.nu))
        )
    finally:
        env.close()
    return ObsLayout(
        domain=domain,
        obs_dim=obs_dim,
        nq_minus_one=nq - 1,
        nv=nv,
        joint_names=joint_names,
        actuator_names=actuator_names,
    )


def assert_dataset_matches_env(dataset_obs_dim: int, domain: str) -> None:
    """T4 acceptance: dataset obs dim must equal the env's.

    All three retained domains were verified to match on 2026-10-03; this
    turns that finding into a regression guard. ant is the counterexample
    (111 vs 105) and is permanently excluded.
    """
    layout = describe(domain)
    if dataset_obs_dim != layout.obs_dim:
        raise AssertionError(
            f"{domain}: dataset obs dim {dataset_obs_dim} != env {layout.obs_dim}. "
            "If this is ant, note ant is permanently excluded for exactly this reason."
        )

