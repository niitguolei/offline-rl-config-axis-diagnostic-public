"""M1's data augmentation: mirror-doubling the dataset as a SECOND BLOCK.

Why blocks and not interleaving
------------------------------
The n-step target bounds its window by episode boundaries. Interleaving each
transition with its mirror would make a window step from a transition onto its
own mirror image, which is not a temporal successor, so the bootstrap would be
silently wrong. Appending the mirrored copy as one block with the SAME internal
episode boundaries keeps every window inside a single real trajectory, and
doubles the number of episodes along with the number of transitions.

Why this belongs to M1 and is not a generic trick
-------------------------------------------------
Under the environment's mirror g, `f(g(s), g(a)) = g(f(s,a))` and
`r(g(s), g(a)) = r(s, a)`, so the mirrored transition is a transition of the
same MDP and the augmented dataset is a valid "more data" claim. What makes it
part of M1 rather than free data is that ONLY an exactly equivariant critic is
guaranteed to agree with itself across the two halves: a freely-fitted critic
sees them as independent samples and can fit them inconsistently. That is
precisely the (a) vs (d) contrast GATE 2 measures, so augmentation without the
constraint is the control, not the method.

Where it sits in the pipeline
-----------------------------
Mirroring is applied in NORMALISED space, after the normaliser has been fitted
and applied, because that is the space the network reads. That is legitimate
only when the normaliser commutes with g, i.e. when its statistics have been
symmetrised (`symmetrise_normer_stats`). Otherwise the augmented datum is not
the mirror of the original one in the space that matters. The guard below
enforces this instead of trusting the caller -- the residual was 3.22 std on
walker2d_medium_replay before symmetrisation, so a silent violation here is
large, not cosmetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..data.episode import EpisodeIndexLookup
from .equivariance import MirrorSpec, normaliser_commutes


@dataclasses.dataclass(frozen=True)
class Augmented:
    """A dataset that is the original block followed by its mirror image."""

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    n_original: int

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])

    @property
    def mirror_slice(self) -> slice:
        return slice(self.n_original, self.n)


def assert_normaliser_carries_mirror(mean: np.ndarray, std: np.ndarray,
                                     spec: MirrorSpec, tol: float = 1e-9) -> None:
    """Fail loudly if the normalisation does not commute with the mirror."""
    r = normaliser_commutes(mean, std, spec)
    if not (r["max_abs_residual_std"] <= tol):
        raise AssertionError(
            "normalisation does not commute with the mirror "
            f"(max residual {r['max_abs_residual_std']:.3e} std): mirroring "
            "normalised observations would not be the environment's symmetry. "
            "Load with a symmetrised normaliser first.")


def require_symmetry(domain: str, spec: MirrorSpec, tol: float = 1e-6,
                     n: int = 8) -> float:
    """Refuse to proceed unless the spec IS a symmetry of the environment.

    Augmenting with a map that is not a symmetry of the dynamics does not add
    data, it adds WRONG data: the mirrored transition is then not a transition
    of the same MDP, so the critic is trained toward targets the environment
    never produces. The same applies to weight tying, which would impose
    V(s) = V(g(s)) for a g the true value does not respect.

    This guard exists because the spec shipped in `derive_mirror` fails it. It
    pairs joint names containing "_left" and flips rooty, which for these
    PLANAR models is not the environment's mirror at all; see
    docs/GATES.md, "M1 的前提被证伪".
    """
    from .equivariance import symmetry_defect

    defect = symmetry_defect(domain, spec, n=n)
    if not defect <= tol:
        raise AssertionError(
            f"{domain}: the mirror spec is NOT a symmetry of the environment "
            f"(defect {defect:.6f} > tol {tol}). Augmenting or tying with it "
            "would inject transitions the MDP never produces. Measured with a "
            "simulator-only oracle, so no dataset or version difference is "
            "involved.")
    return defect


def mirror_augment(
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    next_obs: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    spec: MirrorSpec,
) -> Augmented:
    """Original block, then its mirror image. Rewards/terminals are invariant."""
    n = int(obs.shape[0])
    for name, arr in (("act", act), ("rew", rew), ("next_obs", next_obs),
                      ("terminals", terminals), ("timeouts", timeouts)):
        if arr.shape[0] != n:
            raise ValueError(f"{name} has {arr.shape[0]} rows, observations {n}")
    return Augmented(
        obs=np.concatenate([obs, spec.apply_obs(obs)], axis=0),
        act=np.concatenate([act, spec.apply_act(act)], axis=0),
        # r(g(s), g(a)) == r(s, a): rewards are invariant, so they repeat
        # rather than get transformed. Same for the boundary flags.
        rew=np.concatenate([rew, rew], axis=0),
        next_obs=np.concatenate([next_obs, spec.apply_obs(next_obs)], axis=0),
        terminals=np.concatenate([terminals, terminals], axis=0),
        timeouts=np.concatenate([timeouts, timeouts], axis=0),
        n_original=n,
    )


def augmented_episode_lookup(ep: EpisodeIndexLookup,
                             n_original: int) -> EpisodeIndexLookup:
    """Episode structure of the two-block array: boundaries repeat, offset.

    The mirrored block is an exact copy of the original's temporal structure,
    so episode k of the first block and episode (n_episodes + k) of the second
    have identical lengths. Getting this wrong is the difference between a
    valid n-step window and one that bootstraps across unrelated data, so the
    offset is applied in exactly one place.
    """
    starts = np.concatenate(([0], ep.ends[:-1] + 1))
    lengths = ep.ends - starts + 1
    if int(lengths.sum()) != n_original:
        raise ValueError(
            f"episode lengths sum to {lengths.sum()}, expected {n_original}")
    ends2 = np.concatenate([ep.ends, ep.ends + n_original])
    starts2 = np.concatenate([starts, starts + n_original])
    return EpisodeIndexLookup(starts2, ends2, 2 * n_original)
