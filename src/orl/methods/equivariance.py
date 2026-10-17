"""M1 -- symmetry-equivariant value/critic via weight tying.

Mechanism
---------
Offline RL applies ONE global conservatism coefficient to every state. But
the optimal policy respects the environment's symmetry group, while a freely
learned value function does not. The result is that pessimism is misallocated
across symmetry orbits: too weak where the symmetry already pins the value
down, too strong where it does not.

M1 removes the freedom instead of penalising it. If g is the symmetry, the
equivariant network is the *canonical* one:

    f_eq(x) = A f(B x)        B permutes/flips the INPUT, A the OUTPUT

This is representable INSIDE the same architecture by touching only the first
and last Linear layers:

    W1' = W1 @ B,   Wk' = A @ Wk,   bk' = A @ bk

and averaging with the original gives an exactly equivariant network:

    W_eq = (W + W_sym)/2

No penalty term, no extra hyperparameter, no change to the loss. This is why
it is a mechanism and not a trick: the ONLY downstream change is that
advantage = Q(s,a) - V(s) becomes equivariant, so expectile regression and
AWR weighting act consistently across the orbit.

The per-domain mirror spec is DATA, and getting it wrong silently destroys the
whole direction -- so `tests/test_equivariance.py` verifies each spec against
the live MuJoCo model rather than trusting the table.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from .blocks import concat_blocks, concat_signs


@dataclasses.dataclass(frozen=True)
class MirrorSpec:
    """Permutation + sign flip describing a mirror transform of (s, a)."""

    obs_perm: np.ndarray
    obs_sign: np.ndarray
    act_perm: np.ndarray
    act_sign: np.ndarray

    def apply_obs(self, x: np.ndarray) -> np.ndarray:
        return (x[..., self.obs_perm] * self.obs_sign)

    def apply_act(self, x: np.ndarray) -> np.ndarray:
        return (x[..., self.act_perm] * self.act_sign)

    def is_involutive(self) -> bool:
        """Check g(g(x)) == x SEMANTICALLY.

        `obs_perm` concatenates two independent blocks (qpos[1:] of length
        nq-1 and qvel of length nv) acting on DISJOINT index ranges, so
        `perm[perm]` is not the right test -- numpy would gather both blocks
        through the same index space. Applying the transform twice is.
        """
        for perm, sign in ((self.obs_perm, self.obs_sign),
                           (self.act_perm, self.act_sign)):
            if len(np.unique(perm)) != len(perm):
                return False
            if np.any(np.abs(sign) != 1):
                return False
        x = np.random.default_rng(0).normal(size=(8, len(self.obs_perm)))
        if not np.allclose(self.apply_obs(self.apply_obs(x)), x):
            return False
        a = np.random.default_rng(1).normal(size=(8, len(self.act_perm)))
        return bool(np.allclose(self.apply_act(self.apply_act(a)), a))

def derive_mirror(domain: str, obs_dim: int, act_dim: int) -> MirrorSpec:
    """Derive the left/right mirror from the live MuJoCo model.

    D4RL observations are `qpos[1:] + qvel`. Joint names come from the model,
    so a `<name>_left` joint is mapped onto `<name>`. Lateral rotation DOFs
    (`rooty` and its velocity) flip sign; all other DOFs are unchanged.

    HalfCheetah has NO left/right joint pair (its legs are front/back), so its
    mirror reduces to a single sign flip on `rooty` -- i.e. nearly trivial.
    Walker2d has explicit `*_left` pairs and gets a genuine permutation.
    """
    import gymnasium as gym

    from ..envs.obs_layout import ENV_IDS

    env = gym.make(ENV_IDS[domain])
    try:
        model = env.unwrapped.model
        names = [model.joint(j).name for j in range(model.njnt)]
        act_names = [model.actuator(a).name for a in range(model.nu)]
        # Several of these models ship with EMPTY actuator names, so the
        # actuator -> joint mapping must come from the model itself.
        # These models ship with EMPTY actuator names (walker2d, hopper), and
        # `actuator_trnid2` does not exist in these mujoco bindings. In every
        # D4RL locomotion model the actuators drive the LAST nu joints, which
        # is verified below whenever real actuator names ARE provided.
        derived = names[-model.nu:]
        if all(act_names) and list(act_names) != derived:
            raise AssertionError(
                f"{domain}: actuator names {act_names} != last-nu joints {derived}")
        act_names = derived
    finally:
        env.close()

    # obs = qpos[1:] (nq-1 entries) + qvel (nv entries).
    # NOTE the two blocks have DIFFERENT lengths: qvel also carries the rootx
    # velocity. Using one permutation for both silently yields a 16-dim spec
    # for a 17-dim observation -- the bug this comment documents.
    pos_names = names[1:]
    n_pos, n_vel = len(pos_names), len(names)
    if obs_dim != n_pos + n_vel:
        raise AssertionError(
            f"{domain}: obs_dim {obs_dim} != qpos[1:]({n_pos}) + qvel({n_vel})")

    def pair(i: int, pool: list[str]) -> int | None:
        """Index of the left/right partner of pool[i], or None.

        Matching is on the name with "_left" removed, because the D4RL models
        name the pair `thigh_joint` / `thigh_left_joint` -- "_left" is inserted
        BEFORE "_joint", so an append-style rule finds nothing.
        """
        key = pool[i].replace("_left", "")
        for j, other in enumerate(pool):
            if j != i and other.replace("_left", "") == key:
                return j
        return None
    # --- qpos[1:] block
    perm_pos = np.arange(n_pos)
    sign_pos = np.ones(n_pos)
    for i, nm in enumerate(pos_names):
        j = pair(i, pos_names)
        if j is not None:
            perm_pos[i], perm_pos[j] = j, i
        if nm == "rooty":
            sign_pos[i] = -1.0

    # --- qvel block: same joint ordering, and it also carries rootx velocity
    perm_vel = np.arange(n_vel)
    sign_vel = np.ones(n_vel)
    for i, nm in enumerate(names):
        j = pair(i, names)
        if j is not None:
            perm_vel[i], perm_vel[j] = j, i
        if nm == "rooty":
            sign_vel[i] = -1.0

    # Block-relative indices MUST be offset; concat_blocks is the single
    # place that arithmetic lives (see blocks.py for why).
    obs_perm = concat_blocks(perm_pos, perm_vel)
    obs_sign = concat_signs(sign_pos, sign_vel)

    # --- actions, from the model's ACTUATOR names
    if act_dim != len(act_names):
        raise AssertionError(f"{domain}: act_dim {act_dim} != nu {len(act_names)}")
    a_perm = np.arange(act_dim)
    a_sign = np.ones(act_dim)
    for i, nm in enumerate(act_names):
        j = pair(i, act_names)
        if j is not None:
            a_perm[i], a_perm[j] = j, i

    return MirrorSpec(obs_perm=obs_perm, obs_sign=obs_sign,
                      act_perm=a_perm, act_sign=a_sign)


def prepare_mirror_tensors(
    in_perm: np.ndarray, in_sign: np.ndarray,
    out_perm: np.ndarray, out_sign: np.ndarray, device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upload the mirror indices to `device` ONCE.

    Tying happens after every optimiser step, so rebuilding these four tiny
    tensors per call was 24 host-to-device copies per training step and
    dominated M1's overhead (measured 9.2 -> 15.2 ms/step on this machine).
    """
    return (
        torch.as_tensor(in_perm, dtype=torch.long, device=device),
        torch.as_tensor(in_sign, dtype=torch.float32, device=device),
        torch.as_tensor(out_perm, dtype=torch.long, device=device),
        torch.as_tensor(out_sign, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def symmetrize_linear_stack(
    net: nn.Module,
    in_perm: np.ndarray,
    in_sign: np.ndarray,
    out_perm: np.ndarray,
    out_sign: np.ndarray,
    prepared: tuple[torch.Tensor, ...] | None = None,
) -> None:
    """In-place weight tying so that f(x) becomes equivariant in place.

    Only the first and last Linear layers change: intermediate hidden layers
    act on the hidden representation, which the symmetry does not touch.
    """
    lins = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if len(lins) < 2:
        raise ValueError("need at least two Linear layers to symmetrise")
    first, last = lins[0], lins[-1]
    if prepared is None:
        prepared = prepare_mirror_tensors(
            in_perm, in_sign, out_perm, out_sign, first.weight.device)
    ip, isg, op, osg = prepared

    # Reynolds operator (I + T)/2, where T conjugates by the mirror:
    #     first:  T(W) = W[:, ip] * sign         (b1 untouched: it is hidden)
    #     last:   T(W) = W[op, :] * sign[:, None]
    # Because g is an involution T^2 = I, so this is equally (T + T^2)/2 --
    # averaging the mirrored weights with each other gives the same answer.
    # Written as (I + T)/2 because that is what it means: the projection onto
    # the subspace the symmetry leaves fixed.
    W0 = first.weight.data
    first.weight.data = 0.5 * (W0 + W0[:, ip] * isg.unsqueeze(0))
    Wk = last.weight.data
    last.weight.data = 0.5 * (Wk + Wk[op, :] * osg.unsqueeze(1))
    if last.bias is not None:
        bk = last.bias.data.clone()
        last.bias.data = 0.5 * (bk + bk[op] * osg)


def apply_mirror_obs(spec: MirrorSpec, obs: torch.Tensor) -> torch.Tensor:
    """g(s): permute then sign-flip, batched."""
    idx = torch.as_tensor(spec.obs_perm, dtype=torch.long, device=obs.device)
    sg = torch.as_tensor(spec.obs_sign, dtype=obs.dtype, device=obs.device)
    return obs[:, idx] * sg


def apply_mirror_act(spec: MirrorSpec, act: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(spec.act_perm, dtype=torch.long, device=act.device)
    sg = torch.as_tensor(spec.act_sign, dtype=act.dtype, device=act.device)
    return act[:, idx] * sg


def symmetrise_normer_stats(
    mean: np.ndarray, std: np.ndarray, spec: MirrorSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Orbit-average normalisation statistics so that they COMMUTE with g.

    Why this is not optional
    -------------------------
    The network never sees the raw state, it sees `n(s)_i = (s_i - mu_i)/sd_i`.
    Tying the weights makes it equivariant under the mirror *of the space it
    actually reads*. If normalisation does not commute with g, that space
    carries a different, contaminated group element and the enforced symmetry
    is not the environment's symmetry. Measured on
    walker2d_medium_replay-v2, the raw per-dim statistics break commutation
    badly: `rooty` has a lateral drift of 0.74 std and the foot-joint pair
    differs by 0.70 std in mean and 0.64 std in scale. So this is a real
    property of a medium-quality behaviour policy, not a rounding artefact.

    Commutation holds iff
        sd_i == sd_{perm(i)}          (from matching the s coefficients)
        mu_i == sign_i * mu_{perm(i)} (from matching the constants)
    and averaging each pair over its orbit satisfies both while staying as
    close as possible to the empirical statistics. `sign` is unchanged by the
    permutation because g is an involution, which is what makes the averaged
    offset symmetric rather than merely symmetric-looking.
    """
    mu = np.asarray(mean, dtype=np.float64).copy()
    sd = np.asarray(std, dtype=np.float64).copy()
    perm, sign = spec.obs_perm, spec.obs_sign
    if mu.shape[0] != perm.shape[0] or sd.shape[0] != perm.shape[0]:
        raise ValueError("normaliser width must match the mirror spec")
    sd = 0.5 * (sd + sd[perm])
    mu = 0.5 * (mu + sign * mu[perm])
    return mu, sd


def normaliser_commutes(
    mean: np.ndarray, std: np.ndarray, spec: MirrorSpec, n: int = 512,
    seed: int = 0,
) -> dict:
    """Empirical check that `n(g.s) == g(n(s))`, on random states.

    Deliberately empirical rather than a restatement of the algebra: the
    algebra is three lines, but the failure it guards against happened
    silently twice already. The residual is reported in std units so it is
    comparable to the observed data asymmetry.
    """
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    sd = np.where(sd < 1e-12, 1.0, sd)
    rng = np.random.default_rng(seed)
    s = mu + sd * rng.normal(size=(n, mu.shape[0]))

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - mu) / sd

    lhs = norm(spec.apply_obs(s))              # n(g.s)
    rhs = spec.apply_obs(norm(s))              # g(n(s))
    return {
        "max_abs_residual_std": float(np.abs(lhs - rhs).max()),
        "mean_abs_residual_std": float(np.abs(lhs - rhs).mean()),
    }


def symmetry_defect(domain: str, spec: MirrorSpec, n: int = 8, seed: int = 0,
                    tol: float = 1e-6) -> float:
    """Measure `max |f(g.x, g.a) - g.f(x, a)|` USING ONLY THE SIMULATOR.

    This is the oracle that decides whether a mirror spec is a symmetry of the
    environment at all, and it deliberately touches no dataset: D4RL's stored
    transitions were produced by a different MuJoCo vintage and do not
    reproduce in the current simulator (measured: max |dobs| 0.03-0.13 on
    halfcheetah, 0.008-0.04 on hopper, and up to 13.7 on walker2d, whose
    observations clip qvel to +-10 so a stored state cannot even be
    reconstructed). Using logged transitions as the oracle would have made a
    version difference look like a symmetry violation. Sampling states
    directly samples the model, so the only thing left in the number is the
    symmetry itself.

    Sampling keeps |qvel| <= 6 so walker2d's clipping never saturates.

    Returns the defect; `tol` is only a convenience for callers that want the
    pass/fail in one call. A spec is a genuine symmetry iff the defect is at
    the floating-point noise floor.
    """
    import gymnasium as gym

    from ..envs.obs_layout import ENV_IDS

    env = gym.make(ENV_IDS[domain])
    u = env.unwrapped
    nq, nv = int(u.model.nq), int(u.model.nv)
    rng = np.random.default_rng(seed)
    worst = 0.0
    try:
        for _ in range(n):
            qpos = np.zeros(nq)
            qvel = rng.uniform(-1.0, 1.0, nv)
            qpos[1] = rng.uniform(0.5, 1.5)
            if nq > 2:
                qpos[2] = rng.uniform(-0.3, 0.3)
            if nq > 3:
                qpos[3:] = rng.uniform(-0.8, 0.8, nq - 3)
            a = rng.uniform(-1.0, 1.0, spec.act_perm.shape[0])
            obs = np.concatenate([qpos[1:], qvel])

            env.reset(seed=0)
            u.set_state(qpos, qvel)
            x_next = np.asarray(env.step(a)[0], dtype=np.float64)

            qm, vm = _split_obs(spec.apply_obs(obs), nq, nv)
            env.reset(seed=0)
            u.set_state(qm, vm)
            y_next = np.asarray(
                env.step(np.asarray(spec.apply_act(a), dtype=np.float64))[0],
                dtype=np.float64)

            if np.abs(x_next[nq - 1:]).max() >= 9.9:
                continue        # walker2d would have clipped; skip the sample
            worst = max(worst, float(np.abs(y_next - spec.apply_obs(x_next)).max()))
    finally:
        env.close()
    return worst


def _split_obs(obs_row: np.ndarray, nq: int, nv: int):
    """qpos[0] is decoupled from the dynamics, so 0.0 is a valid stand-in."""
    qpos = np.concatenate([[0.0], obs_row[: nq - 1]]).astype(np.float64)
    qvel = obs_row[nq - 1: nq - 1 + nv].astype(np.float64)
    return qpos, qvel


def symmetry_defect_full(domain: str, spec: MirrorSpec, n: int = 8,
                         seed: int = 0) -> dict:
    """Both halves of what "symmetry of the MDP" means: dynamics AND reward.

    `symmetry_defect` only compares observations, which answers
    `f(g.x, g.a) == g.f(x, a)`. A value function is invariant under g only if
    the REWARD is invariant too, and the two can fail independently: a
    reflection that moves nothing but leaves the reward asymmetric would still
    make `V(s) = V(g(s))` false. Checking only the dynamics is the same
    half-verification that this project was already burned by once, so the
    oracle reports both.
    """
    import gymnasium as gym

    from ..envs.obs_layout import ENV_IDS

    env = gym.make(ENV_IDS[domain])
    u = env.unwrapped
    nq, nv = int(u.model.nq), int(u.model.nv)
    rng = np.random.default_rng(seed)
    d_obs = d_rew = 0.0
    try:
        for _ in range(n):
            qpos = np.zeros(nq)
            qvel = rng.uniform(-1.0, 1.0, nv)
            qpos[1] = rng.uniform(0.5, 1.5)
            if nq > 2:
                qpos[2] = rng.uniform(-0.3, 0.3)
            if nq > 3:
                qpos[3:] = rng.uniform(-0.8, 0.8, nq - 3)
            a = rng.uniform(-1.0, 1.0, spec.act_perm.shape[0])
            obs = np.concatenate([qpos[1:], qvel])

            env.reset(seed=0)
            u.set_state(qpos, qvel)
            x_next, r_x, _, _, _ = env.step(a)

            qm, vm = _split_obs(spec.apply_obs(obs), nq, nv)
            env.reset(seed=0)
            u.set_state(qm, vm)
            y_next, r_y, _, _, _ = env.step(
                np.asarray(spec.apply_act(a), dtype=np.float64))

            d_rew = max(d_rew, abs(float(r_y) - float(r_x)))
            if np.abs(np.asarray(x_next)[nq - 1:]).max() >= 9.9:
                continue        # walker2d would have clipped; skip the sample
            d_obs = max(d_obs, float(np.abs(
                np.asarray(y_next, dtype=np.float64)
                - spec.apply_obs(np.asarray(x_next, dtype=np.float64))).max()))
    finally:
        env.close()
    return {"obs": d_obs, "reward": d_rew}
