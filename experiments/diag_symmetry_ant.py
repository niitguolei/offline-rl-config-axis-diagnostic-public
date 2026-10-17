"""P0: does D4RL's Ant have a non-trivial mirror symmetry? (budget 2 h)

Why this exists: M1 died because its mirror was inferred from JOINT NAMES.
The check that caught it is a simulator oracle, and D-0010 reduced this path to
the single question "does a non-planar model have a real mirror?" plus its use
as a cautionary case study. See docs/P0方案3前提检验计划.md.

Three layout facts established by reconnaissance, each of which would silently
produce a wrong answer if assumed:

  1. D4RL's ant datasets are 111-dimensional: qpos[2:] (13) + qvel (14) +
     cfrc_ext (84). Gymnasium's Ant-v4 defaults to 27 dims
     (use_contact_forces=False), so the oracle MUST be built on
     `Ant-v4(use_contact_forces=True)` to match the data.
  2. Ant drops TWO positions (x and y) from the observation, not one. The
     existing `_split` in envs/replay.py assumes one, because it was written for
     the planar walkers; reusing it here would put z into the y slot. This file
     therefore carries its own layout-aware split.
  3. The mirror is NOT the identity: the four legs sit at (+-0.4, +-0.4), so the
     reflection pairs front_left<->right_back and front_right<->back_leg.

Transformation rules used to build the candidate (all verified by the oracle
rather than trusted):

    polar vector (position, linear velocity, force)   (x, y, z) -> (x, -y, z)
    pseudo vector (quaternion vector part, ang. vel, torque) -> (-x, y, -z)
    hinge joint paired across the mirror:  axis n maps to D n; the mirrored
        angle is +theta if D n == -n' (axis reverses) and -theta otherwise.
    action: torque is conjugate to the angle, so ctrl maps with the same sign as
        the angle, divided by the gear ratio (all gears are +150 here).

    python experiments/diag_symmetry_ant.py --smoke
    python experiments/diag_symmetry_ant.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.path.insert(0, str(ROOT))

from orl.methods.equivariance import MirrorSpec  # noqa: E402
from orl.paths import BENCH_DIR  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

ENV_KW = {"use_contact_forces": True}
D4RL_ANT = "ant_medium_replay-v2"


def split_ant(obs_row: np.ndarray, nq: int, nv: int):
    """Ant's observation drops qpos[0] and qpos[1]; the planar helper drops one."""
    qpos = np.concatenate([[0.0, 0.0], obs_row[: nq - 2]]).astype(np.float64)
    qvel = obs_row[nq - 2: nq - 2 + nv].astype(np.float64)
    return qpos, qvel


def load_samples(env_id: str, n: int, seed: int):
    """Ant states and actions from the real dataset, so contacts are realistic."""
    from orl.data.d4rl_hdf5 import D4RLDataset

    with D4RLDataset(env_id, mode="ram") as ds:
        obs = ds.get("observations")
        act = ds.get("actions")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(obs.shape[0], size=n, replace=False))
    return obs[idx].astype(np.float64), act[idx].astype(np.float64), idx


def build_oracle(env):
    """Return (defect, n_contacts) over a fixed sample set, reusing one env."""
    import mujoco

    u = env.unwrapped
    m = u.model
    nq, nv = int(m.nq), int(m.nv)

    def defect(spec: MirrorSpec, obs: np.ndarray, act: np.ndarray,
               return_parts: bool = False):
        d_obs = d_rew = 0.0
        contacts = 0
        for i in range(obs.shape[0]):
            qpos, qvel = split_ant(obs[i, : nq - 2 + nv], nq, nv)
            env.reset(seed=0)
            u.set_state(qpos, qvel)
            x_next, r_x, _, _, _ = env.step(act[i])
            contacts += int(u.data.ncon)

            qm, vm = split_ant(spec.apply_obs(obs[i]), nq, nv)
            env.reset(seed=0)
            u.set_state(qm, vm)
            y_next, r_y, _, _, _ = env.step(
                np.asarray(spec.apply_act(act[i]), dtype=np.float64))

            d_rew = max(d_rew, abs(float(r_y) - float(r_x)))
            d_obs = max(d_obs, float(np.abs(
                np.asarray(y_next, dtype=np.float64)
                - spec.apply_obs(np.asarray(x_next, dtype=np.float64))).max()))
        if return_parts:
            return {"obs": d_obs, "reward": d_rew, "contacts": contacts}
        return max(d_obs, d_rew)

    return defect


def identity_spec(env) -> MirrorSpec:
    u = env.unwrapped
    m = u.model
    od = int(m.nq) - 2 + int(m.nv)
    if m.nq == 15:          # ant with contact forces
        od = 13 + 14 + 6 * m.nbody
    return MirrorSpec(np.arange(od), np.ones(od), np.arange(m.nu), np.ones(m.nu))


def derive_spec(env) -> dict:
    """Structural derivation. Every rule here is a hypothesis, not a fact."""
    import mujoco

    u = env.unwrapped
    m = u.model
    nq, nv, nbody, nu = int(m.nq), int(m.nv), int(m.nbody), int(m.nu)
    n_hinge = nq - 7

    # --- body pairing under D = diag(1, -1, 1)
    #
    # Pairing by a body's OWN resting position is ambiguous: at the rest pose
    # the torso and all four leg-root bodies sit at the same point, so an
    # argmin tie paired the torso with `world`. The fix is to match SUBTREES:
    # the leg roots coincide but their descendants diverge to +-y, which is
    # exactly what distinguishes front_left from right_back. A signature is the
    # sorted set of a subtree's world positions, so it is invariant to the
    # order bodies are visited in.
    children = {b: [] for b in range(nbody)}
    for b in range(1, nbody):
        children[int(m.body_parentid[b])].append(b)

    def subtree(b):
        out, stack = [b], [b]
        while stack:
            for c in children[stack.pop()]:
                out.append(c)
                stack.append(c)
        return out

    rest = np.zeros(nq)
    rest[2] = 0.75                       # keep bodies off the ground plane
    if nq >= 7:
        rest[3] = 1.0                    # identity quaternion
    u.set_state(rest, np.zeros(nv))
    mujoco.mj_forward(m, u.data)
    xpos = np.array(u.data.xpos)
    D = np.array([1.0, -1.0, 1.0])

    def sig(bs, flip):
        pts = np.array([xpos[b] * (D if flip else 1.0) for b in bs])
        return tuple(np.round(np.sort(pts, axis=0).ravel(), 6).tolist())

    body_pair = {}
    for b in range(1, nbody):            # body 0 is `world`; it owns no geom
        want = sig(subtree(b), flip=True)
        matches = [k for k in range(1, nbody) if sig(subtree(k), flip=False) == want]
        body_pair[b] = matches[0] if len(matches) == 1 else None

    # --- joint pairing and sign
    jnt_pair, jnt_sign = {}, {}
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = int(m.jnt_bodyid[j])
        b2 = body_pair.get(b)
        cands = [k for k in range(m.njnt)
                 if m.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE
                 and int(m.jnt_bodyid[k]) == b2]
        best, bestd = None, 9e9
        for k in cands:
            gap = np.abs(m.jnt_qposadr[k] - m.jnt_qposadr[j])
            if gap < bestd:
                best, bestd = k, gap
        jnt_pair[j] = best
        if best is not None:
            n, n2 = np.array(m.jnt_axis[j]), np.array(m.jnt_axis[best])
            jnt_sign[j] = -1.0 if np.allclose(n * D, n2, atol=1e-6) else 1.0

    od = (nq - 2) + nv + 6 * nbody
    perm = np.arange(od)
    sign = np.ones(od)

    # qpos block: index q -> obs index q-2 (z, quat, joints)
    obs_of_qpos = lambda q: q - 2                    # noqa: E731
    # quaternion: (x, y, z) -> (-x, y, -z)
    for k, s in zip((1, 2, 3), (-1.0, 1.0, -1.0)):
        sign[obs_of_qpos(3 + k)] = s
    for j in range(m.njnt):
        if j not in jnt_pair or jnt_pair[j] is None:
            continue
        o1 = obs_of_qpos(int(m.jnt_qposadr[j]))
        o2 = obs_of_qpos(int(m.jnt_qposadr[jnt_pair[j]]))
        perm[o1], sign[o1] = o2, jnt_sign[j]
    # qvel block: (vx, vy, vz, wx, wy, wz) then joints
    voff = nq - 2
    for k, s in zip((0, 1, 2, 3, 4, 5), (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)):
        sign[voff + k] = s
    for j in range(m.njnt):
        if j not in jnt_pair or jnt_pair[j] is None:
            continue
        o1 = voff + int(m.jnt_dofadr[j])
        o2 = voff + int(m.jnt_dofadr[jnt_pair[j]])
        perm[o1], sign[o1] = o2, jnt_sign[j]
    # contact forces: per body (fx,fy,fz,tx,ty,tz), body-permuted.
    #
    # Component signs are MEASURED, not derived from a polar/pseudo-vector
    # assumption. Sampling the two branches with real contacts gives, e.g.
    #   A[1]=[1.533, 0.116, -0.254, 0.535, 4.927, 5.462]
    #   B[1]=[-1.533, 0.116, 0.254, 0.535, -4.927, 5.462]
    # so force maps as (-x, +y, -z) and torque as (+x, -y, +z). That is the
    # opposite of the naive world-frame polar/pseudo-vector rule, which is why
    # the oracle exists: the first version of this file assumed the naive rule
    # and produced a defect of exactly 2.0 (a clip saturation) on every sample
    # with contact, while the 27-dimensional state part was already exact at
    # 5.6e-17.
    coff = nq - 2 + nv
    for b in range(nbody):
        b2 = body_pair.get(b)
        if b2 is None:
            continue
        for k in range(6):
            o1, o2 = coff + 6 * b + k, coff + 6 * b2 + k
            perm[o1] = o2
            sign[o1] = (-1.0 if k in (0, 2, 4) else 1.0)

    # --- actions: actuator -> joint -> mirrored joint, torque conjugate to angle
    a_perm, a_sign = np.arange(nu), np.ones(nu)
    trn = np.array(m.actuator_trnid)[:, 0]
    for i in range(nu):
        j = int(trn[i])
        if j not in jnt_pair or jnt_pair[j] is None:
            continue
        j2 = jnt_pair[j]
        cands = [k for k in range(nu) if int(trn[k]) == j2]
        if not cands:
            continue
        i2 = cands[0]
        a_perm[i] = i2
        ratio = float(m.actuator_gear[i, 0]) / float(m.actuator_gear[i2, 0])
        a_sign[i] = jnt_sign[j] * ratio

    return {
        "spec": MirrorSpec(perm, sign, a_perm, a_sign),
        "body_pair": {int(k): (None if v is None else int(v))
                      for k, v in body_pair.items()},
        "jnt_pair": {int(k): (None if v is None else int(v))
                     for k, v in jnt_pair.items()},
        "jnt_sign": {int(k): float(v) for k, v in jnt_sign.items()},
        "n_hinge": int(n_hinge),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=D4RL_ANT)
    ap.add_argument("--eval-samples", type=int, default=24,
                    help="samples for the search (seed 0)")
    ap.add_argument("--verify-samples", type=int, default=32,
                    help="independent samples for verification (seed 1)")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.eval_samples, args.verify_samples, args.sweeps = 4, 4, 1

    import gymnasium as gym

    env = gym.make("Ant-v4", **ENV_KW)
    t0 = time.perf_counter()
    print(f"env=Ant-v4{ENV_KW}  obs={env.observation_space.shape}  "
          f"act={env.action_space.shape}")
    print(f"D4RL data={args.env} (111-dim: 13 qpos + 14 qvel + 84 contact force)")
    obs_e, act_e, _ = load_samples(args.env, args.eval_samples, seed=0)
    obs_v, act_v, _ = load_samples(args.env, args.verify_samples, seed=1)
    defect = build_oracle(env)

    out = {"env": "Ant-v4", "d4rl_data": args.env,
           "obs_dim": int(env.observation_space.shape[0]),
           "n_eval": args.eval_samples, "n_verify": args.verify_samples}

    # --- step 0: control
    ident = identity_spec(env)
    c = defect(ident, obs_e, act_e, return_parts=True)
    print(f"\nSTEP 0 control (identity): obs={c['obs']:.3e} reward={c['reward']:.3e}"
          f"  contacts={c['contacts']}")
    if c["obs"] != 0.0 or c["reward"] != 0.0:
        raise SystemExit("control failed: identity must be exactly 0")
    out["control"] = c

    # --- step 1: structural derivation
    d = derive_spec(env)
    spec = d["spec"]
    moved = int((spec.obs_perm != np.arange(len(spec.obs_perm))).sum())
    print(f"STEP 1 derived spec: moved dims={moved}/{len(spec.obs_perm)}  "
          f"flipped={int((spec.obs_sign == -1).sum())}  "
          f"act flips={int((spec.act_sign == -1).sum())}")
    print(f"        body pairing={d['body_pair']}")
    print(f"        joint pairing={d['jnt_pair']}  signs={d['jnt_sign']}")
    s1 = defect(spec, obs_e, act_e, return_parts=True)
    print(f"        defect: obs={s1['obs']:.6f} reward={s1['reward']:.6f}")
    out["derived"] = {"moved": moved, "defect": s1, **{k: d[k] for k in
                      ("body_pair", "jnt_pair", "jnt_sign")}}

    # --- step 2: bounded coordinate descent to fix derivation errors
    best = max(s1["obs"], s1["reward"])
    history = [best]
    for s in range(args.sweeps):
        improved = False
        for k in range(len(spec.obs_sign)):
            cand = spec.obs_sign.copy()
            cand[k] *= -1.0
            trial = MirrorSpec(spec.obs_perm, cand, spec.act_perm, spec.act_sign)
            v = defect(trial, obs_e, act_e)
            if v < best - 1e-12:
                spec, best, improved = trial, v, True
        for k in range(len(spec.act_sign)):
            cand = spec.act_sign.copy()
            cand[k] *= -1.0
            trial = MirrorSpec(spec.obs_perm, spec.obs_sign, spec.act_perm, cand)
            v = defect(trial, obs_e, act_e)
            if v < best - 1e-12:
                spec, best, improved = trial, v, True
        history.append(best)
        print(f"STEP 2 sweep {s+1}: best defect={best:.6f}")
        if not improved:
            break
    out["descent_history"] = history

    # --- step 3: independent verification on held-out samples
    v = defect(spec, obs_v, act_v, return_parts=True)
    print(f"\nSTEP 3 verify (independent samples): obs={v['obs']:.3e} "
          f"reward={v['reward']:.3e}  contacts={v['contacts']}")
    passed = v["obs"] <= 1e-6 and v["reward"] <= 1e-6
    out["verify"] = v
    out["passed"] = bool(passed)
    out["verdict"] = ("VERIFIED non-trivial mirror symmetry" if passed
                      else "bounded search did NOT find a symmetry")
    print(f"\nVERDICT: {out['verdict']}")
    if not passed:
        print("  (report as 'bounded search did not find', NOT as 'does not exist')")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    p = BENCH_DIR / ("symmetry_ant_smoke.json" if args.smoke
                     else "symmetry_ant.json")
    out["spec"] = {"obs_perm": spec.obs_perm.tolist(),
                   "obs_sign": spec.obs_sign.tolist(),
                   "act_perm": spec.act_perm.tolist(),
                   "act_sign": spec.act_sign.tolist()}
    out["elapsed_min"] = (time.perf_counter() - t0) / 60.0
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p.name}  ({out['elapsed_min']:.1f} min)")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
