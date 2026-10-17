"""Print joint / actuator names and the involution check step by step."""
from __future__ import annotations

import pathlib
import sys

import gymnasium as gym
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orl.envs.obs_layout import ENV_IDS  # noqa: E402
from orl.methods.equivariance import derive_mirror  # noqa: E402

DIMS = {"halfcheetah": (17, 6), "walker2d": (17, 6), "hopper": (11, 3)}

for dom, (o, a) in DIMS.items():
    env = gym.make(ENV_IDS[dom])
    m = env.unwrapped.model
    jn = [m.joint(j).name for j in range(m.njnt)]
    an = [m.actuator(a).name for a in range(m.nu)]
    env.close()
    print(dom)
    print("   joints    =", jn)
    print("   actuators =", an)
    s = derive_mirror(dom, obs_dim=o, act_dim=a)
    print("   obs_perm  =", s.obs_perm.tolist())
    print("   act_perm  =", s.act_perm.tolist())
    p = np.arange(len(s.obs_perm))
    q = np.arange(len(s.act_perm))
    ok_o = np.array_equal(s.obs_perm[s.obs_perm], p)
    ok_a = np.array_equal(s.act_perm[s.act_perm], q)
    print("   obs involution:", ok_o, " act involution:", ok_a)
    print("   is_involutive():", s.is_involutive())
    print()

