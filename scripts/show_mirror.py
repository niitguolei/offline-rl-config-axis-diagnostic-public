"""Print the derived mirror spec per domain (diagnostic)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orl.methods.equivariance import derive_mirror  # noqa: E402

DIMS = {"halfcheetah": (17, 6), "walker2d": (17, 6), "hopper": (11, 3)}

for d, (o, a) in DIMS.items():
    try:
        s = derive_mirror(d, obs_dim=o, act_dim=a)
        print(d)
        print("   obs_perm =", s.obs_perm.tolist())
        print("   obs_sign =", s.obs_sign.astype(int).tolist())
        print("   act_perm =", s.act_perm.tolist())
        print("   act_sign =", s.act_sign.astype(int).tolist())
        print("   involutive =", s.is_involutive())
    except Exception as exc:  # noqa: BLE001
        print(d, "ERROR", type(exc).__name__, exc)
    print()

