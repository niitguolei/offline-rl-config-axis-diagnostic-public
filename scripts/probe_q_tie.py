"""Why is Q not equivariant after tie_q, when V is?"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT / "vendor" / "IQL-PyTorch", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.value_functions import TwinQ  # noqa: E402

from orl.methods.equivariance import (  # noqa: E402
    apply_mirror_act,
    apply_mirror_obs,
    derive_mirror,
    symmetrize_linear_stack,
)

dev = "cpu"
torch.manual_seed(0)
spec = derive_mirror("walker2d", obs_dim=17, act_dim=6)
sa_perm = np.concatenate([spec.obs_perm, spec.act_perm])
sa_sign = np.concatenate([spec.obs_sign, spec.act_sign])

net = TwinQ(17, 6).to(dev)
for p in net.parameters():
    with torch.no_grad():
        p.add_(torch.randn_like(p) * 0.3)
for q in (net.q1, net.q2):
    symmetrize_linear_stack(q, sa_perm, sa_sign, np.arange(1), np.ones(1))

s = torch.randn(64, 17)
a = torch.randn(64, 6)
f = lambda x, y: net.q1(torch.cat([x, y], -1)).squeeze(-1)  # noqa: E731
lhs = f(apply_mirror_obs(spec, s), apply_mirror_act(spec, a))
rhs = f(s, a)
print("equivariance error after tie :", float((lhs - rhs).abs().max()))

perm_only = f(apply_mirror_obs(spec, s), a)
print("  mirror s only, keep a     :", float((perm_only - rhs).abs().max()))
act_only = f(s, apply_mirror_act(spec, a))
print("  keep s, mirror a only     :", float((act_only - rhs).abs().max()))

lins = [m for m in net.q1.modules() if isinstance(m, torch.nn.Linear)]
print("q1 linear layers:", [tuple(l.weight.shape) for l in lins])

