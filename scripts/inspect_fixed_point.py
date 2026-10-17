"""Why is the trained critic at -422 when the Bellman fixed point is +309?"""
from __future__ import annotations

import numpy as np
import torch

from orl.algos.iql import IQL, IQLConfig
from orl.data.d4rl_hdf5 import D4RLDataset

ck = torch.load("results/benchmarks/critic_scale_150000.pt", weights_only=False)
c = ck["cfg"]
dev = "cuda"
ag = IQL(IQLConfig(obs_dim=c["obs_dim"], act_dim=c["act_dim"], gamma=0.99,
                   layer_norm=True, device=dev))
ag.q1.load_state_dict(ck["q1"])
ag.pi.load_state_dict(ck["pi"])

ds = D4RLDataset("halfcheetah_medium_replay-v2", mode="ram", normalize=True)
rng = np.random.default_rng(0)
j = np.sort(rng.choice(ds.n, 20000, replace=False))
o = ds.get("observations")[j]
a = ds.get("actions")[j]
r = ds.get("rewards")[j]
no = ds.get("next_observations")[j]

T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
with torch.no_grad():
    Q = ag.q1(ag._sa(T(o), T(a))).squeeze(-1)
    V = ag.v(T(o)).squeeze(-1)
    Vn = ag.v(T(no)).squeeze(-1)

print(f"V(s)        mean={V.mean():10.2f}  std={V.std():9.2f}")
print(f"V(s_next)   mean={Vn.mean():10.2f}  std={Vn.std():9.2f}")
print(f"Q(s,a)      mean={Q.mean():10.2f}  std={Q.std():9.2f}")
print(f"reward      mean={r.mean():10.2f}  std={r.std():9.2f}")
print()
resid = Q.mean() - (r.mean() + 0.99 * Vn.mean())
print(f"Bellman residual  mean(Q) - mean(r + 0.99*Vnext) = {resid:+.2f}")
print(f"fixed point if converged  r/(1-gamma)           = {r.mean() / 0.01:+.2f}")
print(f"what a one-step-from-zero iteration gives        = {r.mean():+.2f}")
print()
print(f"state dependence  std(V)/|mean(V)| = {V.std() / abs(V.mean()):.4f}")
print(f"                  std(Q)/|mean(Q)| = {Q.std() / abs(Q.mean()):.4f}")
print()
last = ag.q1[-1]
print(f"final Linear bias = {last.bias.item():.2f}")
print(f"final Linear weight |mean| = {last.weight.abs().mean().item():.5f}")

# Reconcile: the run printed q_loss ~ 9.6, but Q and its target differ by ~430.
rec = np.asarray(ds.get("rewards")[:20000], dtype=np.float32).reshape(-1)
with torch.no_grad():
    tgt = T(rec) + 0.99 * Vn
    mse = torch.nn.functional.mse_loss(Q, tgt).item()
print()
print(f"recomputed MSE(Q, r + 0.99*Vnext) on the same batch = {mse:.2f}")
print(f"training-time q_loss printed at 150k                 = 9.59")
