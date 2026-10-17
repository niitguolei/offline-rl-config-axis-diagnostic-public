"""Does save->load of the critic preserve the trained Q exactly?

Motivation: the 150k run printed q_loss ~9.6, but the saved checkpoint scores
MSE(Q, r + 0.99*Vnext) ~ 182,144. Those cannot both be true, so either the
save/load is lossy or the loss is being computed on something else.

    python scripts/roundtrip_test.py
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import torch

from orl.algos.iql import IQL, IQLConfig
from orl.data.d4rl_hdf5 import D4RLDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = D4RLDataset("halfcheetah_medium_replay-v2", mode="ram", normalize=True)
KEYS = ("observations", "actions", "rewards", "next_observations", "terminals")
o, a, r, no, te = [ds.get(k) for k in KEYS]
rng = np.random.default_rng(0)

cfg = IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, gamma=0.99, device=dev)
ag = IQL(cfg)

last = None
for i in range(500):
    j = rng.integers(0, ds.n, 256)
    last = ag.update({"observations": o[j], "actions": a[j], "rewards": r[j],
                      "next_observations": no[j], "terminals": te[j]})
print(f"in-memory q_loss after 500 steps = {last['q_loss']:.4f}")

T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
j = rng.integers(0, ds.n, 4096)
with torch.no_grad():
    Q_in = ag.q1(ag._sa(T(o[j]), T(a[j]))).squeeze(-1).clone()

with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "c.pt"
    torch.save({"q1": ag.q1.state_dict(), "v": ag.v.state_dict(),
                "cfg": vars(cfg)}, p)
    ck = torch.load(p, weights_only=False)
    print(f"saved cfg: {sorted(ck['cfg'])}")
    ag2 = IQL(IQLConfig(obs_dim=ck["cfg"]["obs_dim"],
                        act_dim=ck["cfg"]["act_dim"],
                        gamma=0.99,
                        layer_norm=ck["cfg"].get("layer_norm", True),
                        device=dev))
    ag2.q1.load_state_dict(ck["q1"])
    ag2.v.load_state_dict(ck["v"])
    with torch.no_grad():
        Q_out = ag2.q1(ag2._sa(T(o[j]), T(a[j]))).squeeze(-1).clone()

d = (Q_in - Q_out).abs().max().item()
print(f"max |Q_in_memory - Q_reloaded| = {d:.6e}")
print("PASS: save/load lossless" if d < 1e-4 else "FAIL: save/load lossy")

# --- decisive: is the reported q_loss consistent with an independent recompute?
print()
ag3 = IQL(IQLConfig(obs_dim=ds.obs_dim, act_dim=ds.act_dim, gamma=0.99,
                    device=dev))
rep = None
for i in range(2000):
    j = rng.integers(0, ds.n, 256)
    rep = ag3.update({"observations": o[j], "actions": a[j], "rewards": r[j],
                      "next_observations": no[j], "terminals": te[j]})
jb = rng.integers(0, ds.n, 4096)
ob, ab = T(o[jb]), T(a[jb])
with torch.no_grad():
    tgt = T(np.asarray(r[jb], dtype=np.float32).reshape(-1)) \
        + 0.99 * ag3.v(T(no[jb])).squeeze(-1)
    qp = ag3.q1(ag3._sa(ob, ab)).squeeze(-1)
    indep = torch.nn.functional.mse_loss(qp, tgt).item()
print(f"reported q_loss (last minibatch)   = {rep['q_loss']:.4f}")
print(f"independent MSE(Q, r+gV) on fresh batch = {indep:.4f}")
print(f"Q mean = {qp.mean():.2f}   target mean = {tgt.mean():.2f}   V mean = {ag3.v(ob).mean():.2f}")

