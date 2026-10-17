"""I3b — per-sample coverage score via GPU chunked kNN.

Measured on the real 2M x 17 dataset (2026-10-04):

    n_ref      chunk   full-scan time   peak VRAM
    50,000     6000        34.3 s         3.61 GB
    100,000    3000        66.0 s         3.61 GB
    500,000 (fixed chunk 4096)  -> OOM, tried to allocate 7.63 GiB

Two hard-won facts are encoded here:

1. Peak memory is set by the single `chunk x n_ref` distance block, NOT by the
   reference set. Hence `pick_chunk` sizes the block to a fixed budget.
2. Streaming mode is ~10,935x slower per step than RAM mode, so training must
   load into RAM first. This module therefore takes RAM-resident arrays.

Self-match caveat: when a query point is also in the reference set its own
distance is 0. We therefore request k+1 neighbours and drop the closest,
so the reported distance is to the k-th *other* point.

Precision caveat (measured 2026-10-04): the matmul identity suffers float32
cancellation. With coordinates up to ~20.5, an exact self-match returns
~2.8e-3 instead of 0. Real coverage scores are O(1) (median 0.83, p99 2.80 on
hopper_medium_replay), so this error is ~0.3% of the signal and harmless for
ranking or bucketing -- but do not use these values to reason about
sub-millimetre geometric differences. Use float64 inputs if that is needed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from ..data.d4rl_hdf5 import D4RLDataset
from ..paths import COVERAGE_CACHE

DEFAULT_N_REF = 100_000
DEFAULT_K = 10
BLOCK_BUDGET_BYTES = 1_200_000_000


def pick_chunk(n_ref: int, budget_bytes: int = BLOCK_BUDGET_BYTES) -> int:
    """Chunk so the `chunk x n_ref` distance block stays inside budget."""
    return max(256, budget_bytes // max(1, n_ref * 4))


@torch.no_grad()
def knn_distances(
    queries: torch.Tensor,
    refs: torch.Tensor,
    k: int = DEFAULT_K,
    chunk: int | None = None,
    drop_self: bool = True,
) -> torch.Tensor:
    """Distance from each query to its k-th nearest reference point."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = refs.to(dev, non_blocking=True)
    r_sq = (r * r).sum(1)
    kk = k + 1 if drop_self else k
    out = torch.empty(queries.shape[0], kk, device=dev)
    c = chunk or pick_chunk(int(r.shape[0]))
    for s in range(0, queries.shape[0], c):
        q = queries[s : s + c].to(dev, non_blocking=True)
        # squared distance via matmul, avoiding a (c, n_ref, d) tensor
        d2 = (q * q).sum(1, keepdim=True) + r_sq.unsqueeze(0) - 2.0 * (q @ r.T)
        d2.clamp_(min=0)
        vals, _ = torch.topk(d2, kk, dim=1, largest=False)
        out[s : s + c] = vals
        del q, d2
    out = out.sqrt()
    return out[:, 1:] if drop_self else out


def coverage_scores(
    obs: np.ndarray,
    n_ref: int = DEFAULT_N_REF,
    k: int = DEFAULT_K,
    seed: int = 0,
    device: str = "cuda",
) -> np.ndarray:
    """Coverage score per transition: distance to the k-th nearest reference."""
    q = torch.as_tensor(obs, dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(q.shape[0], generator=g)[:n_ref]
    d = knn_distances(q, q[idx], k=k, drop_self=True)
    return d[:, -1].cpu().numpy()


def cache_path(env_id: str, n_ref: int, k: int) -> pathlib.Path:
    return COVERAGE_CACHE / f"{env_id}_nref{n_ref}_k{k}.npy"


def get_coverage(
    env_id: str,
    n_ref: int = DEFAULT_N_REF,
    k: int = DEFAULT_K,
    force: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Coverage scores for a dataset, computed once and cached to disk.

    The cache is what makes this affordable: a full scan costs ~66 s per
    dataset at n_ref=100k, and every later training run reuses it.
    """
    p = cache_path(env_id, n_ref, k)
    if p.exists() and not force:
        return np.load(p)
    with D4RLDataset(env_id, mode="ram") as ds:
        scores = coverage_scores(ds.get("observations"), n_ref=n_ref, k=k, seed=seed)
    COVERAGE_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(p, scores.astype(np.float32))
    return scores.astype(np.float32)
