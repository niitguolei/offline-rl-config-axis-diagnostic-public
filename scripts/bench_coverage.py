"""Benchmark the coverage (kNN) module on the largest real dataset.

Review finding M1 said the plan's compute figures were unmeasured estimates and
that exact kNN over 2M transitions is infeasible. This script measures the real
cost of the proposed replacement: chunked GPU brute-force with the
||q||^2 + ||r||^2 - 2qr trick and a subsampled reference set.
"""
from __future__ import annotations

import time

import h5py
import torch

DATA = r"D:\workspace\codex\findRLIdea\data\d4rl\halfcheetah_medium_expert-v2.hdf5"


def load_obs(max_n: int | None = None) -> torch.Tensor:
    """Load observations only (not full transition tuples) to keep RAM low."""
    t0 = time.time()
    with h5py.File(DATA, "r") as f:
        n = f["observations"].shape[0]
        if max_n is not None:
            n = min(n, max_n)
        arr = f["observations"][:n]
    print(f"loaded {arr.shape[0]:,} x {arr.shape[1]} obs in {time.time() - t0:.1f}s")
    return torch.from_numpy(arr).float()


def pick_chunk(n_ref: int, budget_bytes: int = 1_200_000_000) -> int:
    """Chunk so that the (chunk x n_ref) distance block stays inside budget.

    Measured: peak VRAM is dominated by that one block, so this is the knob
    that keeps the module inside 7.96 GB.
    """
    return max(256, budget_bytes // (n_ref * 4))


def chunked_knn(queries, refs, k: int = 10, chunk: int | None = None):
    """k nearest neighbours of queries among refs, via chunked GPU matmul."""
    dev = "cuda"
    if chunk is None:
        chunk = pick_chunk(refs.shape[0])
    r = refs.to(dev, non_blocking=True)
    r_sq = (r * r).sum(1)
    out_d = torch.empty(queries.shape[0], k, device=dev)
    out_i = torch.empty(queries.shape[0], k, device=dev, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.time()
    peak_block = 0
    for s in range(0, queries.shape[0], chunk):
        q = queries[s : s + chunk].to(dev, non_blocking=True)
        d2 = (q * q).sum(1, keepdim=True) + r_sq.unsqueeze(0) - 2.0 * (q @ r.T)
        d2.clamp_(min=0)
        peak_block = max(peak_block, d2.numel() * 4)
        vals, idx = torch.topk(d2, k, dim=1, largest=False)
        out_d[s : s + chunk] = vals.sqrt()
        out_i[s : s + chunk] = idx
        del q, d2, vals, idx
    torch.cuda.synchronize()
    return time.time() - t0, out_d, out_i, chunk, peak_block


def main() -> None:
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"total VRAM {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    obs = load_obs()
    print(f"obs tensor {tuple(obs.shape)} = {obs.numel() * 4 / 1e6:.0f} MB\n")
    print(f"{'n_ref':>9} {'chunk':>7} {'n_query':>10} {'time':>9} {'peak_VRAM':>11} {'block_MB':>10}", flush=True)
    for n_ref in (50_000, 100_000, 200_000):
        g = torch.Generator().manual_seed(0)
        refs = obs[torch.randperm(obs.shape[0], generator=g)[:n_ref]]
        chunk = pick_chunk(n_ref)
        torch.cuda.reset_peak_memory_stats()
        try:
            dt, d, idx, _, blk = chunked_knn(obs, refs, k=10, chunk=chunk)
        except torch.OutOfMemoryError:
            print(f"{n_ref:>9,} {chunk:>7} {'ALL':>10} {'OOM':>9}", flush=True)
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"{n_ref:>9,} {chunk:>7} {obs.shape[0]:>10,} {dt:>8.1f}s "
            f"{peak:>10.2f}G {blk / 1e6:>9.0f}M  median_nn={d[:, 0].median().item():.3f}",
            flush=True,
        )
        del d, idx, refs
        torch.cuda.empty_cache()
    print("\nTiming covers ALL queries in the dataset (one full coverage pass).")


if __name__ == "__main__":
    main()
