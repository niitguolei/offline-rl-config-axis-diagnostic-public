"""Check dataset observation dims against the gymnasium evaluation envs.

Offline RL training reads transitions straight from HDF5, but reporting a
normalised score needs the simulator. If the two disagree on observation
dimensionality, the learned policy cannot be fed to the env at all.
"""
import glob
import os

import gymnasium as gym
import h5py

DATA_DIR = r"D:\workspace\codex\findRLIdea\data\d4rl"

ENVS = {
    "halfcheetah": "HalfCheetah-v5",
    "walker2d": "Walker2d-v5",
    "hopper": "Hopper-v5",
    "ant": "Ant-v5",
}


def main() -> None:
    dims = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5"))):
        with h5py.File(p, "r") as f:
            dims[os.path.basename(p)] = f["observations"].shape[1]

    print(f"{'domain':<12}{'dataset obs':>14}{'gymnasium obs':>16}   compatible")
    print("-" * 60)
    bad = []
    for dom, eid in ENVS.items():
        env = gym.make(eid)
        g = env.observation_space.shape[0]
        env.close()
        ds = sorted({v for k, v in dims.items() if k.startswith(dom + "_")})
        ok = ds == [g]
        shown = str(ds[0]) if len(ds) == 1 else str(ds)
        print(f"{dom:<12}{shown:>14}{g:>16}   {'YES' if ok else 'NO  <-- mismatch'}")
        if not ok:
            bad.append((dom, ds, g))

    print()
    if not bad:
        print("All domains dimension-compatible with their gymnasium env.")
        return

    print("MISMATCHED domains:")
    for dom, ds, g in bad:
        print(f"  {dom}: dataset {ds} vs env {g}")
    print()
    print("D4RL ant observations carry contact forces (cfrc_ext) for one more")
    print("body than gymnasium Ant-v5 reproduces. Options:")
    print("  (a) drop ant from the first experimental round  <-- recommended")
    print("  (b) slice dataset obs to the env's N dims, then verify the policy")
    print("      still scores sensibly (contact forces rarely matter much)")
    print("  (c) reconstruct a matching ant XML/env")
    print()
    print("ant is also the largest domain on disk (~1.9 GB) and the least")
    print("relevant to the symmetry direction, so (a) costs little.")


if __name__ == "__main__":
    main()

