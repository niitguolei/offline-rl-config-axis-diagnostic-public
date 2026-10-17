"""Inspect a directory of D4RL HDF5 files: dtype, shapes, keys, integrity."""
import glob
import os
import sys

import h5py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DIRS = [r"D:\workspace\codex\findRLIdea\data\d4rl", os.path.expanduser("~/.d4rl/datasets")]

for d in sys.argv[1:] or DEFAULT_DIRS:
    print(f"\n=== {d} ===")
    files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
    if not files:
        print("  (no .hdf5 files)")
        continue
    for p in files:
        name = os.path.basename(p)
        mb = os.path.getsize(p) / 1e6
        try:
            with h5py.File(p, "r") as f:
                keys = sorted(f.keys())
                o = f["observations"]
                a = f["actions"]
                print(f"  {name:<40} {mb:7.1f}MB n={o.shape[0]:>9,} {o.dtype} obs{o.shape[1:]} act{a.shape[1:]}")
                print(f"  {'':<40} keys={keys}")
                if "rewards" in f:
                    print(f"  {'':<40} rewards shape={f['rewards'].shape} dtype={f['rewards'].dtype}")
                for k in ("terminals", "timeouts"):
                    if k in f:
                        arr = f[k]
                        print(f"  {'':<40} {k}: n={arr.shape[0]:,} sum={arr[:].sum()}")
        except Exception as exc:
            print(f"  {name:<40} {mb:7.1f}MB ERROR {type(exc).__name__}: {exc}")

