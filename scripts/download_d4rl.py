"""Download D4RL locomotion datasets directly to ~/.d4rl/datasets/.

Why this exists: the vendored `d4rl` package cannot create environments here
(no `gym`, no `mujoco-py`, and mujoco-py will not build on Python 3.12).
But offline RL training only needs the transition tuples, which are plain HDF5.
So we fetch the same files d4rl would fetch, with the same local filenames
d4rl uses (the URL basename), and load them via h5py.

All URLs below point at the `gym_mujoco_v2/` tree, which serves the canonical
**-v2** releases: float32, with `next_observations`, and the correct transition
counts (e.g. halfcheetah-medium-replay-v2 = 202,000 transitions).

Do NOT use the older `gym_mujoco/` tree. Its files are a different, earlier
vintage: `halfcheetah_mixed.hdf5` there has only 101,000 transitions and no
`next_observations`, so results are not comparable with published -v2 numbers.

Usage:
    python download_d4rl.py
    python download_d4rl.py --tier medium
    python download_d4rl.py --domains halfcheetah walker2d
    python download_d4rl.py --include-antmaze
    python download_d4rl.py --list
    python download_d4rl.py --verify-only
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import urllib.parse
import urllib.request

BASE = "http://rail.eecs.berkeley.edu/datasets/offline_rl"
MUJOCO = f"{BASE}/gym_mujoco_v2"

# Where datasets already live on this machine.
DEFAULT_DEST = r"D:\workspace\codex\findRLIdea\data\d4rl"

LOCOMOTION = {
    "halfcheetah": [
        ("halfcheetah-random-v2", f"{MUJOCO}/halfcheetah_random-v2.hdf5"),
        ("halfcheetah-medium-v2", f"{MUJOCO}/halfcheetah_medium-v2.hdf5"),
        ("halfcheetah-medium-replay-v2", f"{MUJOCO}/halfcheetah_medium_replay-v2.hdf5"),
        ("halfcheetah-expert-v2", f"{MUJOCO}/halfcheetah_expert-v2.hdf5"),
        ("halfcheetah-medium-expert-v2", f"{MUJOCO}/halfcheetah_medium_expert-v2.hdf5"),
    ],
    "walker2d": [
        ("walker2d-random-v2", f"{MUJOCO}/walker2d_random-v2.hdf5"),
        ("walker2d-medium-v2", f"{MUJOCO}/walker2d_medium-v2.hdf5"),
        ("walker2d-medium-replay-v2", f"{MUJOCO}/walker2d_medium_replay-v2.hdf5"),
        ("walker2d-expert-v2", f"{MUJOCO}/walker2d_expert-v2.hdf5"),
        ("walker2d-medium-expert-v2", f"{MUJOCO}/walker2d_medium_expert-v2.hdf5"),
    ],
    "hopper": [
        ("hopper-random-v2", f"{MUJOCO}/hopper_random-v2.hdf5"),
        ("hopper-medium-v2", f"{MUJOCO}/hopper_medium-v2.hdf5"),
        ("hopper-medium-replay-v2", f"{MUJOCO}/hopper_medium_replay-v2.hdf5"),
        ("hopper-expert-v2", f"{MUJOCO}/hopper_expert-v2.hdf5"),
        ("hopper-medium-expert-v2", f"{MUJOCO}/hopper_medium_expert-v2.hdf5"),
    ],
    "ant": [
        ("ant-random-v2", f"{MUJOCO}/ant_random-v2.hdf5"),
        ("ant-medium-v2", f"{MUJOCO}/ant_medium-v2.hdf5"),
        ("ant-medium-replay-v2", f"{MUJOCO}/ant_medium_replay-v2.hdf5"),
        ("ant-expert-v2", f"{MUJOCO}/ant_expert-v2.hdf5"),
        ("ant-medium-expert-v2", f"{MUJOCO}/ant_medium_expert-v2.hdf5"),
    ],
}

# NOTE: these are the filenames d4rl/infos.py lists for antmaze -v2, but the
# verify() next_observations check has NOT been confirmed against them, and
# antmaze is large plus irrelevant to the symmetry work (maze walls are
# asymmetric). Deferred on purpose.
ANTMAZE = [
    ("antmaze-umaze-v2", f"{BASE}/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5"),
    ("antmaze-umaze-diverse-v2", f"{BASE}/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"),
    ("antmaze-medium-play-v2", f"{BASE}/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5"),
    ("antmaze-medium-diverse-v2", f"{BASE}/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"),
    ("antmaze-large-play-v2", f"{BASE}/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5"),
    ("antmaze-large-diverse-v2", f"{BASE}/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"),
]

TIERS = ["random", "medium", "medium-replay", "expert", "medium-expert"]

# d4rl/infos.py REF_MIN_SCORE / REF_MAX_SCORE, needed for normalised score.
REF_SCORES = {
    "halfcheetah": (-280.178953, 12135.0),
    "walker2d": (1.629008, 4592.3),
    "hopper": (-20.272305, 3234.3),
    "ant": (-325.6, 3879.7),
}

UA = {"User-Agent": "Mozilla/5.0 (d4rl-offline-rl-setup)"}


def verify(path: pathlib.Path) -> tuple[bool, str]:
    """Open with h5py and report transition count. Does not load arrays into RAM."""
    try:
        import h5py
    except ImportError:
        return False, "h5py missing"
    try:
        with h5py.File(path, "r") as f:
            if "observations" not in f or "actions" not in f:
                return False, f"unexpected keys {list(f.keys())[:6]}"
            n = f["observations"].shape[0]
            obs = f["observations"].shape[1:]
            act = f["actions"].shape[1:]
            # Vintage check: canonical -v2 files carry next_observations.
            # The older gym_mujoco/ tree does not, and has wrong transition
            # counts, so it must not be mixed in.
            if "next_observations" not in f:
                return False, (f"{n:>9,} transitions obs{obs} act{act} "
                              f"WRONG VINTAGE: no next_observations")
            return True, f"{n:>9,} transitions obs{obs} act{act} next_obs OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def download(url: str, dest: pathlib.Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB  {100 * done / total:5.1f}%",
                          end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except Exception as exc:
        print(f"\n    FAILED: {type(exc).__name__}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--tier", choices=TIERS, help="only this quality tier")
    ap.add_argument("--domains", nargs="*", choices=list(LOCOMOTION))
    ap.add_argument("--include-antmaze", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    def tier_of(env_id: str, domain: str) -> str:
        """'halfcheetah-medium-replay-v2' + 'halfcheetah' -> 'medium-replay'."""
        stem = env_id[: -len("-v2")] if env_id.endswith("-v2") else env_id
        return stem[len(domain) + 1:]

    jobs: list[tuple[str, str]] = []
    for dom, entries in LOCOMOTION.items():
        if args.domains and dom not in args.domains:
            continue
        for env_id, url in entries:
            if args.tier and tier_of(env_id, dom) != args.tier:
                continue
            jobs.append((env_id, url))
    if args.include_antmaze:
        jobs.extend(ANTMAZE)

    if args.list:
        for env_id, url in jobs:
            print(f"{env_id:<30} {url}")
        print(f"\n{len(jobs)} files -> {args.dest}")
        return 0

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"destination: {dest}\n")

    ok = fail = 0
    total_bytes = 0
    for i, (env_id, url) in enumerate(jobs, 1):
        # Local filename is the URL basename, e.g. halfcheetah_medium_replay-v2.hdf5.
        # Note the underscore between domain and tier (d4rl's own env id uses a
        # hyphen there, but the files on disk do not).
        target = dest / os.path.basename(urllib.parse.urlparse(url).path)
        print(f"[{i}/{len(jobs)}] {env_id}")
        print(f"    {url}")
        if not args.verify_only and not (target.exists() and target.stat().st_size > 0):
            print("    downloading ...")
            if not download(url, target):
                fail += 1
                continue
        elif target.exists():
            print(f"    already on disk ({target.stat().st_size / 1e6:.1f} MB)")
        if target.exists():
            total_bytes += target.stat().st_size
            good, msg = verify(target)
            print(f"    {'OK ' if good else 'BAD'} {msg}")
            ok += int(good)
            fail += int(not good)
        else:
            print("    BAD missing on disk")
            fail += 1

    print("\n" + "=" * 62)
    print(f"verified OK : {ok}")
    print(f"failed      : {fail}")
    print(f"on disk     : {total_bytes / 1e6:.1f} MB in {dest}")
    print("\nnormalised = 100 * (return - REF_MIN) / (REF_MAX - REF_MIN)")
    for k, (lo, hi) in REF_SCORES.items():
        print(f"  {k:<12} min={lo:<12} max={hi}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
