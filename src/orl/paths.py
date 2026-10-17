"""Canonical filesystem locations, so scripts never hard-code paths."""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# D4RL locomotion lives OUTSIDE the repo (123 MB of PDFs and 5 GB of HDF5
# must not enter version control). Override with ORL_DATA_DIR if it moves.
DATA_DIR = pathlib.Path(
    __import__("os").environ.get(
        "ORL_DATA_DIR", r"D:\workspace\codex\findRLIdea\data\d4rl"
    )
)

RESULTS_DIR = REPO_ROOT / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"
COVERAGE_CACHE = RESULTS_DIR / "coverage_cache"

VALID_DOMAINS = ("halfcheetah", "walker2d", "hopper")
VALID_TIERS = ("random", "medium", "medium-replay", "expert", "medium-expert")


def dataset_path(env_id: str) -> pathlib.Path:
    """`halfcheetah_medium_replay-v2` -> the .hdf5 path.

    Note the naming asymmetry that bit us before: the FILE uses an underscore
    between domain and tier (`halfcheetah_medium_replay-v2`) while d4rl's env
    id uses a hyphen (`halfcheetah-medium-replay-v2`). Build filenames from the
    file convention, never from the env id.
    """
    return DATA_DIR / f"{env_id}.hdf5"


def parse_env_id(env_id: str) -> tuple[str, str]:
    """`walker2d_medium_replay-v2` -> ('walker2d', 'medium-replay').

    Note the normalisation: the FILE name uses `medium_replay` but the tier
    label used everywhere in the plan and in run records is `medium-replay`.
    An earlier version returned the underscored form, which silently broke
    filtering records by tier (caught by tests/test_data.py).
    """
    stem = env_id[:-3] if env_id.endswith("-v2") else env_id
    for dom in VALID_DOMAINS:
        if stem.startswith(dom + "_"):
            return dom, stem[len(dom) + 1 :].replace("_", "-")
    return "unknown", "unknown"


def all_env_ids() -> list[str]:
    """The 12 cells of the paper-1 main table (3 domains x 4 tiers)."""
    return [
        f"{dom}_{tier.replace('-', '_')}-v2"
        for dom in VALID_DOMAINS
        for tier in VALID_TIERS
        if tier != "random"  # main table is medium / medium-replay / expert / medium-expert
    ]
