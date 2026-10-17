"""D4RL normalised-score reference values.

Source: `d4rl/infos.py` REF_MIN_SCORE / REF_MAX_SCORE, extracted from the
vendored copy under D:/workspace/codex/findRLIdea/tmp/altriaex_d4rl.

IMPORTANT CAVEAT (read before using these numbers):
These references were produced under the mujoco-py era environments. We
evaluate with gymnasium v5, whose `mujoco` bindings differ slightly. Absolute
returns are therefore NOT strictly comparable across the two stacks.
The paper's primary metric is raw return; normalised score is reported
alongside for cross-paper reference only, and the env version is always
declared in the write-up.

antmaze is intentionally absent: ant and antmaze are permanently excluded
(111 vs 105 observation dims; symmetry premise fails on maze walls).
"""

from __future__ import annotations

REF_MIN_SCORE = {
    "halfcheetah": -280.178953,
    "walker2d": 1.629008,
    "hopper": -20.272305,
}

REF_MAX_SCORE = {
    "halfcheetah": 12135.0,
    "walker2d": 4592.3,
    "hopper": 3234.3,
}


def normalized_score(domain: str, raw_return: float) -> float:
    """D4RL normalised score. Cross-simulator-version: reference only."""
    lo = REF_MIN_SCORE[domain]
    hi = REF_MAX_SCORE[domain]
    return 100.0 * (raw_return - lo) / (hi - lo)

