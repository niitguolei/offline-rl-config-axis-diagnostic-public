"""Single entry point for concatenating per-block permutations/signs.

WHY THIS FILE EXISTS
-------------------
The same off-by-one-length bug bit us TWICE:

  1. In `derive_mirror`, the qvel block's indices are block-relative
     (0..nv-1) but were concatenated straight after the qpos block's
     (0..nq-2). The result was not a permutation at all -- indices 0..7
     appeared twice.
  2. In `tie_q`, `act_perm` (0..act_dim-1) was concatenated after
     `obs_perm` without offsetting by the observation width, so the action
     columns of the tied Q were gathered from the OBSERVATION block. The
     symptom was nasty and silent: the tied critic became
     action-INDEPENDENT. Equivariance error was identical whether or not the
     action was mirrored, which is what gave it away.

Both fixes were one missing `+ offset`. Comments did not stop it happening
twice, so the arithmetic now lives in exactly one place.
"""

from __future__ import annotations

import numpy as np


def concat_blocks(*blocks: np.ndarray) -> np.ndarray:
    """Concatenate per-block permutations, offsetting each by prior widths.

    >>> import numpy as np
    >>> concat_blocks(np.array([1, 0]), np.array([0, 1, 1, 0])).tolist()
    [1, 0, 2, 3, 5, 4]
    """
    out: list[int] = []
    offset = 0
    for b in blocks:
        arr = np.asarray(b, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() >= arr.size):
            raise ValueError(
                f"block of size {arr.size} has indices outside [0, {arr.size}); "
 "it must be block-relative, not global")
        out.extend((arr + offset).tolist())
        offset += arr.size
    return np.asarray(out, dtype=np.int64)


def concat_signs(*blocks: np.ndarray) -> np.ndarray:
    """Concatenate per-block sign vectors (elementwise, so no offset needed)."""
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks])

