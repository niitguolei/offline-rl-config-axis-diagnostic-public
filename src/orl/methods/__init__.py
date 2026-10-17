"""Mechanism modules (M1, M2) ported onto the vendored reference baseline.

These are the *new* contributions. The baseline stays untouched at
vendor/IQL-PyTorch/ so provenance stays auditable -- each mechanism is a
subclass that overrides exactly one quantity.

  M2 (support_aware.py) : support-aware per-sample backup length.
  M1                    : symmetry-equivariant pessimism (not yet ported).
"""

