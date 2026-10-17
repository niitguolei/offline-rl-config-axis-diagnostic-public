"""Side-by-side divergence comparison: 1-step vs n-step+clip vs the original."""
from __future__ import annotations

import json
import pathlib

B = pathlib.Path(__file__).resolve().parent.parent / "results" / "benchmarks"


def load(name: str):
    p = B / name
    return json.loads(p.read_text(encoding="utf-8"))["records"] if p.exists() else None


cal = load("calibration.json")
one = load("diag_divergence_onestep.json")
clip = load("diag_divergence_nstep_clip.json")

rows = {}
for tag, recs in (("n5_raw", cal), ("1step", one), ("n5_clip", clip)):
    if not recs:
        continue
    for r in recs:
        rows.setdefault(r["step"], {})[tag] = r

print("q_loss")
print(f"{'step':>11}" + "".join(f"{t:>18}" for t in ("n5_raw", "1step", "n5_clip")))
print("-" * 65)
for s in sorted(rows):
    line = f"{s:>11,}"
    for t in ("n5_raw", "1step", "n5_clip"):
        v = rows[s].get(t)
        line += f"{v['q_loss']:>18.3f}" if v else f"{'-':>18}"
    print(line)

print()
print("signed bias per bucket  (0=low coverage, 1=mid, 2=high)")
print(f"{'step':>11}" + "".join(f"{t:>26}" for t in ("n5_raw", "1step", "n5_clip")))
print("-" * 89)
for s in sorted(rows):
    line = f"{s:>11,}"
    for t in ("n5_raw", "1step", "n5_clip"):
        r = rows[s].get(t)
        if not r or "buckets" not in r:
            line += f"{'-':>26}"
            continue
        sb = "/".join(f"{r['buckets'][b]['signed_bias_mean']:+8.1f}" for b in ("0", "1", "2"))
        line += f"{sb:>26}"
    print(line)

print()
print("MC reference per bucket (measurement chain must be stable)")
print(f"{'step':>11}" + "".join(f"{t:>26}" for t in ("n5_raw", "1step", "n5_clip")))
print("-" * 89)
for s in sorted(rows):
    line = f"{s:>11,}"
    for t in ("n5_raw", "1step", "n5_clip"):
        r = rows[s].get(t)
        if not r or "buckets" not in r:
            line += f"{'-':>26}"
            continue
        mc = "/".join(f"{r['buckets'][b]['mc_mean']:8.1f}" for b in ("0", "1", "2"))
        line += f"{mc:>26}"
    print(line)

