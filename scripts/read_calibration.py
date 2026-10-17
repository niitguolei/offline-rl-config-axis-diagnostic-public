"""Pretty-print results/benchmarks/calibration.json."""
from __future__ import annotations

import json
import pathlib

p = pathlib.Path(__file__).resolve().parent.parent / "results" / "benchmarks" / "calibration.json"
d = json.loads(p.read_text(encoding="utf-8"))

print(f"n_step={d['n_step']}  seed={d['seed']}  checkpoints={d['checkpoints']}")
print()
head = f"{'step':>11}{'policy_raw':>12}{'q_loss':>14}{'v_loss':>13}   overest b0/b1/b2"
print(head)
print("-" * len(head))
for r in d["records"]:
    ov = "/".join(f"{r['buckets'][b]['overest_mean']:.1f}" for b in ("0", "1", "2"))
    step = f"{r['step']:,}"
    print(f"{step:>11}{r['policy_raw']:>12.1f}{r['q_loss']:>14.3f}"
          f"{r['v_loss']:>13.3f}   {ov}")

print()
print("MC reference (must stay stable -- it validates the measurement chain):")
for r in d["records"]:
    mc = [f"{r['buckets'][b]['mc_mean']:.1f}" for b in ("0", "1", "2")]
    sb = [f"{r['buckets'][b].get('signed_bias_mean', 0):+.1f}" for b in ("0", "1", "2")]
    print(f"  {r['step']:>10,}  mc={'/'.join(mc):<28} signed_bias={'/'.join(sb)}")

print(f"\ntotal elapsed: {d['records'][-1]['elapsed_min']:.0f} min")

