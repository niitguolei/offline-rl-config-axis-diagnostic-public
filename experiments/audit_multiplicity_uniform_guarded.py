"""Uniform-grid full-family table with the 20x divergence gate applied.

This is the table used in the manuscript appendix. It keeps the uniform
three-level grid, excludes runs marked `diverged`, and reports the number of
seeds and the number of excluded runs for each cell.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from audit_multiplicity import load, paired_test  # noqa: E402


def build_family(iql: list[dict], cql: list[dict]) -> list[dict]:
    rows = []
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            clean = [r for r in sub if not bool(r.get("diverged", False))]
            excluded = len(sub) - len(clean)
            short = env.split("_")[0]
            for key in ("critic_mean", "fqe"):
                result = paired_test(clean, key)
                if result:
                    rows.append({
                        "test": f"{label} pooled {short} {key}",
                        **result,
                        "excluded": excluded,
                    })
            for axis, within, tag in (
                (knob, "lr", f"along {knob}"),
                ("lr", knob, "along lr"),
            ):
                result = paired_test(clean, "critic_mean", axis=axis, within=within)
                if result:
                    rows.append({
                        "test": f"{label} {tag} {short}",
                        **result,
                        "excluded": excluded,
                    })
    return rows


def main() -> int:
    iql_uniform = [
        r for r in load("e2b_runs*.jsonl")
        if float(r.get("tau", -1)) in (0.5, 0.7, 0.9)
    ]
    cql_uniform = [
        r for r in load("e2b_cql_runs*.jsonl")
        if float(r.get("min_q_weight", -1)) in (5.0, 10.0, 20.0)
    ]
    rows = sorted(build_family(iql_uniform, cql_uniform), key=lambda r: r["p"])
    m = len(rows)
    survivors = []
    print(f"guarded uniform-grid family size = {m}")
    print(f"{'test':<44} {'n':>2} {'excl':>4} {'beats':>8} {'p':>9} {'p*N':>9}")
    for row in rows:
        bonf = min(1.0, row["p"] * m)
        row["p_times_family"] = bonf
        row["survives"] = bool(bonf < 0.05)
        if row["survives"]:
            survivors.append(row["test"])
        print(
            f"{row['test']:<44} {row['n']:>2} {row['excluded']:>4} "
            f"{row['beats']:+8.1%} {row['p']:9.4f} {bonf:9.4f}"
        )
    print(f"\nsurviving Bonferroni: {len(survivors)}/{m}")
    for name in survivors:
        print(f"  {name}")
    out = ROOT / "results" / "benchmarks" / "multiplicity_uniform_guarded.json"
    out.write_text(json.dumps({
        "family_size": m,
        "gate": "20x critic/FQE divergence flag",
        "survivors": survivors,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
