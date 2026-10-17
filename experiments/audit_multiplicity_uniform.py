"""Full-family correction on uniform grids only.

The historical audit mixed the five-level extension (seeds 0-2) with the
three-level top-up (seeds 3-4). That averages different selection problems.
This script restricts every seed to the same three-level grid before computing
the 22-item Bonferroni sensitivity table.
"""

from __future__ import annotations

import pathlib
import sys
import json

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from audit_multiplicity import load, paired_test  # noqa: E402


def build_family(iql: list[dict], cql: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for label, runs, knob in (("IQL", iql, "tau"), ("CQL", cql, "min_q_weight")):
        for env in sorted({r["env"] for r in runs}):
            sub = [r for r in runs if r["env"] == env]
            short = env.split("_")[0]
            for key in ("critic_mean", "fqe"):
                result = paired_test(sub, key)
                if result:
                    out.append((f"{label} pooled {short} {key}", result))
            for axis, within, tag in (
                (knob, "lr", f"along {knob}"),
                ("lr", knob, "along lr"),
            ):
                result = paired_test(sub, "critic_mean", axis=axis, within=within)
                if result:
                    out.append((f"{label} {tag} {short}", result))
    return out


def main() -> int:
    iql = [
        r for r in load("e2b_runs*.jsonl")
        if float(r.get("tau", -1)) in (0.5, 0.7, 0.9)
    ]
    cql = [
        r for r in load("e2b_cql_runs*.jsonl")
        if float(r.get("min_q_weight", -1)) in (5.0, 10.0, 20.0)
    ]
    tests = sorted(build_family(iql, cql), key=lambda kv: kv[1]["p"])
    m = len(tests)
    survivors = []
    rows = []
    print(f"uniform-grid family size = {m}")
    print(f"{'test':<44} {'beats':>8} {'p':>9} {'p*N':>9}")
    for name, row in tests:
        bonf = min(1.0, row["p"] * m)
        if bonf < 0.05:
            survivors.append(name)
        rows.append({
            "test": name,
            "beats": row["beats"],
            "raw_p": row["p"],
            "p_times_22": bonf,
            "survives": bool(bonf < 0.05),
        })
        print(f"{name:<44} {row['beats']:+8.1%} {row['p']:9.4f} {bonf:9.4f}")
    print(f"\nsurviving Bonferroni: {len(survivors)}/{m}")
    for name in survivors:
        print(f"  {name}")
    out = ROOT / "results" / "benchmarks" / "multiplicity_uniform.json"
    out.write_text(json.dumps({
        "family_size": m,
        "survivors": survivors,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
