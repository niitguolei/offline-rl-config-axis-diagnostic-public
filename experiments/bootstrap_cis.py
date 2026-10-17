"""Seed-level bootstrap intervals for the paper's key effects.

The intervals are descriptive: with only three seeds they are wide by
construction. They are reported so the manuscript does not present point
estimates without uncertainty.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmarks"
RNG = np.random.default_rng(20261011)


def bootstrap_mean(values: list[float], n_boot: int = 20000) -> dict:
    x = np.asarray(values, dtype=np.float64)
    draws = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "values": x.tolist(),
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_seeds": len(x),
    }


def main() -> int:
    innovation = json.loads(
        (BENCH / "innovation_upgrade.json").read_text(encoding="utf-8")
    )
    cql = json.loads(
        (BENCH / "cql_alpha_five_level.json").read_text(encoding="utf-8")
    )

    same_batch = {}
    for domain in ("hopper", "walker2d"):
        means = [
            row["mean_difference"]
            for row in innovation["matched_contrast"][domain]["per_seed"].values()
        ]
        same_batch[domain] = bootstrap_mean(means)

    cql_r2 = bootstrap_mean(
        [row["alpha_r2"] for row in cql["per_seed"].values()]
    )
    fqe_r2 = bootstrap_mean(
        [
            row["tau"]
            for row in innovation["variance_decomposition"]["hopper_fqe"][
                "per_seed"
            ].values()
        ]
    )
    result = {
        "same_batch_seed_clustered_mean_difference": same_batch,
        "cql_alpha_r2_across_seeds": cql_r2,
        "fqe_tau_r2_across_seeds": fqe_r2,
        "note": "Descriptive percentile bootstrap over three seed-level values.",
    }
    print(json.dumps(result, indent=2))
    out = BENCH / "bootstrap_cis.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
