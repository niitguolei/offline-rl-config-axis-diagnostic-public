"""Run the deterministic analysis and figure pipeline from committed records."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SCRIPTS = [
    "experiments/innovation_upgrade.py",
    "experiments/round12a_five_level.py",
    "experiments/fqe_sensitivity_judgement.py",
    "experiments/am_judgement.py",
    "experiments/cql_alpha_five_level.py",
    "experiments/prospective_taxonomy_multiseed.py",
    "experiments/audit_multiplicity_uniform.py",
    "experiments/audit_multiplicity_uniform_guarded.py",
    "experiments/bootstrap_cis.py",
    "experiments/make_paper_figures.py",
]


def main() -> int:
    for script in SCRIPTS:
        path = ROOT / script
        print(f"\n=== {script} ===", flush=True)
        result = subprocess.run([PYTHON, str(path)], cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED: {script} (exit {result.returncode})", flush=True)
            return result.returncode
    print("\nAll deterministic analysis and figure steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
