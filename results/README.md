# Results Mapping

Updated: 2026-10-17.

| Paper artifact | Script | Result file |
|---|---|---|
| Figure 1 dominant factor | `experiments/make_paper_figures.py` | `results/benchmarks/dominant_axis_r2.json` |
| Figure 2 same-batch contrast | `experiments/make_paper_figures.py` | `results/benchmarks/innovation_upgrade.json` |
| Figure 3 diagnostic flow | `experiments/make_paper_figures.py` | conceptual diagram |
| IQL five-level check | `experiments/round12a_five_level.py` | `results/benchmarks/round12a_five_level.json` |
| CQL five-level check | `experiments/cql_alpha_five_level.py` | `results/benchmarks/cql_alpha_five_level.json` |
| FQE sensitivity | `experiments/fqe_sensitivity_judgement.py` | `results/benchmarks/fqe_sensitivity_judgement.json` |
| AM negative result | `experiments/am_judgement.py` | `results/benchmarks/am_judgement.json` |
| Prospective taxonomy | `experiments/prospective_taxonomy_multiseed.py` | `results/benchmarks/prospective_taxonomy_multiseed.json` |
| Unguarded 22-item family | `experiments/audit_multiplicity_uniform.py` | `results/benchmarks/multiplicity_uniform.json` |
| Guarded 19-item stability | `experiments/audit_multiplicity_uniform_guarded.py` | `results/benchmarks/multiplicity_uniform_guarded.json` |
| Bootstrap intervals | `experiments/bootstrap_cis.py` | `results/benchmarks/bootstrap_cis.json` |

Raw run records are the JSONL files in `results/benchmarks/`. They are the
primary scientific record.
