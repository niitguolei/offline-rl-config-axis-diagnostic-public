# Reproduction Guide

Updated: 2026-10-17.

## 1. Environment

Create or activate the Conda environment:

```powershell
conda activate pytorch_env
```

Install the packages listed in `requirements.txt`. The full training pipeline
also requires a working MuJoCo installation and the D4RL package.

## 2. Data

Download the D4RL MuJoCo datasets from the official D4RL release. The project
uses hopper, walker2d, and halfcheetah medium-replay data.

The original development path was:

```text
D:\workspace\codex\findRLIdea\data\d4rl
```

The datasets are not included in this repository.

## 3. Deterministic Analysis

Run:

```powershell
python scripts/reproduce_all.py
```

This runs the analysis and figure-generation pipeline from the committed
result records.

## 4. Main Analysis Scripts

- `experiments/innovation_upgrade.py`
- `experiments/round12a_five_level.py`
- `experiments/fqe_sensitivity_judgement.py`
- `experiments/am_judgement.py`
- `experiments/cql_alpha_five_level.py`
- `experiments/prospective_taxonomy_multiseed.py`
- `experiments/audit_multiplicity_uniform.py`
- `experiments/audit_multiplicity_uniform_guarded.py`
- `experiments/bootstrap_cis.py`
- `experiments/make_paper_figures.py`

## 5. Expected Outputs

The scripts write or refresh the corresponding JSON files in
`results/benchmarks/` and the figures in `paper/figures/`.

See `results/README.md` for the mapping from paper artifacts to scripts and
result files.

## 6. Full Training Runs

Full retraining uses the runners in `experiments/` and the official D4RL data.
It is GPU-intensive and was performed on a single RTX 5060 Laptop GPU with
8 GB memory and 16 GB system RAM.
