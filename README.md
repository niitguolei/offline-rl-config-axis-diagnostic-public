# When Free Value Estimates Rank the Configuration Axis

Public release package for the paper:

> **When Free Value Estimates Rank the Configuration Axis: Confounding,
> Stability, and a Diagnostic Protocol for Offline RL Model Selection.**

The repository contains the analysis code, raw run records, result JSON files,
paper figures, manuscript sources, and pre-submission materials.

## What the Project Studies

Offline RL practitioners often rank training configurations with the algorithm's
own critic. This project shows that the free ranking signal can be dominated by
one factor, usually a configuration axis but in one case the seed. The paper
provides a practical diagnostic based on axis independence, within-run quality
tracking, stability across the estimator's own hyperparameters, and a divergence
gate for unstable runs.

## Repository Layout

```text
src/orl/                 Core library code
experiments/             Analysis and experiment scripts
scripts/                 Utility and reproduction scripts
results/benchmarks/      Raw run records and result JSON files
docs/                    Research plans, gates, writing constraints
paper/                   Manuscript, figures, references, submission package
vendor/                  Third-party reference code and provenance notes
```

## Data

The experiments use the public D4RL MuJoCo datasets. The datasets are not
redistributed in this repository. Download them from the official D4RL release
described by Fu et al. (arXiv:2004.07219).

The processed run records generated in this study are stored as JSONL files in
`results/benchmarks/`. They contain no human-subject or personally identifiable
data.

## Environment

The primary development environment is a Conda environment named
`pytorch_env` with PyTorch and the packages listed in `requirements.txt`.
The experiments were developed for a single RTX 5060 Laptop GPU with 8 GB
memory and 16 GB system RAM.

## Reproduction

See `docs/REPRODUCE.md` for the full workflow. The main analysis scripts are
listed in `paper/submission/supplementary_manifest.md`.

Run:

```powershell
python scripts/reproduce_all.py
```

This script runs the deterministic analysis and figure-generation pipeline from
the committed result records.

## Paper Artifacts

- Markdown manuscript: `paper/manuscript_v1.md`
- C&IS template source: `paper/submission/manuscript_cis.tex`
- Compile-checked fallback: `paper/submission/manuscript_compile.tex`
- Verified references: `paper/references_verified.md`
- Figures: `paper/figures/`
- Submission package: `paper/submission/`

## Result Mapping

`results/README.md` maps the paper figures and tables to the JSON records and
scripts that generate them.

## License

- Code: MIT License, see `LICENSE`.
- Derived run records and result files: CC BY 4.0, see `DATA_LICENSE.md`.

## Third-Party Code

See `vendor/THIRD_PARTY.md` for upstream provenance and license notes. Do not
redistribute third-party code without preserving its upstream license.

## Citation

See `CITATION.cff`. The Zenodo DOI will be added after the first public
release.

## Contact

For questions about the code or data, contact the corresponding author.
