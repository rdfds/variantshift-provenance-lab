# Getting started

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
variantshift --help
```

On Windows, activate with `.venv\Scripts\activate` instead. The base installation supports
the benchmark and analysis commands; optional model-execution dependencies are described in
[compute and release](COMPUTE_AND_RELEASE.md).

## Explore existing results

Start with [methods and results](PROJECT_REPORT.md). To view the interactive explorer locally,
run this from the repository root and open `http://localhost:8000/site/`:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

## Run a TEV benchmark

Obtain the Align TEV release CSV under its provider's data-use terms. Raw measurements are
not included in the repository. The [TEV methods](METHODS.md) describe the expected data and filters.
Replace `path/to/tev.csv` below with your dataset path:

```bash
variantshift inspect path/to/tev.csv
variantshift benchmark path/to/tev.csv --output artifacts/benchmark.csv
```

The benchmark evaluates the configured baselines across the supported split regimes. New
outputs go in `artifacts/`. For repeated splits or other datasets, inspect the relevant command:

```bash
variantshift robustness --help
variantshift proteingym-benchmark --help
```

## Check the committed result files

```bash
variantshift verify-artifacts results/run-manifest.json
variantshift verify-artifacts results/proteingym/run-manifest.json
variantshift verify-artifacts results/proteingym/extended/run-manifest.json
variantshift verify-artifacts results/mavedb-external-v1/run-manifest.json
```

These checks compare artifact sizes and hashes. Missing raw inputs are reported separately;
passing the checks does not mean the analyses were rerun.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
pytest
```

See [research records](RESEARCH_RECORDS.md) for dataset-specific methods and detailed workflows.
