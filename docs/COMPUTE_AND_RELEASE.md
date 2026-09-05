# Compute, reproducibility, and release

## ARCH execution

Install the workflow extra and make a local copy of `workflow/config.example.yaml`. The default
workflow builds the development features, refits transport, audits model metadata, and rebuilds the
site. It does not run confirmation scoring automatically.

```bash
pip install -e '.[dev,workflow]'
snakemake --snakefile workflow/Snakefile --profile workflow/profiles/slurm \
  --configfile workflow/config.local.yaml
```

The Slurm profile exposes memory, runtime, GPU, and partition resources. Change its partition to an
actual ARCH GPU partition before submission. The local environment used to validate this workflow
has no `sbatch`, `apptainer`, or `foldseek`; cluster execution has therefore not been claimed.

## Container procedure

Build `containers/variantshift.def` for CPU analysis and `containers/fair-esm.def` for CUDA scoring.
After each build, record:

```bash
apptainer inspect image.sif
sha256sum image.sif
```

The resulting digest belongs in the model specification before executable preflight. A recipe path
is not a container digest and does not satisfy provenance.

## Preflight procedure

Use a small multi-target panel that includes short, long, and structurally resolved proteins.
Execution must be requested explicitly:

```bash
variantshift models-preflight configs/model-panel-v1.json \
  --targets preflight/targets.csv --variants preflight/variants.csv \
  --parity-dir preflight/official-scores --execute \
  --output results/model-panel-v1/executable-audit.csv
```

Not-run, failed, restricted, undocumented-license, sub-95%-coverage, sub-0.99-parity, or
sub-0.999-repeat configurations remain visible exclusions.

## Cost controls

Profile one representative target per model before scheduling a panel. Record wall time, GPU type,
GPU-hours, peak memory, cache size, and substitution throughput. Use ARCH first. Cloud jobs require
a separate cost manifest, an alert at 50%, 75%, and 90%, and a hard stop at $2,000. No workflow in
this repository silently falls back to paid compute.

## Release checklist

1. Full tests, lint, workflow dry-run, and static-link audit pass.
2. Target, model, method, environment, checkpoint, structure, MSA, and prediction hashes are fixed.
3. The public preregistration bundle is uploaded before confirmation outcomes are retrieved.
4. The reveal ledger is committed without modifying frozen artifacts.
5. All preregistered and sensitivity results are generated from the recorded revision.
6. The static explorer, package, manifests, intermediate predictions, and analysis tables are
   deposited as a Zenodo release candidate.
7. The Zenodo DOI and release digest are added to `CITATION.cff` without rewriting study history.
