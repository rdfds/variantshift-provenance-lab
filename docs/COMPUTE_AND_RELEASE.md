# Compute, reproducibility, and release

## ARCH and Modal execution

Install the workflow extra and make a local copy of `workflow/config.example.yaml`. NumPy is held
below 2 for compatibility with the pinned PyTorch 2.2 fair-esm image, and the Slurm executor is
pinned to 2.5.4 so the workflow and PLM extras remain dependency-compatible. The default workflow
builds the development features, refits transport, audits model metadata, reruns target-only
sequence overlap auditing, and rebuilds the site. It does not run confirmation scoring
automatically.

```bash
pip install -e '.[dev,workflow]'
snakemake --snakefile workflow/Snakefile --profile workflow/profiles/slurm \
  --configfile workflow/config.local.yaml
```

The Slurm profile exposes memory, runtime, GPU, and partition resources. Change its partition to an
actual ARCH GPU partition before submission. ARCH execution remains unclaimed until scheduler
receipts exist. The outcome-sealed Modal runner is the qualified GPU path: it mounts only frozen
target/variant tables and prevalidated structures, writes immutable per-run shards to named
volumes, and records checkpoint, container, input, prediction, runner, source-tree, runtime,
hardware, cache, and failure evidence. Confirmation outcomes are not mounted or read.

```bash
pip install -e '.[dev,cloud]'
modal run workflow/modal_qualification.py --model-id esm2_650m \
  --dataset domainome --run-id qualification-a --shard-count 10 --expected-targets 426
modal run workflow/modal_qualification.py --model-id esm2_650m \
  --dataset domainome --run-id qualification-b --shard-count 10 --expected-targets 426
modal run workflow/modal_qualification.py --model-id esm2_650m \
  --dataset domainome --run-id qualification-a --merge --expected-targets 426
```

The sharded path is target-disjoint. Merge refuses to complete unless all 426 target audit rows are
present. The two run IDs use separate fresh cache namespaces; qualification rejects cache reuse.
Parity is executed separately against frozen official ProteinGym scores, then all three evidence
roots are evaluated together:

```bash
variantshift models-qualification-audit \
  configs/qualification-v1.json configs/model-panel-v1.json \
  containers/qualification-lock-v1.json \
  results/confirmation/domainome-v1/targets.csv \
  results/confirmation/domainome-v1/variants.csv \
  results/confirmation/domainome-v1/outcome-lock.json \
  artifacts/qualification/proteingym-parity-v1/targets.csv \
  artifacts/qualification/proteingym-parity-v1/variants.csv \
  artifacts/qualification/proteingym-parity-v1/official-scores.csv.gz \
  artifacts/qualification/runs/domainome/qualification-a \
  artifacts/qualification/runs/domainome/qualification-b \
  artifacts/qualification/runs/parity/parity-reference-v3 \
  --output-dir results/model-qualification-v1
```

The initial frozen audit passes eight configurations from four families on 413 shared targets.
The final composed audit under `results/model-qualification-final-v1/` passes ten configurations
from six families on the same 413-target intersection. Coverage is 97.34% to 100%, eight-target
official-score parity is at least 0.999993, and every independent zero-cache rerun has effectively
perfect rank correlation. Detailed evidence remains separated by qualification stage.

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

Before preflight, the execution-only audit verifies schema-valid predictions and the shared target
intersection without advancing qualification:

```bash
variantshift models-execution-audit configs/executable-panel-v1.json \
  results/confirmation/domainome-v1/targets.csv \
  results/confirmation/domainome-v1/variants.csv \
  --output-dir results/model-panel-v1/execution
```

## Cost controls

Profile one representative target per model before scheduling a panel. Record wall time, GPU type,
GPU-hours, peak memory, cache size, and substitution throughput. Use ARCH first. Cloud jobs require
a separate cost manifest, an alert at 50%, 75%, and 90%, and a hard stop at $2,000. No workflow in
this repository silently falls back to paid compute.

The executable guard reads `configs/compute-ledger.csv`, includes the proposed job cost, emits a
checksum-bound report, and exits with status 2 if the hard cap would be exceeded:

```bash
variantshift compute-budget-check configs/compute-ledger.csv \
  --planned-cost-usd 125 --output results/compute-budget-status.json
```

Model preflight records first-run wall time and substitution throughput. GPU type, GPU-hours, peak
memory, and cache bytes must be added by the ARCH job wrapper because a local process cannot
reliably infer scheduler-level resource accounting.

## Outcome-sealed Domainome targets

The Domainome target freeze is reproducible without loading its experimental residuals into the
analysis process:

```bash
variantshift domainome-freeze-targets \
  --output-dir data/confirmation-targets/domainome-v1
variantshift panel-freeze configs/panels/domainome-v1.json \
  results/confirmation/domainome-v1
```

The extractor hashes every source byte, decodes only the first three tab-delimited fields
(`dom_ID`, `PFAM_ID`, and `wt_seq`), and discards every trailing byte without decoding it. The
receipt is itself included in the target lock. The pinned Zenodo MD5 must match before the live
freeze succeeds.

## Release checklist

1. Full tests, lint, workflow dry-run, and static-link audit pass.
2. Target, model, method, environment, checkpoint, structure, MSA, and prediction hashes are fixed.
3. The registration is approved and the exact post-registration evaluator is immutably archived
   before confirmation outcomes are retrieved.
4. The reveal ledger is committed without modifying frozen artifacts.
5. All preregistered and sensitivity results are generated from the recorded revision.
6. The static explorer, package, manifests, intermediate predictions, and analysis tables are
   deposited as a Zenodo release candidate.
7. The Zenodo DOI and release digest are added to `CITATION.cff` without rewriting study history.
