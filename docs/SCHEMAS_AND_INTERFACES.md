# Schemas and execution interfaces

`variantshift` 1.0 uses seven versioned row contracts.

| Schema | Unique key | Purpose |
| --- | --- | --- |
| `targets-v1` | panel, target | Canonical sequence and sequence hash |
| `variants-v1` | panel, target, variant | Complete single-substitution universe |
| `predictions-v1` | protocol, panel, target, variant, model | Frozen outcome-free scores |
| `outcomes-v1` | protocol, panel, dataset, assay, target, variant | Revealed experimental effects |
| `task-metrics-v1` | protocol, panel, dataset, assay, target, model | Equal-weight task evaluation |
| `transport-features-v1` | protocol, panel, dataset, assay, target, model | Outcome-free task–model descriptors |
| `risk-coverage-v1` | policy, coverage | Legacy failure-risk and utility curves |
| `risk-coverage-v2` | policy, coverage | Selection regret, failure risk, tail utility, and confidence |

Every prediction row carries protocol, panel, target, variant, model, model version, score, and
status. Every task metric carries dataset, assay, target, protein, family, and model identifiers.
Sequence and frame hashes are deterministic under row and column reordering.

`PanelAdapter` exposes target acquisition only. It has no outcome method. `ModelAdapter` exposes
target preparation, scoring, and provenance. Fair-ESM, explicit fair-ESM ensembles, precomputed
scores, and shell-free external command adapters share the same prediction schema.

Prediction cache keys bind the model specification, checkpoint identity, target sequence hash, and
variant table. Interrupted jobs can resume without recomputing complete targets. External commands
receive explicit paths as argument tokens and are never executed through a shell.

Primary commands:

```text
variantshift panel-freeze
variantshift mavedb-freeze-complement-targets
variantshift models-preflight
variantshift models-execution-audit
variantshift models-qualification-audit
variantshift models-freeze-final-panel
variantshift predict-panel
variantshift transport-features-proteingym
variantshift transport-features-confirmation
variantshift transport-fit
variantshift conservative-auditor-fit
variantshift conservative-auditor-score
variantshift conservative-auditor-evaluate
variantshift confirmation-overlap-audit
variantshift confirmation-freeze-pfam
variantshift confirmation-freeze-structure-families
variantshift confirmation-freeze-tasks
variantshift compute-budget-check
variantshift confirmation-freeze
variantshift preregistration-build
variantshift confirmation-register
variantshift confirmation-reveal
variantshift transport-evaluate
variantshift site-build
```

`confirmation-overlap-audit` creates exact-sequence and MMseqs2-family novelty strata before
outcome access. Optional Foldseek structure-family and Pfam-clan annotations use three-state
auditing: `audited` with an overlap result, or `undocumented`; absence never means clean.

`models-execution-audit` reports the number of schema-valid executed configurations, model/input
families, and targets reaching 95% shared substitution coverage. It always records qualification as
not started. Executable model preflight is the separate parity, repeatability, license, and
container-qualification stage. Preregistration requires at least eight qualified models, four
model/input families, and 300 common passing targets. Confirmation evaluation requires the lock to
be in `revealed` state and verifies the requested bundle, predictions, and outcomes against the
recorded hashes before reading them.
