# Schemas and execution interfaces

`variantshift` 1.0 uses five versioned row contracts.

| Schema | Unique key | Purpose |
| --- | --- | --- |
| `targets-v1` | panel, target | Canonical sequence and sequence hash |
| `variants-v1` | panel, target, variant | Complete single-substitution universe |
| `predictions-v1` | protocol, panel, target, variant, model | Frozen outcome-free scores |
| `outcomes-v1` | protocol, panel, dataset, assay, target, variant | Revealed experimental effects |
| `task-metrics-v1` | protocol, panel, dataset, assay, target, model | Equal-weight task evaluation |

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
variantshift predict-panel
variantshift transport-features-proteingym
variantshift transport-fit
variantshift confirmation-freeze
variantshift preregistration-build
variantshift confirmation-register
variantshift confirmation-reveal
variantshift transport-evaluate
variantshift site-build
```
