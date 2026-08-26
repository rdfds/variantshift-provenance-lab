# Multi-protein validation methods

## Study objective

The ProteinGym extension tests whether VariantShift's random-versus-unseen-position result
generalizes beyond the original TEV case study. It also compares those supervised models with
published ESM zero-shot scores without conflating fixed-score subset sensitivity with a
training-distribution generalization gap.

## Data release

The analysis uses the ProteinGym v1.3 substitution benchmark, its reference index, and the
official v1.3 zero-shot score archive. Raw and per-variant files remain outside version control.
Their URLs, byte lengths, and SHA-256 digests are recorded in the study manifest.

ProteinGym contains 217 substitution DMS assays. The primary cohort is determined before model
evaluation using the following fixed criteria:

1. At least 500 finite single-substitution measurements.
2. At least 20 distinct mutated residue positions.
3. At least 10 distinct experimental scores.
4. No invalid or duplicate single-substitution identifiers.
5. Every encoded reference residue agrees with the assay's target sequence.
6. Every supplied mutated sequence equals the target sequence after applying its mutation.

The complete inclusion ledger is committed with the results, including exclusion reasons for
every rejected assay. Of 217 audited assays, 195 assays spanning 169 UniProt IDs pass. The cohort
contains 689,994 single-substitution measurements across human, other eukaryotic, prokaryotic,
and viral proteins and across activity, binding, expression, organismal-fitness, and stability
assays.

## Supervised protocol

Each eligible assay is evaluated independently. No labels, fitted parameters, or normalization
statistics cross assay boundaries. The same fixed baselines used in the TEV study are fit within
each assay:

- a training-mean control;
- ridge regression on transparent biochemical mutation features; and
- additive ridge regression with biochemical, residue-position, substitution-class, and exact-
  mutation indicators.

For consecutive seeds 42–51, models are evaluated on a grouped random-variant split and a
residue-position holdout. Each training partition is divided into 80% fitting and 20%
calibration subsets. Test metrics include Spearman correlation, RMSE, MAE, R², normalized RMSE,
and observed coverage of nominal 80% split-conformal intervals.

Every split records exact-variant overlap. Position holdout additionally records the number of
shared residue positions, which must equal zero. The command fails rather than writing results if
either leakage invariant is violated.

Random and unseen-position metrics are paired within assay, model, and seed. Repeated seeds are
averaged within assay; assays are then averaged within UniProt ID. Headline means and 95%
bootstrap intervals resample UniProt IDs 10,000 times. Thus, repeated split seeds and proteins
represented by multiple assays are not treated as independent biological observations.

## Official ESM score audit

The zero-shot comparison uses ProteinGym's published columns for `ESM1v_ensemble` and the complete
ESM-2 scaling series: 8M, 35M, 150M, 650M, 3B, and 15B parameters. Variant identifiers are joined
one-to-one against the already-audited source assays. Before evaluation, VariantShift records:

- source and score-archive single-variant counts;
- duplicate score identifiers;
- matched experimental-score counts;
- maximum disagreement between source and score-archive DMS values;
- finite-score counts for every model; and
- common finite-score coverage across all requested ESM columns.

An assay is excluded from the ESM analysis if the variant join is incomplete, experimental values
disagree, identifiers are duplicated, or common model-score coverage is below 95%. All ESM models
are evaluated on the same complete-case variants within an assay.

ESM scores are fixed and are not fit to the ProteinGym assay labels. Random and unseen-position
test subsets are constructed with the same seeds used for supervised evaluation, but their score
difference measures sensitivity to test-set composition. It is therefore reported as a **subset
difference**, not a supervised generalization gap. The benchmark does not claim that protein
sequences or homologs were absent from model pretraining.

## Reproduction

```bash
make install
make proteingym-download
make proteingym-audit
make proteingym-benchmark
make proteingym-zero-shot
make proteingym-figure
```

The official zero-shot archive is approximately 1.9 GB compressed. The assay archive, reference
index, and score archive are downloaded only into the ignored `data/raw/proteingym/` directory.
