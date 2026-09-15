# Predictor-span pilot v1

## Decision

**Pursue the phenotype-steering hypothesis, but do not claim a representational
ceiling or a publication-level result yet.**

The pilot rejects the simplest version of the predictor-span hypothesis. The
95 published ProteinGym scores are highly redundant, but their residual
differences contain reproducible assay-specific information. A position-held-out
linear recombination improves over selecting the best single predictor on every
one of the 22 development assays by Spearman correlation.

The resulting candidate claim is:

> Published variant-effect predictors form a compact reusable basis. Their main
> deployment failure is not always absence of signal; it is failure to identify
> which direction in that basis corresponds to the requested experimental
> phenotype.

This is an exploratory result on public ProteinGym development data. It did not
read any VariantShift confirmation outcome or artifact.

## Design

- Eleven proteins with paired, contrasting assays such as abundance versus
  activity, surface expression versus channel function, and expression versus
  binding.
- 22 assay tasks, 53,499 shared mutation measurements across pairs.
- The complete set of 95 published score columns in the ProteinGym v1.3
  zero-shot archive.
- Predictor scores rank-normalized within each assay.
- Five-fold outer cross-validation grouped by mutated position.
- Ridge regularization chosen in an inner position-grouped cross-validation.
- Comparators: the uniform panel mean and the best individual score selected
  using each training fold only.
- Protein-pair bootstrap with 2,000 resamples; both phenotypes from a sampled
  protein remain together.

The linear model is a **lower bound on information extractable from the panel**.
It is not an oracle upper bound, and failure would not establish that the outcome
is absent from nonlinear combinations or other representations.

## Results

| Quantity | Result |
| --- | ---: |
| Paired proteins | 11 |
| Assay tasks | 22 |
| Published predictors | 95 |
| Shared mutations across assay pairs | 53,499 |
| Median correlation between paired experimental outcomes | 0.325 |
| Median agreement of predictor scores across the paired assays | 1.000 |
| Median effective rank of the 95-score matrix | 6.46 |
| Median PCs needed for 90% predictor variance | 17.5 |

Mean position-held-out performance:

| Policy | Spearman | Selection gain (SD) | Top-decile recall |
| --- | ---: | ---: | ---: |
| Uniform predictor mean | 0.4888 | 0.6725 | 0.2050 |
| Cross-fitted best individual predictor | 0.5097 | 0.6075 | 0.2069 |
| Cross-fitted linear predictor span | 0.5938 | 0.7753 | 0.2421 |

Linear span minus best individual predictor:

| Endpoint | Mean difference | Protein-pair bootstrap 95% interval |
| --- | ---: | ---: |
| Spearman | +0.0840 | +0.0662 to +0.1058 |
| Selection gain (SD) | +0.1678 | +0.0965 to +0.2742 |
| Top-decile recall | +0.0352 | +0.0193 to +0.0589 |

The Spearman improvement is positive on all 22 assay tasks. Top-decile recall
improves on 18 of 22.

## Interpretation

The predictor panel is neither 95 independent biological views nor one generic
score. Its effective dimensionality is small relative to its nominal size, but
the remaining directions matter: assay-specific supervision can consistently
extract more signal than any one score.

This supports a distinction that ordinary leaderboards do not make:

1. **Representation failure:** the predictor panel contains no recoverable
   information about the assay.
2. **Readout failure:** the information exists, but an assay-agnostic score or
   global model ranking uses the wrong combination.

The present result supports systematic readout failure. It does not yet
establish how often true representation failure occurs.

## Novelty boundary

Paired-assay disagreement, phenotype-specific prediction, few-shot protein
fitness learning, and multi-objective learning are already published or under
active study. The publishable opening is therefore not simply that two assays
of the same protein disagree or that a supervised ensemble beats one model.

The sharper contribution would be a comprehensive geometry and identifiability
analysis of the published predictor ecosystem, followed by an assay-support
test that distinguishes representation failure from readout failure before a
large experiment is run.

## Next cheap falsification gate

Run a nested label-budget curve at 0, 8, 16, 32, and 64 measurements per assay.
Pilot substitutions must be chosen without outcomes and removed, together with
their mutated positions, from evaluation. Compare:

- best fixed predictor;
- uniform panel;
- pilot-selected single predictor;
- low-rank phenotype-steered panel; and
- an assay-description-conditioned score when variant-level predictions are
  available.

Proceed only if, by 16 or 32 measurements, the steered panel improves
top-decile selection gain over the uniform panel with a protein-pair bootstrap
interval excluding zero and the effect is directionally positive in at least
9 of 11 proteins. Also require the assay-description-conditioned score to add
held-out information beyond the 95-score span; otherwise that part of the
hypothesis is not supported.

## Reproducibility

Run:

```bash
PYTHONPATH=src python3 -m variantshift.predictor_span_pilot \
  configs/predictor-span-pilot-v1.json \
  --output-dir results/predictor-span-pilot-v1
```

Outputs:

- `paired-phenotype-audit.csv`
- `task-panel-recoverability.csv`
- `protein-pair-bootstrap.csv`
- `summary.json`
- `manifest.json`
