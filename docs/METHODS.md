# Methods

## Dataset

The first VariantShift case study uses version 1.1.0 of the Align Foundation TEV
protease pilot release. The source contains 18,486 variant rows, including site-saturation
and error-prone PCR libraries measured by GROQ-seq across 24 experimental conditions.
Measurements were collected at the Living Measurement Systems Foundry at NIST.

Raw measurements are not redistributed. The benchmark run is tied to the source CSV by
SHA-256 in `results/run-manifest.json`.

## Cohort construction

The default cohort applies three filters before any train/test split:

1. At least 1,000 total barcode reads.
2. No amino-acid insertion or deletion.
3. No nonsense mutation.

The resulting cohort contains 9,514 rows: 9,492 substitution variants and 22 separately
measured wild-type barcode aggregates. Two fitted functional endpoints are evaluated:
`log_ec50_prot_Sal10` and `log_ec50_prot_Sal25`.

## Variant representation

Mutation codes are parsed into reference residue, one-indexed position, and alternate
residue. The parser rejects noncanonical amino acids, nonpositive positions, synonymous
substitutions, and duplicate edits at the same position. A separate validation operation
checks encoded reference residues against the supplied wild-type sequence.

The biochemical representation contains mutation depth; position moments; signed and
absolute changes in Kyte-Doolittle hydropathy, side-chain volume, and approximate charge;
and changes in aromatic, polar, glycine, and proline content.

The additive representation augments those features with sparse indicators for residue
position, substitution class, and exact mutation identity.

## Evaluation splits

All splits use deterministic seed 42 where randomness is required.

### Random variant

Groups are defined by the complete mutation string and assigned with a grouped 80/20
shuffle. Exact variants therefore cannot cross the split. Residue positions can appear in
both partitions, matching a common interpolation setting.

### Unseen position

Twenty percent of mutated residue positions are selected for testing. Training variants
must be disjoint from those positions; test variants must contain only held-out positions.
Variants mixing training and held-out positions are excluded. The leakage audit asserts
zero shared positions.

### Higher mutation depth

Training uses only single substitutions. Testing uses variants with two through five amino-
acid substitutions. Wild-type rows and variants outside that range are excluded.

## Models

The mean baseline predicts the training mean. Biophysical ridge regression uses only the
continuous biochemical representation with standardized features. Additive ridge
regression uses the sparse augmented representation. Regularization parameters are fixed
before evaluation rather than tuned against a test regime.

## Metrics and uncertainty

Every training partition is split again into 80% model-fitting and 20% calibration subsets.
Point predictions are evaluated with Spearman correlation, RMSE, MAE, and R². Symmetric
80% split-conformal intervals use the finite-sample order statistic of absolute calibration
residuals. Observed interval coverage is reported separately for each evaluation regime.

Under covariate shift, conformal coverage need not remain at its nominal level. The coverage
drop is treated as a model-diagnostics result, not as a valid guarantee for unseen positions.

## Reproduction

```bash
make install
variantshift download data/raw --accept-data-use-agreement
make test
make report
```

Aggregate outputs can be regenerated without storing raw measurements in the repository.

