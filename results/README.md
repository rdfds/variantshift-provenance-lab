# Committed results

All files in this directory are aggregate outputs derived from the TEV GROQ-seq v1.1 release. Raw measurements remain under the provider's data-use agreement and are not stored in Git.

| Artifact | Contents |
| --- | --- |
| [`benchmark.csv`](benchmark.csv) | Seed-42 metrics, conformal coverage, row counts, and leakage-audit fields for every target/split/model combination |
| [`robustness/benchmark-runs.csv`](robustness/benchmark-runs.csv) | Complete metrics for 10 repeated benchmarks using seeds 42–51 |
| [`robustness/summary.csv`](robustness/summary.csv) | Mean, standard deviation, range, and 5th/95th seed percentiles by target/split/model |
| [`robustness/generalization-gaps.csv`](robustness/generalization-gaps.csv) | Random-versus-unseen-position Spearman and coverage gaps paired within seed |
| [`robustness/generalization-gap-summary.csv`](robustness/generalization-gap-summary.csv) | Distribution of the paired generalization penalties |
| [`transfer/condition-transfer.csv`](transfer/condition-transfer.csv) | Full 20×20 source/target assay matrix under random and unseen-position splits |
| [`transfer/condition-transfer-summary.csv`](transfer/condition-transfer-summary.csv) | Diagonal and off-diagonal transfer statistics by split |
| [`report.html`](report.html) | Standalone seed-42 benchmark report with no external assets |
| [`run-manifest.json`](run-manifest.json) | Dataset, source revision, configuration, environment, artifact byte lengths, and SHA-256 hashes |

Verify the committed outputs:

```bash
variantshift verify-artifacts results/run-manifest.json
```

Regenerate the expanded analyses after downloading the dataset:

```bash
variantshift robustness data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv --output-dir artifacts/robustness
variantshift condition-transfer data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv --output-dir artifacts/transfer
variantshift shift-figure artifacts/robustness/generalization-gaps.csv artifacts/transfer/condition-transfer.csv --output artifacts/shift-analysis.svg
```
