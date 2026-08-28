# Committed results

Files in this directory are aggregate outputs from the TEV GROQ-seq case study and the public
ProteinGym v1.3 validation. Raw and per-variant source measurements are not stored in Git.

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

## ProteinGym validation

| Artifact | Contents |
| --- | --- |
| [`proteingym/eligibility.csv`](proteingym/eligibility.csv) | All 217 assay decisions, audit counts, and explicit exclusion reasons |
| [`proteingym/benchmark-runs.csv`](proteingym/benchmark-runs.csv) | 11,700 supervised assay/seed/split/model evaluations |
| [`proteingym/generalization-gaps.csv`](proteingym/generalization-gaps.csv) | Random-versus-position metrics paired within assay, seed, and model |
| [`proteingym/assay-summary.csv`](proteingym/assay-summary.csv) | Repeated-split means for every eligible assay and supervised model |
| [`proteingym/aggregate-summary.csv`](proteingym/aggregate-summary.csv) | UniProt-aggregated supervised results and 10,000-replicate bootstrap intervals |
| [`proteingym/esm-score-audit.csv`](proteingym/esm-score-audit.csv) | Variant joins, value agreement, duplicates, and per-model score completeness |
| [`proteingym/esm-subset-differences.csv`](proteingym/esm-subset-differences.csv) | 13,650 random-versus-position comparisons paired within assay, seed, and model |
| [`proteingym/esm-assay-summary.csv`](proteingym/esm-assay-summary.csv) | Per-assay ESM-1v and ESM-2 scaling results |
| [`proteingym/esm-aggregate-summary.csv`](proteingym/esm-aggregate-summary.csv) | UniProt-aggregated zero-shot results and bootstrap intervals |
| [`proteingym/run-manifest.json`](proteingym/run-manifest.json) | Three input hashes, protocol, environment, source revision, and result hashes |

The modern-baseline, calibration, selection, held-out-protein, and crossover extension is indexed
separately in [`proteingym/extended/`](proteingym/extended/). It omits the 8.7 MB per-variant
held-out prediction file; that file is reproducible locally and contains no aggregate claim used in
the README.

Verify the committed outputs:

```bash
variantshift verify-artifacts results/run-manifest.json
variantshift verify-artifacts results/proteingym/run-manifest.json
variantshift verify-artifacts results/proteingym/extended/run-manifest.json
```

Regenerate the expanded analyses after downloading the dataset:

```bash
variantshift robustness data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv --output-dir artifacts/robustness
variantshift condition-transfer data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv --output-dir artifacts/transfer
variantshift shift-figure artifacts/robustness/generalization-gaps.csv artifacts/transfer/condition-transfer.csv --output artifacts/shift-analysis.svg
```

ProteinGym commands and the exact cohort protocol are documented in
[`docs/PROTEINGYM_METHODS.md`](../docs/PROTEINGYM_METHODS.md).
