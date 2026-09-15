# VariantShift

**Do protein mutation predictors work when the biological setting changes?**

VariantShift is a Python benchmark and analysis framework for protein variant-effect models.
It tests predictions on unfamiliar mutation positions, proteins, sequence families, and
external datasets using TEV protease measurements, ProteinGym, and MaveDB.

## What it does

- Compares simple supervised baselines, pretrained scores, and published model predictions.
- Measures how performance and uncertainty change beyond familiar training examples.
- Evaluates whether choosing a predictor—or abstaining—helps on unfamiliar biological tasks.
- Provides result tables, figures, and an interactive benchmark explorer.

## Main findings

| Study | Finding |
| --- | --- |
| ProteinGym baseline, 195 assays | Mean Spearman falls from **0.617 to 0.351** when test positions are unseen. |
| Transfer across sequence families | A pooled nonlinear model reaches **0.533** mean Spearman; removing pretrained ESM score features reduces it to **0.360**. |
| External MaveDB evaluation, 21 assays | ESM-2 8M reaches **0.105** mean Spearman; top-decile recall is close to random. |
| Exploratory predictor combinations, 22 assays | A position-held-out linear combination improves mean Spearman from **0.5097 to 0.5938** over selecting one predictor. |

These results show that evaluation design matters. Strong benchmark rankings do not always
translate into useful predictions in a different biological setting. The predictor-combination
and selective-deployment studies remain development results; confirmation outcomes remain sealed.

The studies use existing measurements. Pretrained models may have seen related sequences,
and the results do not establish prospective experimental benefit.

## Explore

- [Getting started](docs/GETTING_STARTED.md) — installation and a first benchmark.
- [Methods and results](docs/PROJECT_REPORT.md) — study design, findings, and limitations.
- [Interactive explorer](site/index.html) — browse the benchmark results.
- [Research records](docs/RESEARCH_RECORDS.md) — detailed methods and reproducibility materials.

Code is released under the [MIT License](LICENSE). Source datasets retain their providers' terms.
