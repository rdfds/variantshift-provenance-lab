# Structured-shift results

These files extend the 195-assay, 169-protein ProteinGym v1.3 cohort with modern supervised
baselines, four structured split protocols, interval diagnostics, top-variant selection metrics,
held-out-protein transfer, and protein-grouped model-selection evaluation. The complete protocol
and its limits are in [`docs/EXTENDED_METHODS.md`](../../../docs/EXTENDED_METHODS.md).

## Findings

- Kermut is the strongest audited official model: mean Spearman is 0.785 on random variants, 0.672
  on modulo positions, and 0.633 on contiguous positions. ProteinNPT reaches 0.776, 0.630, and
  0.584; the official ESM-1v embedding probe reaches 0.667, 0.549, and 0.507.
- The local frozen ESM-2 8M residue probe is deliberately smaller and weaker. Mean Spearman is
  0.688 on random variants, 0.400 on randomly grouped unseen positions, 0.407 on modulo positions,
  and 0.317 on contiguous positions.
- Standard nominal-80% conformal coverage for that local probe is 0.801 on random variants but
  falls to 0.607 on random unseen positions and 0.564 on contiguous positions. Distance scaling
  raises contiguous coverage to 0.911 by expanding normalized mean width to 5.88×. This is a
  sharpness tradeoff, not a calibrated-shift guarantee.
- Under contiguous shift, Kermut's top-10%-variant recall is 0.280; ProteinNPT's is 0.281 and the
  official ESM-1v probe's is 0.230. Global ranking and top-set recovery are related but distinct
  evaluation targets.
- With complete UniProt IDs absent from training, pooled nonlinear regression reaches 0.537 mean
  within-assay Spearman versus 0.512 for ridge. Standard intervals achieve 0.816 and 0.810
  observed coverage respectively.
- A logistic classifier predicts local-supervised versus ESM-2-650M wins with protein-grouped
  out-of-fold ROC-AUC 0.829 and 0.770 accuracy across 975 split-specific decisions. The majority
  baseline is 0.592. These repeated splits are not independent biological replicates.

## Artifact index

| File | Contents |
| --- | --- |
| `official-supervised-audit.csv` | Join, completeness, and target-agreement audit for 585 official assay/split files |
| `official-supervised-runs.csv` | Assay-level official-model ranking and top-selection results |
| `official-supervised-summary.csv` | UniProt-weighted means and 10,000-replicate bootstrap intervals |
| `esm2-embedding-index.csv` | Sequence, model, shape, byte-length, and SHA-256 record for every cached embedding |
| `embedding-probe-runs.csv` | Five repetitions × four splits × three interval methods for every assay |
| `embedding-probe-summary.csv` | Protein-weighted performance, selection, coverage, and width summaries |
| `embedding-probe-risk-coverage.csv` | Assay-level error at four retained-confidence fractions |
| `embedding-probe-risk-summary.csv` | Protein-weighted risk-coverage curves |
| `heldout-protein-assays.csv` | Assay-level ridge/nonlinear results on unseen proteins |
| `heldout-protein-risk-coverage.csv` | Selective-risk results for the cross-protein models |
| `heldout-protein-summary.csv` | Protein-weighted transfer and interval summary |
| `heldout-protein-risk-summary.csv` | Protein-weighted selective-risk summary |
| `crossover-examples.csv` | Training-only assay properties and paired supervised/zero-shot outcomes |
| `crossover-heldout-predictions.csv` | UniProt-grouped out-of-fold classifier probabilities |
| `crossover-summary.csv` | Classifier discrimination, calibration, and baseline comparisons |
| `crossover-logistic-coefficients.csv` | Standardized logistic coefficients for interpretation |
| `run-manifest.json` | Public-input, embedding-index, source-revision, environment, and artifact hashes |

The per-variant held-out-protein prediction table and frozen ESM-2 embedding matrices are generated
locally and excluded from version control. None of the aggregate claims above requires committing
source measurements or embeddings.
