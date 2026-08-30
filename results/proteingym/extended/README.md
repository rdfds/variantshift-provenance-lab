# Structured-shift results

These files extend the 195-assay, 169-protein ProteinGym v1.3 cohort with modern supervised
baselines, four structured split protocols, interval diagnostics, top-variant selection metrics,
held-out-protein, sequence-family, and sequence/structure-family transfer, and protein-grouped
model-selection evaluation. The complete protocol and its limits are in
[`docs/EXTENDED_METHODS.md`](../../../docs/EXTENDED_METHODS.md).

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
- MMseqs2 groups the 169 proteins into 156 components at ≥30% identity and ≥80% bidirectional
  assayed-segment coverage. Ten multi-protein families cover 23 proteins; zero qualifying alignments
  cross components. Holding out complete components yields 0.535 Spearman for nonlinear regression
  and 0.512 for ridge, with standard coverage of 0.803 and 0.800. The close-homolog safeguard does
  not materially change the performance estimate: paired Spearman changes are −0.0019 (95% interval
  −0.0068 to 0.0029) and +0.0002 (−0.0022 to 0.0028), using 10,000 UniProt bootstrap replicates.
- Foldseek searches all 169 official ProteinGym AlphaFold structures exhaustively and retains a
  structure edge only when both directions reach ≥0.95 homology probability, ≥0.50 TM-score, and
  ≥80% coverage. Union with the sequence graph yields 148 combined families; 14 multi-protein
  families cover 35 proteins and the largest contains five. Zero qualifying structure pairs cross
  components. Combined-family holdout reaches 0.533 Spearman for nonlinear regression and 0.511 for
  ridge. Versus sequence-family holdout, paired changes are −0.0021 (95% interval −0.0077 to 0.0029)
  and −0.0015 (−0.0042 to 0.0012). The detectable remote structure matches do not materially change
  the transfer estimate.
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
| `sequence-family-assignments.csv` | Deterministic UniProt-to-family mapping and component membership |
| `sequence-family-alignments.csv` | Exhaustive MMseqs2 alignment ledger with family-edge and cross-cluster audit fields |
| `sequence-family-sensitivity.csv` | Cluster composition across identity and coverage thresholds |
| `sequence-family-audit.csv` | MMseqs2 version, primary thresholds, cohort counts, and leakage result |
| `heldout-family-assays.csv` | Assay-level results with complete sequence-family components held out |
| `heldout-family-risk-coverage.csv` | Selective-risk results under family holdout |
| `heldout-family-summary.csv` | Protein-weighted family-transfer and interval summary |
| `heldout-family-risk-summary.csv` | Protein-weighted selective-risk summary under family holdout |
| `heldout-family-comparison.csv` | Paired protein-bootstrap comparison against ordinary protein holdout |
| `structure-input-audit.csv` | Per-UniProt official PDB filename, range, byte length, and SHA-256 |
| `structure-family-alignments.csv` | Complete reciprocal Foldseek pair ledger with primary-edge and cluster audit fields |
| `structure-family-sensitivity.csv` | Combined graph composition across probability, TM-score, and coverage thresholds |
| `structure-family-audit.csv` | Archive hash, Foldseek version, exact search protocol, thresholds, and leakage result |
| `sequence-structure-family-assignments.csv` | Deterministic combined-family mapping and component membership |
| `heldout-structure-family-assays.csv` | Assay-level results with complete combined families held out |
| `heldout-structure-family-risk-coverage.csv` | Selective-risk results under combined-family holdout |
| `heldout-structure-family-summary.csv` | Protein-weighted combined-family transfer and interval summary |
| `heldout-structure-family-risk-summary.csv` | Protein-weighted selective-risk summary under combined-family holdout |
| `heldout-structure-family-vs-protein.csv` | Paired bootstrap comparison against ordinary protein holdout |
| `heldout-structure-family-vs-sequence-family.csv` | Paired bootstrap comparison against sequence-only family holdout |
| `crossover-examples.csv` | Training-only assay properties and paired supervised/zero-shot outcomes |
| `crossover-heldout-predictions.csv` | UniProt-grouped out-of-fold classifier probabilities |
| `crossover-summary.csv` | Classifier discrimination, calibration, and baseline comparisons |
| `crossover-logistic-coefficients.csv` | Standardized logistic coefficients for interpretation |
| `run-manifest.json` | Public-input, embedding-index, source-revision, environment, and artifact hashes |

The per-variant held-out-protein, held-out-family, and held-out-structure-family prediction tables
and frozen ESM-2 embedding matrices are generated locally and excluded from version control. None
of the aggregate claims above requires committing source measurements, structures, or embeddings.
