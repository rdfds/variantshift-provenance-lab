# Structured-shift results

These files extend the 195-assay, 169-protein ProteinGym v1.3 cohort with modern supervised
baselines, a paired twelve-model zero-shot landscape, four structured split protocols, interval
diagnostics, top-variant selection metrics, held-out-protein, sequence-, structure-, and curated
Pfam-family transfer, and protein-grouped model-selection evaluation. The complete protocol and its
limits are in
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
- All 195 assays have complete paired coverage across twelve official zero-shot score columns.
  VenusREM leads at 0.542 protein-balanced Spearman and beats ESM-2 650M by 0.114; its simultaneous
  95% interval across eleven baseline comparisons is 0.081–0.146.
- With complete UniProt IDs absent from training across five shuffled group partitions, pooled
  nonlinear regression reaches 0.539 mean within-assay Spearman versus 0.515 for ridge. Standard
  intervals achieve 0.807 and 0.804 observed coverage respectively.
- MMseqs2 groups the 169 proteins into 156 components at ≥30% identity and ≥80% bidirectional
  assayed-segment coverage. Ten multi-protein families cover 23 proteins; zero qualifying alignments
  cross components. Holding out complete components yields 0.533 Spearman for nonlinear regression
  and 0.513 for ridge, with standard coverage of 0.802 and 0.801. Component-bootstrap paired
  Spearman changes are −0.0060 (95% interval −0.0124 to −0.0011) and −0.0021 (−0.0038 to −0.0007),
  resolving a small close-homolog label-transfer effect.
- Foldseek searches all 169 official ProteinGym AlphaFold structures exhaustively and retains a
  structure edge only when both directions reach ≥0.95 homology probability, ≥0.50 TM-score, and
  ≥80% coverage. Union with the sequence graph yields 148 combined families; 14 multi-protein
  families cover 35 proteins and the largest contains five. Zero qualifying structure pairs cross
  components. Combined-family holdout reaches 0.532 Spearman for nonlinear regression and 0.512 for
  ridge. Versus sequence-family holdout, component-bootstrap changes are −0.00045 (95% interval
  −0.00315 to 0.00235) and −0.00016 (−0.00121 to 0.00082).
- InterPro/Pfam mapping adds exact curated domain families over the assayed region, producing 132
  primary components. Holdout reaches 0.533 Spearman for nonlinear regression and 0.513 for ridge;
  neither exact Pfam families nor the broader 97-component clan stress test resolves an additional
  penalty beyond sequence/structure holdout.
- Removing fixed ESM scores under the exact-Pfam split reduces nonlinear Spearman from 0.533 to
  0.360 and ridge from 0.513 to 0.318. Paired family-bootstrap differences are 0.174 (0.151–0.195)
  and 0.195 (0.169–0.219), showing that pretrained score priors supply most of pooled performance.
- A logistic classifier predicts local-supervised versus ESM-2-650M wins with protein-grouped
  out-of-fold ROC-AUC 0.829 and 0.770 accuracy across 975 split-specific decisions. The majority
  baseline is 0.592. These repeated splits are not independent biological replicates.

## Artifact index

| File | Contents |
| --- | --- |
| `official-supervised-audit.csv` | Join, completeness, and target-agreement audit for 585 official assay/split files |
| `official-supervised-runs.csv` | Assay-level official-model ranking and top-selection results |
| `official-supervised-summary.csv` | UniProt-weighted means and 10,000-replicate bootstrap intervals |
| `modern-zero-shot-audit.csv` | Per-assay exact-join and twelve-model common-coverage ledger |
| `modern-zero-shot-runs.csv` | Exactly paired assay-level ranking and selection metrics |
| `modern-zero-shot-summary.csv` | Protein-balanced means and bootstrap intervals for twelve models |
| `modern-zero-shot-vs-esm2.csv` | Paired, multiplicity-controlled comparisons and rank stability |
| `esm2-embedding-index.csv` | Sequence, model, shape, byte-length, and SHA-256 record for every cached embedding |
| `embedding-probe-runs.csv` | Five repetitions × four splits × three interval methods for every assay |
| `embedding-probe-summary.csv` | Protein-weighted performance, selection, coverage, and width summaries |
| `embedding-probe-risk-coverage.csv` | Assay-level error at four retained-confidence fractions |
| `embedding-probe-risk-summary.csv` | Protein-weighted risk-coverage curves |
| `heldout-protein-assays.csv` | Assay-level ridge/nonlinear results on unseen proteins |
| `heldout-protein-risk-coverage.csv` | Selective-risk results for the cross-protein models |
| `heldout-protein-summary.csv` | Protein-weighted transfer and interval summary |
| `heldout-protein-repeat-estimates.csv` | Protein-balanced result for each shuffled group partition |
| `heldout-protein-risk-summary.csv` | Protein-weighted selective-risk summary |
| `sequence-family-assignments.csv` | Deterministic UniProt-to-family mapping and component membership |
| `sequence-family-alignments.csv` | Exhaustive MMseqs2 alignment ledger with family-edge and cross-cluster audit fields |
| `sequence-family-sensitivity.csv` | Cluster composition across identity and coverage thresholds |
| `sequence-family-audit.csv` | MMseqs2 version, primary thresholds, cohort counts, and leakage result |
| `heldout-family-assays.csv` | Assay-level results with complete sequence-family components held out |
| `heldout-family-risk-coverage.csv` | Selective-risk results under family holdout |
| `heldout-family-summary.csv` | Protein-weighted family-transfer and interval summary |
| `heldout-family-repeat-estimates.csv` | Per-repeat sequence-family transfer estimates |
| `heldout-family-risk-summary.csv` | Protein-weighted selective-risk summary under family holdout |
| `heldout-family-comparison.csv` | Paired component-bootstrap comparison against ordinary protein holdout |
| `structure-input-audit.csv` | Per-UniProt official PDB filename, range, byte length, and SHA-256 |
| `structure-family-alignments.csv` | Complete reciprocal Foldseek pair ledger with primary-edge and cluster audit fields |
| `structure-family-sensitivity.csv` | Combined graph composition across probability, TM-score, and coverage thresholds |
| `structure-family-audit.csv` | Archive hash, Foldseek version, exact search protocol, thresholds, and leakage result |
| `sequence-structure-family-assignments.csv` | Deterministic combined-family mapping and component membership |
| `heldout-structure-family-assays.csv` | Assay-level results with complete combined families held out |
| `heldout-structure-family-risk-coverage.csv` | Selective-risk results under combined-family holdout |
| `heldout-structure-family-summary.csv` | Protein-weighted combined-family transfer and interval summary |
| `heldout-structure-family-repeat-estimates.csv` | Per-repeat sequence/structure transfer estimates |
| `heldout-structure-family-risk-summary.csv` | Protein-weighted selective-risk summary under combined-family holdout |
| `heldout-structure-family-vs-protein.csv` | Paired bootstrap comparison against ordinary protein holdout |
| `heldout-structure-family-vs-sequence-family.csv` | Paired bootstrap comparison against sequence-only family holdout |
| `curated-uniprot-mapping.csv` | Exact entry-name resolution, canonical length, and sequence hash audit |
| `curated-coordinate-mapping.csv` | Contiguous assayed-region mapping and exact coverage ledger |
| `curated-pfam-domain-overlaps.csv` | Complete mapped-domain overlap ledger and primary inclusion flag |
| `curated-family-assignments.csv` | Exact-Pfam primary family components |
| `curated-clan-family-assignments.csv` | Broader Pfam-clan sensitivity components |
| `curated-family-edges.csv` | Exact-Pfam protein edges and final-component audit |
| `curated-clan-family-edges.csv` | Pfam-clan sensitivity edges and final-component audit |
| `curated-family-audit.csv` | InterPro, Pfam, UniProt versions, thresholds, coverage, and component counts |
| `curated-api-snapshot.tar.gz` | Frozen public UniProt/InterPro responses used to reconstruct curated mappings |
| `heldout-curated-family-assays.csv` | Assay-level repeated exact-Pfam-family holdout results |
| `heldout-curated-family-summary.csv` | Protein-balanced curated-family transfer and interval summary |
| `heldout-curated-family-repeat-estimates.csv` | Per-repeat curated-family transfer estimates |
| `heldout-curated-family-vs-protein.csv` | Component-bootstrap comparison against protein holdout |
| `heldout-curated-family-vs-structure-family.csv` | Component-bootstrap comparison against sequence/structure holdout |
| `clan-sensitivity/` | Broad Pfam-clan holdout outputs and comparison against exact Pfam families |
| `feature-ablation/` | Strict curated-family full-feature versus mutation-only evaluation |
| `crossover-examples.csv` | Training-only assay properties and paired supervised/zero-shot outcomes |
| `crossover-heldout-predictions.csv` | UniProt-grouped out-of-fold classifier probabilities |
| `crossover-summary.csv` | Classifier discrimination, calibration, and baseline comparisons |
| `crossover-logistic-coefficients.csv` | Standardized logistic coefficients for interpretation |
| `run-manifest.json` | Public-input, embedding-index, source-revision, environment, and artifact hashes |

The per-variant prediction tables for protein, sequence-family, structure-family, curated-family,
clan-sensitivity, and feature-ablation runs—and the frozen ESM-2 embedding matrices—are generated
locally and excluded from version control. None of the aggregate claims above requires committing
source measurements, structures, or embeddings.
