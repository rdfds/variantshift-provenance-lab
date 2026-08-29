# Structured-shift extension

## Scope and contribution

This extension separates four questions that should not be collapsed into one leaderboard:

1. How do current supervised models perform under ProteinGym's official random, modulo, and
   contiguous five-fold protocols?
2. How does a locally fitted ESM-2 residue-embedding probe behave under those protocols and a
   repeated randomly grouped unseen-position split?
3. How reliable are prediction intervals and top-variant selections under each shift?
4. Can mutation effects or model-selection rules transfer to an entirely unseen protein?

ProteinGym already established that random folds can overestimate extrapolation to unseen residue
positions. VariantShift does not present that observation as novel. The extension focuses on a
reproducible comparison of model ranking, calibration, selection reliability, and transfer under
explicitly different estimands.

## Official supervised models

ProteinNPT, Kermut, and ProteinGym's ESM-1v embedding baseline are evaluated from the official
mutation-level `DMS_supervised_substitutions_scores.zip` archive. These are author-produced,
out-of-fold predictions under the published five-fold protocols; VariantShift does not retrain or
rename them as local models.

For every assay and split directory, the audit requires:

- a one-to-one mutation join against the independently downloaded ProteinGym assay;
- no duplicate mutation identifiers;
- complete finite predictions for every requested model;
- complete normalized targets; and
- experimental scores agreeing to numerical precision.

Metrics are calculated per assay. Assays are averaged within UniProt ID before aggregate means and
10,000-replicate protein-bootstrap intervals are computed. Official predictions are not reused for
the custom random-position split, because their training folds would make that evaluation leaky.

## Local ESM-2 residue probe

The local probe uses the frozen `esm2_t6_8M_UR50D` model. Each distinct wild-type sequence is
embedded once. Proteins longer than 1,022 residues are processed in overlapping windows and
overlapping residue representations are averaged. Cached matrices are keyed to the SHA-256 digest
of the exact target sequence.

For a single substitution, the supervised feature vector contains:

- the frozen ESM-2 representation at the wild-type residue position;
- one-hot reference and alternate amino acids;
- normalized residue position;
- transparent biochemical mutation descriptors; and
- protein length through the coordinate normalization.

A standardized ridge regression is fitted independently within each assay. This is a lightweight
embedding probe, not ESM-2 fine-tuning and not a claim of state-of-the-art performance.

## Four evaluation protocols

- **Random variant:** mutation identifiers are grouped, then 20% are sampled for testing.
- **Random position:** 20% of observed positions are selected randomly; every test position is
  absent from training.
- **Modulo position:** observed positions are assigned to one of five folds by their one-indexed
  residue number modulo five.
- **Contiguous position:** sorted observed positions are divided into five near-equal contiguous
  blocks.

Modulo and contiguous folds cycle exactly once over five repetitions. Random-variant and
random-position splits use seeds 42–46. Every run records exact-variant and residue-position overlap;
position-based evaluations must have zero shared positions.

Within each outer training partition, 20% of rows are reserved for calibration. The model never sees
calibration or test labels during fitting.

## Calibration methods

All intervals target 80% marginal coverage.

- **Standard split conformal:** one absolute-residual radius pooled across calibration variants.
- **Mondrian substitution conformal:** separate radii for coarse reference-to-alternate biochemical
  classes, with a pooled fallback when a group has fewer than 20 calibration examples.
- **Position-distance scaled conformal:** calibration residuals are normalized and test widths are
  scaled by the square root of distance to the nearest fitted residue position.

The latter two are empirical comparators. In particular, position-distance scaling is a transparent
heuristic and has no claimed distribution-free coverage guarantee under biological shift.

Reported interval diagnostics include mutation-level marginal coverage, coverage after assigning
each test position equal weight, the 10th percentile and minimum position coverage, and mean and
median interval width. Risk-coverage curves retain the 25%, 50%, 75%, and 100% least-uncertain
predictions and report normalized mean absolute error.

## Selection reliability

For each assay/split/model, VariantShift reports recovery of the experimentally best 10% of variants,
the standardized gain in experimental score among selected variants, and regret relative to the best
measured variant. These metrics test the protein-engineering use case more directly than a global
rank correlation alone.

## Held-out-protein and sequence-family transfer

Cross-protein models receive only assay-independent mutation descriptors and fixed ProteinGym
ESM-1v/ESM-2 scores. Targets are converted to within-assay percentile ranks so unrelated assay
units cannot leak into the pooled objective. At most 1,000 variants per assay are chosen by a
deterministic, label-independent ordering, and sample weights give every assay equal total weight.

Two outer five-fold protocols are reported. The first groups by UniProt ID. The second groups by a
deterministic sequence-family proxy constructed as follows:

1. Extract the exact ProteinGym `MSA_start:MSA_end` assayed segment for every eligible assay.
2. Run MMseqs2 exhaustive all-versus-all search with a broad 15% identity and 50% bidirectional
   coverage floor, retaining the complete alignment audit.
3. Add a homology edge at ≥30% sequence identity and ≥80% coverage of both sequences.
4. Form connected components across UniProt IDs. If any assay segment connects two proteins, all
   assays for both proteins inherit the same family ID.
5. Verify that zero qualifying alignments cross the resulting components. Report cluster
   composition at 20%, 25%, 30%, and 40% identity and at 50% and 80% coverage.

Within each outer training fold, a further 20% of groups are reserved for calibration. Consequently,
fit, calibration, and test proteins—or complete sequence-family components—are disjoint. Evaluated
models are a standardized ridge regression and a histogram gradient boosting regressor. The latter
supplies a nonlinear baseline without claiming equivalence to a protein language model.

The family rule is an explicit sequence-homology proxy, not a curated Pfam or structure-based family
definition. Remote homologs below the selected thresholds may remain separated. Fixed ESM scores
may also encode information from related pretraining sequences; the evaluation isolates assay labels,
not language-model pretraining corpora.

## Supervised-versus-zero-shot crossover

For each assay and random-position seed, the target is whether the supervised model's test Spearman
exceeds the fixed ESM-2 650M score on exactly the same held-out variants. Predictor features are
computed only from the outer training partition and include sample size, position coverage, target
dispersion, site-mean and within-site variability, and zero-shot training performance.

Logistic and histogram-gradient-boosting classifiers use fixed hyperparameters. All reported
probabilities are five-fold out-of-fold predictions grouped by UniProt ID, so every tested protein is
absent from the corresponding training fold. Repeated seeds measure split sensitivity and are not
treated as independent biological replicates.

## Interpretation boundaries

- Official and local model results have different training provenance and are labeled separately.
- Protein-level bootstrap intervals describe variation across the audited benchmark, not all
  possible proteins or wet-lab outcomes.
- Split seeds are not biological replication.
- Marginal coverage can conceal poor coverage for individual positions, which is why both are
  reported.
- A model-selection classifier can be useful without identifying a causal mechanism.
- The sequence-family split excludes directly detected homologous assay segments at the declared
  thresholds; it does not prove the absence of all evolutionary or structural relationships.
- No result establishes clinical validity or prospective protein-design success.

## Primary sources

- [ProteinGym repository and v1.3 resources](https://github.com/OATML-Markslab/ProteinGym)
- [ProteinGym benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)
- [ProteinNPT paper](https://openreview.net/pdf?id=AwzbQVuDBk)
- [Kermut paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf)
- [Kermut reference implementation](https://github.com/petergroth/kermut)
- [ESM reference implementation](https://github.com/facebookresearch/esm)
- [MMseqs2 reference implementation](https://github.com/soedinglab/MMseqs2)
- [MMseqs2 user guide](https://mmseqs.com/latest/userguide.pdf)
- [MMseqs2 methods paper](https://www.nature.com/articles/nbt.3988)
