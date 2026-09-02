# VariantShift: audited generalization under biological shift

## Abstract

Variant-effect benchmarks often mix three distinct questions: interpolation among measured
mutations, extrapolation to unmeasured residue positions, and transfer to unseen proteins. This
study separates those estimands on ProteinGym v1.3, audits official supervised and zero-shot
predictions at mutation level, and adds repeated protein-, sequence-family-, structure-family-, and
curated Pfam-family holdouts. The final cohort contains 195 assays, 169 proteins, and 689,994
single-substitution measurements. Across twelve exactly paired zero-shot score sets, VenusREM ranks
first at protein-balanced mean Spearman 0.542 and leads ESM-2 650M by 0.114; a simultaneous 95%
paired-bootstrap interval across all model comparisons is 0.081–0.146. Five repeated grouped
partitions show a small decrease from protein holdout to sequence-family holdout, but adding remote
structure and curated Pfam relationships does not produce an additional resolved decrease. The
feature ablation shows that fixed ESM scores account for most of the pooled models' advantage over
mutation descriptors alone. The results support a narrow conclusion: detectable homologous-label
transfer slightly inflates the pooled estimate, while a largely pretrained-score-driven ranking
signal is stable across the declared family rules.

## Study design

The eligibility ledger is fixed independently of model performance. Every included assay must have
a consistent wild-type sequence and mutation identifier, at least 500 finite single substitutions,
at least 20 measured positions, and at least 10 distinct normalized scores. All reported aggregate
means first average assays within UniProt ID so proteins with multiple assays do not receive extra
weight.

The analysis has four linked layers:

1. Official supervised out-of-fold predictions under random, modulo-position, and contiguous-
   position splits.
2. A local frozen ESM-2 residue probe with standard, Mondrian, and position-distance-scaled
   conformal intervals.
3. An exactly paired twelve-model zero-shot landscape with ranking and top-variant metrics.
4. Pooled ridge and nonlinear transfer under five repeated group partitions, where groups progress
   from UniProt IDs to sequence, sequence/structure, and independently curated Pfam families.

## Modern zero-shot landscape

All 195 eligible assays have one-to-one mutation joins and 100% common finite coverage across the
twelve declared score columns. The model ordering is therefore calculated from identical variants,
not from model-specific subsets.

| Model | Modality | Mean Spearman | 95% protein-bootstrap interval |
| --- | --- | ---: | ---: |
| VenusREM | structure + MSA | **0.542** | 0.520–0.562 |
| ProSST 2048 | sequence + structure | 0.528 | 0.506–0.551 |
| S3F-MSA | structure + MSA | 0.508 | 0.487–0.528 |
| ESM3 | sequence + structure + function | 0.505 | 0.478–0.531 |
| SaProt 650M | sequence + structure | 0.470 | 0.443–0.497 |
| GEMME | MSA | 0.456 | 0.436–0.476 |
| ESM-2 650M | sequence | 0.428 | 0.398–0.457 |
| ESM-C 600M | sequence | 0.423 | 0.393–0.453 |
| Tranception-L | MSA retrieval | 0.422 | 0.401–0.443 |
| SiteRM | structure + MSA | 0.404 | 0.382–0.425 |
| ProGen3 3B | sequence | 0.389 | 0.368–0.409 |
| xTrimoPGLM 100B | sequence | 0.357 | 0.334–0.381 |

VenusREM ranks first in all 10,000 protein-bootstrap resamples. Against ESM-2 650M, its paired mean
difference is 0.114; the pointwise 95% interval is 0.091–0.139 and the simultaneous max-deviation
interval across all eleven comparisons is 0.081–0.146. ProSST's corresponding difference is 0.100,
with a simultaneous interval of 0.068–0.133. These intervals control the main multiplicity problem
created by comparing many models to the same baseline.

## Independent family definitions

The family audit intentionally combines independent evidence rather than relabeling one sequence
heuristic as ground truth.

| Grouping rule | Components | Multi-protein components | Proteins in multi-protein components | Largest component |
| --- | ---: | ---: | ---: | ---: |
| MMseqs2 assayed-segment sequence | 156 | 10 | 23 | 3 |
| Sequence + reciprocal Foldseek | 148 | 14 | 35 | 5 |
| Sequence/structure + exact Pfam family | 132 | 23 | 60 | 10 |
| Pfam clan sensitivity | 97 | 23 | 95 | 24 |

The curated layer maps ProteinGym assayed intervals onto one contiguous UniProt canonical segment,
then requires at least 80% exact coordinate coverage and at least 50% overlap with a Pfam domain.
The InterPro 109.0 / Pfam 38.2 snapshot maps 183 of 195 assays across 158 of 169 proteins. Unmapped
proteins retain their prior sequence/structure component. Exact Pfam families are primary; clans are
reported as a broader stress test because they can connect remote folds.

## Repeated grouped transfer

Each protocol uses five shuffled outer group partitions. A disjoint 20% of the outer-training groups
is reserved for calibration. Repeat-level means are published, and paired protocol intervals
resample whole components from the stricter alternative grouping.

| Held-out unit | Nonlinear HistGB | Ridge |
| --- | ---: | ---: |
| Protein | 0.539 | 0.515 |
| Sequence family | 0.533 | 0.513 |
| Sequence + structure family | 0.532 | 0.512 |
| Sequence + structure + Pfam family | 0.533 | 0.513 |
| Pfam clan sensitivity | 0.531 | 0.512 |

Sequence-family holdout reduces nonlinear Spearman by 0.0060 relative to protein holdout (95%
family-bootstrap interval −0.0124 to −0.0011) and ridge by 0.0021 (−0.0038 to −0.0007). This is a
small but resolved homologous-label-transfer effect. Relative to sequence-family holdout, adding
reciprocal Foldseek relationships changes nonlinear Spearman by −0.00045 (−0.00315 to 0.00235).
Adding exact Pfam relationships beyond sequence/structure changes it by +0.00068 (−0.00147 to
0.00294). The broad Pfam-clan stress test changes the exact-Pfam estimate by −0.00222 (−0.00397 to
0.00004). The declared remote-homology layers therefore do not resolve a further transfer penalty
after the sequence-family safeguard.

Across five repeats, nonlinear mean Spearman ranges from 0.536–0.541 under protein holdout and
0.531–0.536 under exact-Pfam holdout. Standard nominal-80% intervals attain mean marginal coverage
of 0.807 and 0.801 in those protocols. These are empirical coverage measurements under shift, not
distribution-free guarantees.

## Feature ablation under the strictest split

The pooled models combine transparent mutation descriptors with seven fixed ESM-1v/ESM-2 score
features. To separate learned descriptor transfer from pretrained priors, the exact-Pfam-family
protocol is repeated with the score columns removed and all partitions held fixed.

| Model | Full features | Mutation descriptors only | Paired difference | 95% family-bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| Nonlinear HistGB | 0.533 | 0.360 | +0.174 | 0.151–0.195 |
| Ridge | 0.513 | 0.318 | +0.195 | 0.169–0.219 |

Mutation descriptors alone retain positive cross-family rank correlation, especially for the
nonlinear model, but fixed ESM scores supply most of the final performance. The pooled result should
therefore be interpreted as a learned combination of pretrained variant scores and descriptors—not
as de novo transfer from assay labels alone. This ablation narrows the contribution while preserving
the main family-audit result: the same exact partitions are used for both feature sets.

## What the study establishes

- Model rankings are based on exactly paired mutation sets with audited target agreement.
- The strongest zero-shot results are not an artifact of proteins with multiple assays receiving
  extra weight.
- A small portion of pooled transfer performance is attributable to detectable close sequence
  relationships.
- Most of the pooled models' cross-family advantage over mutation-only baselines comes from fixed
  pretrained ESM scores rather than assay-label transfer alone.
- The remaining estimate is stable to reciprocal structure, exact Pfam, and broader Pfam-clan
  grouping at the declared thresholds.
- Marginal coverage, position-balanced coverage, risk–coverage, top-set recovery, and best-variant
  regret are all published rather than substituting one rank correlation for every use case.

## What remains outside the evidence

This is a retrospective computational benchmark. It does not establish prospective wet-lab hit
rates, clinical validity, causal mechanisms, or coverage over all protein families. ProteinGym's
official structures are predicted, the curated API snapshot does not map every protein, and the
official v1.3 score archive does not contain every model available or unpublished in 2026. A
top-tier empirical follow-up would preregister a new assay panel, freeze model and calibration
choices before measurements, and evaluate prospective selection enrichment on proteins excluded
from both model development and this benchmark.

## Reproducibility

Raw ProteinGym archives, structures, and per-variant predictions remain outside Git. Committed
aggregate artifacts include complete eligibility, join, family-edge, coordinate, repeat, and
bootstrap ledgers. A compressed public-data snapshot freezes every UniProt and InterPro response
used for curated mapping. The run manifest records public input hashes, exact source revision,
library and tool versions, protocol thresholds, and hashes for every committed output. CI re-runs
unit and leakage-invariant tests and verifies committed artifact bytes against that manifest.
