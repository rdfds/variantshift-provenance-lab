# VariantShift outcome-blind transport analysis

## Claim boundary

The planned paper is an Analysis of benchmark transport and deployment reliability. It does not
claim successful protein design or prospective experimental validation.

Primary claim:

> Standard variant-effect benchmarks do not reliably estimate performance on new proteins and
> assays. VariantShift measures that transport failure and uses outcome-free task properties to
> choose a model or abstain before measurements are available.

ProteinGym v1.3 and the previously revealed 45-score-set MaveDB run are development data. Human
Domainome, the untouched MaveDB complement, and VenusMutHub are confirmation panels only after
their targets, complete 19L prediction landscapes, fitted method, protocol, and public
registration are frozen.

## Primary questions and estimands

1. How do performance and model rank change from random variants to unseen positions, proteins,
   families, assay modalities, and datasets?
2. Can outcome-free task and model descriptors predict standardized top-decile selection gain?
3. Can model selection or abstention reduce failed deployments?

The unit of analysis is a task, not a variant. Each assay receives equal weight.

- Primary utility: standardized selection gain in the model-ranked top decile.
- Primary failure: selection gain less than or equal to zero.
- Primary reliability endpoint: area under task-level failure risk–coverage curve.
- Secondary metrics: Spearman correlation, top-decile recall, nDCG, best-variant regret, model-rank
  stability, marginal conformal coverage, and position-conditional coverage.
- Uncertainty: 10,000 nested bootstrap resamples of families, proteins within families, and assays
  within proteins.

## Development layer

The current development table has 2,340 task–model records: 195 ProteinGym assays, 169 proteins,
132 curated families, and 12 exactly paired official score columns. Model-score descriptors are
computed from prediction columns only. The extractor never reads `DMS_score` from the prediction
archive; task-level audited selection gain is joined separately as the development target.

Outer folds hold out complete curated families. Within each outer training fold, fit and conformal
calibration families are disjoint. Hyperparameters are selected only inside held-out inner family
folds. The deployable bundle uses a separate group calibration set and records its training-frame
hash.

The current pilot is a negative development result. VariantShift risk–coverage AUC is 0.00987,
versus 0.00026 for the development-selected elastic-net comparator, and the nested-bootstrap
improvement estimate is −0.00961 (95% interval −0.02952 to 0.00174). Observed cross-fitted
lower-bound coverage is 98.89%, indicating over-conservative calibration. The method must improve
on development data before it is frozen for confirmation. The negative result is retained rather
than hidden by changing the outcome, comparator, or confirmation threshold.

## Confirmation firewall

Every panel has an `outcome-lock.json` with one-way states:

1. `targets_frozen`
2. `predictions_frozen`
3. `registered`
4. `revealed`

The transition to `predictions_frozen` hashes all model predictions and method artifacts. The
registration transition requires a public HTTP(S) record. Confirmation evaluation refuses to run
before registration. A reveal records the outcome artifact hashes and cannot be repeated.

The MaveDB complement was selected using registry and detailed metadata only. No score endpoint was
requested. Of 705 target-only candidates, 23 score sets covering 8 unique sequences passed the
strict pre-outcome rules; 94,981 possible substitutions are frozen. At reveal, the already-declared
minimum of 100 single substitutions and 10 assayed positions is applied without changing direction.

Human Domainome cannot yet be frozen strictly outcome-blind because its public target manifest does
not identify the retained 522 domain sequences separately from Supplementary Table 2, which also
contains the measurements. The repository will not download that mixed file before predictions.
An author-provided target-only manifest or a genuinely sealed extraction process is required.

VenusMutHub was frozen through file-tree metadata, its separate DOI table, and RCSB/UniProt target
sequence APIs. No mutation CSV was requested. The audit retained 126 assays across 91 unique
sequences after exact development DOI and sequence exclusions, then enumerated 737,713 possible
substitutions. Files without a DOI or an unambiguous target sequence remain visible exclusions.

An outcome-free combined audit groups the 99 currently frozen MaveDB and VenusMutHub targets into
96 MMseqs2 families. Ninety-seven targets are exact-sequence unseen and 96 are sequence-family
unseen relative to ProteinGym under the fixed 30% identity and 80% bidirectional-coverage rule.
Foldseek structure-family and Pfam-clan status remain `undocumented` because those confirmation
annotations have not been generated; missing data are never promoted to clean status.

## Modern model panel gate

The planned configuration records ESM-1v ensemble, ESM-2 650M, ESM-C 600M, ESM3-open, ESM-IF1,
ProteinMPNN, Tranception-L with and without frozen retrieval, GEMME, SaProt 650M, ProSST-2048,
VenusREM, and VespaG. Label-using ProteinNPT and Kermut remain a separate supervised/few-shot regime.

A primary configuration must:

- have reviewable academic code, weights, and license terms;
- run from a recorded container on ARCH;
- score at least 95% of eligible substitutions;
- reproduce matching official ProteinGym rankings at Spearman at least 0.99;
- repeat at rank correlation at least 0.999; and
- record checkpoint, container, target, prediction, MSA, and structure hashes.

The study proceeds to confirmation only with at least eight executable configurations across four
materially different input/model families and at least 300 common targets on which every model
covers at least 95% of substitutions. `preregistration-build` now enforces this gate and refuses to
produce a registration bundle when the executable audit is insufficient.

## Frozen Transportability Score

The candidate implementation is a histogram gradient-boosting regressor over task and model
descriptors. It predicts selection gain and group-held-out absolute error for each task–model pair.
A one-sided lower bound uses family-max conformal scores normalized by the predicted error scale.
The model with the largest lower bound is selected only when that bound exceeds zero. The frozen
bundle also contains the development-selected elastic-net comparator and MSA-only,
ensemble-only, without-MSA, and without-ensemble ablations.

Comparators are uncalibrated transport prediction, elastic-net task regression, MSA depth, model
score dispersion, ensemble agreement, the existing crossover classifier, random selection, the
best average development model, and an unattainable oracle.

Formal conformal claims are restricted to exchangeable family-level tasks. Coverage under stronger
dataset, structure-family, Pfam-clan, and temporal shifts is an empirical diagnostic.

## Confirmation success gates

- The best frozen label-free comparator is beaten on pooled confirmation risk–coverage AUC, with a
  family-bootstrap 95% interval excluding zero.
- At 50% task coverage, failed deployments fall by at least 25% without lower mean selection gain.
- Nominal 90% lower-bound coverage is between 85% and 95%.
- Effects agree in direction between Domainome and untouched MaveDB.
- Feature ablations rule out MSA depth or ensemble disagreement as the sole explanation.
- At least one robust negative result changes how ProteinGym results should be interpreted.

Failure of these gates is reported as failure; it does not authorize threshold refitting on
confirmation data.

`transport-evaluate` writes every gate to `confirmation-acceptance.json`, applies Holm adjustment
to the frozen policy-comparison family, reports panel-specific directions, and requires the exact
prediction, model, and revealed outcome hashes recorded in the one-way lock. A useful negative
conclusion remains a separately documented, evidence-linked requirement rather than an automated
claim generator.
