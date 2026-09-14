# Conservative Auditor v2 development report

Decision date: 2026-09-01

## What changed

The failed pilot selector has been replaced by a conservative deployment policy. VespaG is the
only model it may deploy. The auditor uses experimental-outcome-free task descriptors and the
score distributions of six executed models to rank tasks for either VespaG deployment or
abstention; it cannot switch to a different model.

The confidence score is fixed as the harmonic mean of:

1. the family-cross-fitted percentile of predicted VespaG top-decile selection gain; and
2. one minus the family-cross-fitted percentile of predicted regret relative to the best model in
   the six-model panel.

The utility regressor uses protein length, sequence identity to development data, structure
similarity, domain coverage, taxon, assay modality, and structure availability. The regret
regressor uses task-wide score dispersion, tail separation, missingness, and model agreement or
disagreement. Both are Extra Trees models trained with panel-balanced task weights. The primary
comparator is the analytical expectation of always deploying VespaG with random task abstention.
Empirical percentile inputs are rounded to 12 decimal places so submachine floating-point noise
cannot change decisions at tied tree predictions.

## Development evidence

The analysis includes 231 complete six-model tasks in 150 families: 195 ProteinGym tasks, 21
MaveDB pilot tasks, and 15 VenusMutHub pilot tasks. All external pilot outcomes are development
data. No remaining confirmation outcome was requested by the v2 fitting or scoring code.

Five-fold family-held-out predictions and 10,000 family → protein → assay bootstrap replicates
gave:

| Endpoint versus always VespaG | Point estimate | 95% interval |
| --- | ---: | ---: |
| Regret–coverage AUC improvement | +0.0230 | +0.0067 to +0.0425 |
| Regret improvement at 50% coverage | +0.0232 | +0.0023 to +0.0502 |
| Failure-risk AUC improvement | +0.0037 | 0.0000 to +0.0173 |
| Mean selection-gain change at 50% coverage | +0.0039 | −0.0395 to +0.0478 |

This passes the prespecified development screen: the regret interval excludes zero, failure risk
does not worsen, and the point estimate for mean utility at 50% coverage is nonnegative. It is not
confirmatory evidence because the policy was redesigned using these data.

## Adversarial stress test

Leave-one-panel-out transport does not pass. Regret–coverage improvement was −0.0034 when MaveDB
was held out, +0.0085 when ProteinGym was held out, and −0.1050 when VenusMutHub was held out. The
Venus holdout also worsened failure-risk AUC by 0.0865. This small, heterogeneous stress test shows
that cross-panel transport remains unproven and prevents a claim that the method is already ready
for a top-journal submission.

## Freeze and claim boundary

The fitted model, input hashes, cross-fitted predictions, nested candidate audits, bootstrap
replicates, panel stress tests, and artifact hashes are frozen under
`results/conservative-auditor-v2/`. The confirmation scorer rejects outcome columns before model
execution and emits a hash manifest. An independent refit produced identical cross-fit confidence
values, confirmation confidence values, and bootstrap summaries (rank correlation 1.000; maximum
absolute difference 0). The only defensible next scientific action is a preregistered, one-shot
test on untouched outcomes in which cross-panel failure is accepted as a publishable negative
result. No feature, threshold, model, or exclusion may change after that reveal.
