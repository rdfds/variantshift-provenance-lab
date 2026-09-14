# External development pilot: go/no-go decision

Decision date: 2026-09-01

## Decision

**No-go for confirming the current VariantShift Selective Transport Auditor.** The external
development pilot supports the broader claim that model reliability changes across task types,
but it does not support the present selector over the best fixed model. The remaining GPU budget
should not be spent completing the confirmation matrix until the selector is redesigned and
passes a new development gate.

This is development evidence, not confirmation. Human Domainome outcomes and the nonpilot Venus
holdout remain untouched.

The untouched holdout contains 426 Domainome tasks/targets, 77 Venus tasks on 58 targets, and two
MaveDB tasks on two targets that lacked complete six-model coverage. The frozen protocol's two
fields labeled `untouched_venus_*` accidentally aggregate the MaveDB and Venus rows; the immutable
partition table is correct and `post-freeze-count-correction.json` records the documentation fix.

## Frozen pilot

- Six executed models: CARP-640M, ESM-2 8M/35M/150M/650M, and VespaG.
- Fifty outcome-blind tasks on 30 shared-coverage targets were frozen before outcome access.
- Thirty-six tasks passed the frozen minimum of ten aligned substitutions: 21 MaveDB tasks and
  15 VenusMutHub tasks.
- Predictions were repeated deterministically under an exact runtime lock before reveal.
- The pilot uses 10,000 family → protein → assay bootstrap resamples.

The first Venus file exposed a target-coordinate mismatch. Automatic per-assay exclusion was then
added as a parser-operability fix. Three Venus assays were excluded for target-coordinate mismatch,
and 14 task-model groups (corresponding to small assays) failed the frozen ten-variant alignment
minimum. No observed effects or model performance were used to choose exclusions, but this parser
revision is another reason the exercise is classified only as development.

## Evidence

Performance changed materially between panels. On the 15 usable Venus tasks, model failure rates
(`selection_gain_sd <= 0`) were:

| Model | Failure rate | Mean selection gain (SD) |
| --- | ---: | ---: |
| VespaG | 6.7% | 0.679 |
| CARP-640M | 13.3% | 0.573 |
| ESM-2 650M | 26.7% | 0.486 |
| ESM-2 150M | 40.0% | 0.239 |
| ESM-2 8M | 40.0% | 0.158 |
| ESM-2 35M | 46.7% | 0.131 |

The frozen selector did not clear the methodological gate:

| Contrast | Regret-AUC improvement | 95% interval | One-sided p | Risk-AUC improvement |
| --- | ---: | ---: | ---: | ---: |
| VariantShift vs score dispersion | +0.0945 | −0.0712 to +0.2559 | 0.136 | −0.0361 |
| VariantShift vs always VespaG | −0.0674 | −0.2040 to +0.1565 | 0.444 | −0.0462 |

At 50% task coverage, VariantShift had a 5.6% failure rate and mean selection gain of 0.603 SD.
Always using VespaG had a 0% failure rate and mean selection gain of 0.509 SD. VariantShift traded
higher mean utility for worse failure protection and higher model-selection regret, which is the
opposite of the primary reliability claim.

## Required redesign gate

Before external scoring resumes, a replacement method must use only development data and satisfy
all of the following under protein/family-held-out evaluation:

1. Beat the best fixed model—not only a weak score-dispersion comparator—on regret–coverage AUC.
2. Have a hierarchical 95% interval for regret-AUC improvement that excludes zero.
3. Reduce or match failed-deployment risk at 50% coverage while retaining mean selection gain.
4. Demonstrate that gains persist after removing model identity and after leave-one-panel-out tests.
5. Freeze the replacement once; then test it on the untouched Venus tasks and Domainome without
   another method or threshold change.

If no method clears this gate, the defensible paper is a benchmark/negative-result analysis about
transport failure and benchmark saturation, not a paper claiming successful outcome-free model
selection.

## Auditable artifacts

- Pilot protocol: `protocols/external-development-pilot-v1/pilot-protocol.json`
- Prediction/method lock: `protocols/external-development-pilot-v1/pilot-outcome-lock.json`
- Outcome-access ledger: `data/raw/external-development-pilot-v1/pilot-outcome-access-ledger.json`
- Model/task metrics: `results/external-development-pilot-v1/revealed-metrics/pilot-task-metrics.csv`
- Primary policy summary: `results/external-development-pilot-v1/evaluation-primary/pilot-policy-summary.csv`
- Hierarchical intervals: `results/external-development-pilot-v1/evaluation-primary/pilot-primary-bootstrap-summary.csv`
