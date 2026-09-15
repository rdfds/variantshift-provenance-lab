# Conservative Auditor v2 confirmation evaluator

Status: **implemented after OSF registration but before confirmation outcome access**

The implementation entered the public history in commit `98d18d1`; the frozen development
candidate and audit artifacts entered in commit `d9c732b`. This provides a content-addressed public
timestamp, not a substitute for the planned immutable external archive.

This evaluator is a mechanical implementation of the analysis registered under
`variantshift-confirmation-freeze-v2`. It does not alter the frozen model panel, VespaG baseline,
auditor confidence values, task ranks, coverage grid, inclusion rules, or acceptance thresholds.
The registered protocol remains authoritative if an implementation detail conflicts with it.

The command is deliberately unavailable until the aggregate outcome lock is `revealed` and every
frozen prediction and method artifact still matches its registered hash:

```bash
variantshift conservative-auditor-evaluate \
  configs/conservative-auditor-v2.json \
  configs/final-freeze-v2.json \
  results/confirmation-preregistration-v2/auditor/frozen-confirmation-decisions.csv \
  results/confirmation-preregistration-v2/selected-predictions/selected-prediction-registry.csv \
  results/confirmation-reveal-v2/confirmation-outcomes.csv.gz \
  results/confirmation-preregistration-v2/immutable-freeze/outcome-lock.json \
  --output-dir results/confirmation-evaluation-v2
```

## Frozen interpretation

- Confirmation panels are exactly Human Domainome and the untouched VenusMutHub holdout.
- Conservative Auditor v2 may deploy VespaG or abstain; it never selects another model.
- Pooled analyses use the frozen pooled confidence ranks. Panel-direction analyses use the frozen
  within-panel ranks.
- Selection regret is computed against the best available model in the registered six-model
  auditor panel. The ten-model Domainome comparison remains a secondary benchmark table.
- Domainome task-model rows require at least ten aligned substitutions and at least 95% alignment
  to available outcome variants. Venus rows require at least ten aligned substitutions.
- Missing task-model rows are retained in an explicit audit and are never imputed.
- The primary comparator is analytical expected performance from always deploying VespaG with
  random task abstention at each coverage.
- Uncertainty uses 10,000 deterministic family → protein → assay bootstrap replicates.
- The primary regret-coverage interval is evaluated first. Holm-adjusted values are reported for
  the gated secondary and panel-specific tests.

## Acceptance gates

1. The pooled regret-coverage improvement has a family-bootstrap 95% interval with lower bound
   strictly above zero.
2. The pooled failure-risk improvement has a 95% interval with lower bound at least zero.
3. Mean selection-gain change at 50% task coverage is nonnegative.
4. Regret-coverage improvement is positive in both Domainome and untouched VenusMutHub.
5. No model, feature, threshold, confidence rank, or exclusion rule is refit after reveal.

If any scientific gate fails, the evaluator emits `status: fail`. It does not request or accept a
post hoc narrative to turn a failed confirmation into a positive method claim.

Before outcome access, this implementation and its tests must be deposited in an immutable
external archive or attached through a transparent registration update because they were completed
after the original OSF submission.

## Registered one-time reveal

After the linked evaluator addendum is approved and its receipt and archive are bound into the
aggregate lock, retrieve only the two panels named above:

```bash
variantshift confirmation-outcomes-retrieve \
  results/confirmation-preregistration-v2/immutable-freeze/outcome-lock.json \
  results/confirmation-preregistration-v2/tasks/untouched-confirmation-task-registry.csv \
  results/confirmation/domainome-v1/targets.csv \
  results/confirmation/domainome-v1/variants.csv \
  protocols/venusmuthub-v1/frozen/targets.csv \
  protocols/venusmuthub-v1/assay-audit.csv \
  protocols/venusmuthub-v1/target-freeze-protocol.json \
  --output-dir results/confirmation-reveal-v2

variantshift confirmation-reveal \
  results/confirmation-preregistration-v2/immutable-freeze/outcome-lock.json \
  --outcome results/confirmation-reveal-v2/confirmation-outcomes.csv.gz \
  --outcome results/confirmation-reveal-v2/outcome-parsing-audit.csv \
  --outcome results/confirmation-reveal-v2/outcome-access-ledger.json
```

The retriever requires the lock to be exactly `registered`, validates the frozen target inputs,
has no MaveDB score-access path, and records a source URL, SHA-256 digest, byte count, and UTC
access timestamp for every downloaded artifact. A partially interrupted run reuses only raw files
whose adjacent receipt still matches, preventing a second network retrieval from being hidden.
