# Transport-method development audit

All entries below use ProteinGym development outcomes. No confirmation outcome was opened. The
purpose of this table is to retain rejected methods and prevent the final method from appearing to
have been specified without iteration.

| Stage | Model choice | Coverage ranking | Regret–coverage AUC | Comparator AUC | Calibration | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Original score | Largest family-max lower bound | Same lower bound | 0.10643 | 0.07133 for mean-choice sensitivity | 98.89% | Reject: conservative bound damaged model choice |
| Selection-aware hierarchical calibration | Nested shrunken expected gain | Hierarchical lower bound | 0.08263 | 0.06783 | 90.77% | Reject: calibrated but weak ranking |
| Broad conservative-policy grid | Nested conservative override | Re-tuned crossover/gain mixture | 0.06823 | 0.06783 | 89.23% | Reject: 175 candidates produced unstable fold choices |
| Reduced crossover-priority grid | Nested conservative override | Fixed crossover/gain mixture | 0.06095 | 0.06783 | 89.23% | Reject: priority used an outcome-informed same-assay diagnostic |
| Outcome-free selective auditor | Nested conservative override | Fixed MSA depth, length, and decision score | 0.06452 | 0.07273 for MSA depth | 89.23% | Freeze for confirmation |

For the frozen outcome-free method, the comparator-minus-auditor regret–coverage improvement is
0.00821. The 10,000-repeat nested family/protein/assay bootstrap interval is 0.00214 to 0.01494,
with 99.42% of replicates above zero. Because the method itself was adaptively developed on this
dataset, that interval describes resampling stability rather than a confirmatory claim. Only the
locked external panels can establish superiority.

Compact machine-readable summaries for rejected replacement stages are retained under
`results/transport-method-development/`, including the apparently stronger but invalid
crossover-priority method. The original method remains recoverable from version control, and the
frozen outcome-free artifacts occupy `results/transport-v1/`.

The crossover probability was originally described as label-free, but its feature builder used
same-assay outcome summaries. It is therefore an outcome-informed development diagnostic, not a
deployable task property. The confirmation predictor now rejects any input column containing that
diagnostic, and the confirmation evaluator has no crossover-policy path.
