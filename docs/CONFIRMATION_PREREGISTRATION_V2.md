# Confirmation preregistration v2 readiness

Status: **approved on OSF under embargo; local outcome lock remains closed**

The complete outcome-blind confirmation package is under
`results/confirmation-preregistration-v2/`. No remaining experimental effect file was accessed.
The new aggregate outcome lock is `predictions_frozen`; its registration and reveal fields are
null.

## Frozen scope

- Ten qualified configurations from six model/input families on Domainome.
- 413 Domainome targets in the complete ten-model, at-least-95%-coverage benchmark intersection.
- Six auditor-input models on all 426 Domainome tasks and 76 untouched VenusMutHub tasks.
- 502 total frozen auditor tasks: 426 Domainome and 76 VenusMutHub.
- At 50% coverage, 251 pooled tasks are assigned to VespaG and 251 to abstention.
- Within-panel 50% coverage assigns 213 of 426 Domainome tasks and 38 of 76 Venus tasks to
  VespaG.

Two untouched MaveDB tasks and one untouched Venus task were excluded before outcomes because
VespaG predictions were absent. These rows are preserved in
`auditor/confirmation-features.exclusions.csv`; they were not imputed or performance-filtered.

## Integrity boundary

The immutable package contains 13 target/protocol artifacts, 96 prediction/provenance artifacts,
24 method inputs plus the final freeze manifest (25 method-lock entries), a deterministic code
snapshot, and the complete local registration record.
The primary comparator is always VespaG. Conservative Auditor v2 may deploy VespaG or abstain and
cannot override the model choice.

The registration was approved at `https://osf.io/axy7k/` under embargo. The private registration
bundle and ZIP are intentionally excluded from the public Git repository. The post-registration
confirmation evaluator is separately timestamped in the public commit history, but it still needs
an immutable external archive or transparent registration update before outcome access. The local
aggregate lock remains `predictions_frozen`; outcome access is prohibited until the approved
registration is recorded and the lock advances successfully.
