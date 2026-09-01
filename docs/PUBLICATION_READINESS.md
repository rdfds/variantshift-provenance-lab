# Publication readiness audit

Status is based on executable evidence in this repository, not the intended study design.

## 2026-09-01 selector redesign decision

The original Selective Transport Auditor remains a failed pilot. Its replacement, Conservative
Auditor v2, passes the family-held-out development screen against always using VespaG: the
regret–coverage improvement is +0.0230 with a hierarchical 95% interval of +0.0067 to +0.0425,
failure risk does not worsen, and the 50%-coverage utility point estimate is nonnegative. The
replacement does **not** pass leave-one-panel-out transport, so it is a frozen confirmation
candidate rather than top-publication evidence. See `docs/CONSERVATIVE_AUDITOR_V2.md`.

| Requirement | Status | Evidence or blocker |
| --- | --- | --- |
| 195-assay ProteinGym development layer | Complete | Audited results under `results/proteingym/` |
| Previously revealed MaveDB development layer | Complete | 45 downloaded, 21 direction-eligible assays |
| Stable public schemas | Complete | Seven versioned table contracts and tests |
| Outcome-access firewall | Complete | One-way lock, hash checks, and leakage tests |
| Task-level transport pipeline | Complete | Fixed VespaG-or-abstain policy, family cross-fitting, analytical fixed-model comparator, outcome firewall, and 10,000 hierarchical bootstrap replicates |
| Development scientific gate | Passed descriptively, not confirmatory | Regret–coverage improvement +0.0230 versus always VespaG; family-bootstrap 95% interval +0.0067 to +0.0425; failure risk not worse |
| Untouched MaveDB target freeze | Complete | 23 score sets, 8 sequences, zero score requests |
| Human Domainome target freeze | Complete | 426 unique domains extracted from a checksum-pinned Zenodo supplement; only `dom_ID`, `PFAM_ID`, and `wt_seq` were decoded, while all trailing fields were hashed but discarded |
| VenusMutHub target freeze | Complete | 126 assays, 91 unique sequences, zero mutation-file requests |
| Exact/sequence-family confirmation audit | Complete | 525 targets in 282 confirmation families; 523 exact-sequence-unseen and 467 MMseqs2-family-unseen relative to development |
| Structure/Pfam confirmation audit | Complete with explicit missingness | Pfam-clan and Foldseek novelty are recorded; unavailable annotations remain `undocumented`, never relabeled clean |
| Ten executable model configurations | Passed | Ten configurations passed the frozen qualification audit |
| Six model/input families | Passed | Masked PLM, inverse-folding, structure-aware, autoregressive, convolutional, and distilled-evolutionary configurations qualify |
| 300 shared confirmation targets | Passed | All ten qualified configurations share 413 Domainome targets at at least 95% substitution coverage |
| Formal model qualification | Passed | Coverage, official-score parity, zero-cache determinism, checkpoint/container/input/prediction hashes, runtime, hardware, and failure evidence pass |
| External development pilot | Current method failed | 36 usable tasks; VariantShift vs always VespaG regret-AUC improvement −0.0674 (95% interval −0.2040 to +0.1565) |
| Conservative Auditor v2 | Family gate passed; panel stress test failed | Fixed VespaG-or-abstain policy; regret-AUC improvement +0.0230 (95% interval +0.0067 to +0.0425), but leave-one-panel-out transport is inconsistent |
| OSF preregistration | Approved under embargo; literal public-timestamp gate remains pending | V2 protocol, 10-model Domainome panel, 6-model auditor inputs, frozen confidence ranks, code snapshot, checksums, and negative-result rule are registered at `https://osf.io/axy7k/`; the record remains private while embargoed |
| Confirmation outcome reveal | Original panels locked | Domainome and the nonpilot Venus holdout remain untouched; only the explicitly labeled development pilot was revealed |
| Conservative Auditor v2 evaluator | Implemented after registration; publicly timestamped but not run | Dedicated hash-verified Domainome/Venus evaluator, frozen VespaG-or-abstain ranks, task-metric audit, 10,000 hierarchical replicates, Holm reporting, and the four registered scientific gates; immutable external archiving remains required before reveal |
| Zenodo 1.0 DOI | Pending | Create only after final immutable release candidate |

## Venue interpretation

No top-journal submission is justified at the current state. The software and development study are
substantial, but the central selector has not passed an untouched confirmation and the qualified
confirmation/model panel has not been evaluated against outcomes. If the
declared confirmation gates pass, a Nature Methods Analysis becomes plausible but remains highly
selective. Strong broad external findings with a
less decisive method fit Nature Communications; a human-genomics-centered result may fit Genome
Biology. A rigorous negative benchmark remains suitable for Bioinformatics or PLOS Computational
Biology.
