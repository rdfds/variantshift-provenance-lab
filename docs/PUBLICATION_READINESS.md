# Publication readiness audit

Status is based on executable evidence in this repository, not the intended study design.

| Requirement | Status | Evidence or blocker |
| --- | --- | --- |
| 195-assay ProteinGym development layer | Complete | Audited results under `results/proteingym/` |
| Previously revealed MaveDB development layer | Complete | 45 downloaded, 21 direction-eligible assays |
| Stable public schemas | Complete | Seven versioned table contracts and tests |
| Outcome-access firewall | Complete | One-way lock, hash checks, and leakage tests |
| Task-level transport pipeline | Complete | Nested family selection, hierarchical conformal calibration, frozen comparators/ablations, 10,000 nested bootstrap |
| Development scientific gate | Passed descriptively, not confirmatory | Outcome-free regret–coverage AUC 0.06452 versus 0.07273; adaptive-development bootstrap interval for improvement 0.00214 to 0.01494 |
| Untouched MaveDB target freeze | Complete | 23 score sets, 8 sequences, zero score requests |
| Human Domainome target freeze | Blocked | No public retained-domain target-only manifest identified |
| VenusMutHub target freeze | Complete | 126 assays, 91 unique sequences, zero mutation-file requests |
| Exact/sequence-family confirmation audit | Complete | 99 targets, 96 confirmation families, 97 exact-unseen, 96 family-unseen |
| Structure/Pfam confirmation audit | Pending | Missing annotations are explicitly `undocumented`, not clean |
| Eight executable model configurations | Failed | Metadata audit only; zero configurations have executable parity evidence |
| Four model/input families | Pending | Configuration exists, execution evidence does not |
| 300 shared confirmation targets | Pending | MaveDB plus VenusMutHub provide 99; Domainome remains blocked |
| OSF preregistration | Intentionally pending | Command refuses to build until the 8-model/4-family/300-target execution gate passes |
| Confirmation outcome reveal | Locked | Prohibited before public registration |
| Confirmation acceptance report | Implemented, not run | Hash-verified evaluation, hierarchical intervals, Holm correction, panel directions, ablations, six gates |
| Zenodo 1.0 DOI | Pending | Create only after final immutable release candidate |

## Venue interpretation

No top-journal submission is justified at the current state. The software and development study are
substantial, but the central selector has not passed an untouched confirmation and the broad
confirmation/model panel has not run. If the declared confirmation gates pass, a Nature Methods
Analysis becomes plausible but remains highly selective. Strong broad external findings with a
less decisive method fit Nature Communications; a human-genomics-centered result may fit Genome
Biology. A rigorous negative benchmark remains suitable for Bioinformatics or PLOS Computational
Biology.
