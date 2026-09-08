# Selective Transport Auditor

## Statistical target

For an outcome-free task descriptor `x`, a candidate model `m`, and standardized top-decile
selection gain `Y(x, m)`, the auditor must choose one model and may abstain. The confirmation
sampling target is explicitly two stage:

1. sample a previously unseen protein family; and
2. sample a task within that family.

The primary loss at coverage `c` is selection regret,

`R(c) = E[max_m Y(x, m) - Y(x, selected(x)) | confidence(x) is in the top c fraction]`.

The primary endpoint is the trapezoidal area under `R(c)` for coverages 0.1 through 1.0. A lower
value is better. Non-positive selected gain, mean selected gain, and the worst-quintile mean gain
are safety and utility endpoints. They remain reported at every coverage but are not substituted
for the primary endpoint after outcomes are revealed.

## Why the original score was replaced

The original implementation maximized a conformal lower bound over models. Its calibration score
was the worst task-model residual in each family. This combined two different decisions:

- which model has the highest expected utility; and
- whether evidence is strong enough to deploy any model.

On the development panel, family-maximum calibration achieved 98.89% coverage at a nominal 90%
level and changed otherwise useful model choices. Binary deployment failure was also sparse: the
best average model failed on only one of 195 tasks. Optimizing failure AUC in that setting makes the
primary result depend on one or two observations.

The replacement was specified and evaluated using development outcomes only. Confirmation
outcomes remain locked.

## Decision rule

A histogram gradient-boosting regressor predicts task-model gain from outcome-free descriptors.
For each training partition, let `mu_hat(x, m)` be its prediction and `mu_bar(m)` be the mean gain
of model `m` in that same partition. The decision score is

`d_lambda(x, m) = lambda * mu_hat(x, m) + (1 - lambda) * mu_bar(m)`.

The final shrinkage candidate set is `{0.5, 0.75, 1}`. The auditor defaults to the model
with the best mean gain in the fit families. It overrides that model only when another model's
decision score exceeds it by a margin in `{0.05, 0.075, 0.1, 0.15}`. Task priority
is a fixed standardized support score: `1.0 × log(MSA Neff) + 0.5 × log(protein length) +
0.25 × selected decision score`. All three inputs are available before assay outcomes are opened.
Missing MSA or length values contribute zero after development-frozen centering, so missingness is
neutral rather than silently promoted or rejected.

Within each outer family-held-out fold, four-fold inner family predictions choose shrinkage and the
override margin by regret–coverage AUC, with mean regret as the first tie-break. The priority
weights and reduced policy grid were fixed before confirmation. The chosen rule is refit on the
outer fit families. No outer test or calibration outcome enters model hyperparameters or policy
selection.

The deployed model is the conservative override decision. Uncertainty never changes that choice.

## Selection-aware hierarchical calibration

A second family-cross-fitted regressor estimates the absolute error scale `s_hat(x, m)` of the
shrunken decision score. Calibration families are disjoint from model-fit families. On each
calibration task, the frozen decision rule first selects one model without reading its outcome. Its
one-sided normalized nonconformity score is

`A_ij = (d_lambda(x_ij, selected_ij) - Y_ij) / s_hat(x_ij, selected_ij)`,

where `i` indexes families and `j` indexes tasks within family.

For `n` calibration families with `K_i` tasks, the hierarchical empirical distribution assigns
mass `1 / ((n + 1) K_i)` to each `A_ij` and mass `1 / (n + 1)` to positive infinity. Its nominal
coverage quantile `q` gives the lower bound

`L(x) = d_lambda(x, selected(x)) - q * s_hat(x, selected(x))`.

Tasks are ranked by the frozen outcome-free priority score for risk–coverage analysis. The
conformal lower bound is an independent safety certificate, and deployment is trusted only when
`L(x) > 0`. Under two-level hierarchical exchangeability, this construction targets marginal
coverage for a new task drawn from a new family. It does not guarantee conditional coverage for a
particular family, simultaneous coverage of every task in a family, or validity under arbitrary
dataset shift.

## Evaluation discipline

- Outer `GroupKFold` partitions hold out complete curated families.
- Hyperparameter and shrinkage selection occurs only in inner family folds.
- Calibration families are disjoint from fit and outer test families.
- Assays receive equal weight in point estimates.
- Uncertainty resamples families, then proteins, then assays 10,000 times.
- The development-selected comparator is the label-free policy with the smallest regret–coverage
  AUC, not the policy with the most favorable confirmation result.
- Confirmation comparisons use the same frozen model, feature list, shrinkage grid, confidence
  threshold semantics, and bootstrap code.
- Formal coverage is assessed on the selected task-model pair. Marginal coverage over every
  unselected model is descriptive and is not a claimed guarantee.
- Development bootstrap intervals measure resampling stability, not confirmatory evidence,
  because the method and fixed priority weights were chosen during development.

## Required sensitivity analyses

The confirmation report must include the raw uncalibrated regressor, the shrunken decision rule
without conformal abstention, elastic net, best-average model, MSA depth, score dispersion,
ensemble agreement, random selection, the oracle, and the four frozen feature ablations. The
supervised-versus-zero-shot crossover score is reported only as an outcome-informed development
diagnostic because its inputs require within-assay labels. Results must be stratified by panel, exposure status,
exact-sequence novelty, sequence family, structure family, Pfam clan, assay modality, and documented
training cutoff. Missing novelty annotations remain `undocumented`.

## Methodological foundations

- Romano, Patterson and Candès, [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222),
  NeurIPS 2019.
- Tibshirani, Barber, Candès and Ramdas,
  [Conformal Prediction Under Covariate Shift](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html),
  NeurIPS 2019.
- Angelopoulos et al., [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg),
  ICLR 2024.
- Lee, Barber and Willett, *Distribution-free inference with hierarchical data*, ACM Journal of
  Data Science, 2026, DOI `10.1145/3786352`.
- Hu et al.,
  [Towards Reliable Model Selection for Unsupervised Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f50cebc22663df45ce619645bfabb3b3-Abstract-Datasets_and_Benchmarks_Track.html),
  NeurIPS 2024 Datasets and Benchmarks.
