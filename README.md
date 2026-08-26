# VariantShift

**When does a protein mutation predictor actually generalize?**

VariantShift is a leakage-aware benchmark for protein variant-effect models. It compares
ordinary random splits with biologically harder tests: unseen residue positions, increased
mutational depth, and transfer between experimental conditions.

The initial case study uses the Align Foundation's TEV protease GROQ-seq release: 18,486
variants measured across 24 conditions at NIST's Living Measurement Systems Foundry.

## Research questions

1. How much does random splitting overstate performance?
2. Which models retain rank accuracy at residue positions absent from training?
3. Do single-mutant models transfer to combinatorial variants?
4. Which experimental conditions preserve variant rankings across assay shifts?
5. Is model confidence calibrated when the biological distribution shifts?
6. Does the random-versus-unseen-position gap replicate across independent proteins?
7. How do supervised baselines compare with audited zero-shot model scores?

## Main result

On 9,514 quality-filtered variants, the additive baseline loses **0.393 Spearman on Sal10**
and **0.373 on Sal25** when every test residue position is absent from training. These are
means across 10 complete benchmark repetitions (seeds 42–51), not a single favorable split.
Nominal 80% conformal coverage falls by **14.0** and **21.3 percentage points** respectively.

![VariantShift robustness and condition-transfer analysis](docs/shift-analysis.svg)

The result is the point of the project: preventing exact-variant overlap is not enough. A model
can interpolate mutations at residue positions it has already observed and still fail to
generalize to unmeasured regions of the protein.

| Target | Random Spearman | Unseen-position Spearman | Paired gap | 80% coverage shift |
| --- | ---: | ---: | ---: | ---: |
| Sal10 EC50 | 0.795 | 0.401 | **0.393** | 79.5% → 65.5% |
| Sal25 EC50 | 0.763 | 0.389 | **0.373** | 79.9% → 58.5% |

The paired gap remained positive in all 20 target/seed comparisons. Full seed-level and
aggregate results are in [`results/robustness/`](results/robustness/).

## Multi-protein validation

The result generalizes beyond the initial case study. VariantShift audited all 217 assays in the
ProteinGym v1.3 substitution benchmark using criteria fixed before evaluation. **195 assays across
169 proteins** passed sequence, identifier, measurement-count, and position-coverage checks,
yielding 689,994 single-substitution measurements.

Across ten paired split seeds, the additive baseline falls from **0.617 mean Spearman on random
variants to 0.351 at unseen positions** after first aggregating assays within UniProt ID. The mean
gap is **0.266** with a protein-bootstrap 95% interval of **0.246–0.287**. The protein-level gap is
positive for all 169 proteins.

![VariantShift multi-protein validation](docs/proteingym-analysis.svg)

| Supervised model | Random Spearman | Unseen-position Spearman | Paired gap |
| --- | ---: | ---: | ---: |
| Biophysical ridge | 0.371 | 0.324 | 0.048 |
| Additive ridge | **0.617** | **0.351** | **0.266** |

The complete eligibility ledger, seed-level evaluations, assay summaries, and UniProt-bootstrap
results are in [`results/proteingym/`](results/proteingym/). The protocol is specified in
[`docs/PROTEINGYM_METHODS.md`](docs/PROTEINGYM_METHODS.md).

### Audited ESM comparison

All 195 eligible assays also passed the official-score audit: complete one-to-one variant joins,
100% common score coverage, no duplicate identifiers, and experimental values agreeing to within
`8.9e-16`. The ESM-2 650M model has the strongest aggregate ranking performance; increasing model
size to 3B or 15B does not improve it.

| Fixed zero-shot scores | Random subset | Unseen-position subset | Subset difference |
| --- | ---: | ---: | ---: |
| ESM-1v ensemble | 0.404 | 0.395 | 0.008 |
| ESM-2 8M | 0.203 | 0.200 | 0.003 |
| ESM-2 35M | 0.314 | 0.305 | 0.009 |
| ESM-2 150M | 0.393 | 0.385 | 0.008 |
| **ESM-2 650M** | **0.427** | **0.420** | **0.007** |
| ESM-2 3B | 0.421 | 0.412 | 0.008 |
| ESM-2 15B | 0.411 | 0.402 | 0.009 |

The supervised additive model is much stronger on random variants (0.617 versus 0.427) but falls
below ESM-2 650M at unseen positions (0.351 versus 0.420). Because ESM scores are fixed and never
fit to assay labels, their random-to-position change describes subset composition rather than the
training-shift penalty measured for supervised models.

## Condition transfer

VariantShift also trains on each of the 20 measured assay conditions and evaluates the resulting
ranking against all 20 target conditions. This produces a complete 20×20 transfer matrix under
both random-variant and unseen-position splits: **800 source/target evaluations** with zero exact
variant overlap.

At unseen positions, mean in-condition Spearman is 0.345 and mean cross-condition Spearman is
0.340. Average transfer is stable, but the worst source/target pair falls to 0.132 and the largest
pair-specific transfer gap is 0.167. That distinction matters: condition shift is not uniformly
damaging, while residue-position novelty produces a large, persistent penalty.

The full matrix and summary are in [`results/transfer/`](results/transfer/).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

variantshift download data/raw --accept-data-use-agreement
variantshift inspect data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
variantshift benchmark data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
variantshift report artifacts/benchmark.csv --filtered-rows 9514
variantshift robustness data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
variantshift condition-transfer data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
```

Run the public multi-protein study separately:

```bash
make proteingym-download
make proteingym-audit
make proteingym-benchmark
make proteingym-zero-shot
make proteingym-figure
```

Every command is deterministic at the default seed. Raw data and per-variant predictions stay
outside version control.

## Evaluation regimes

- **Random variant:** identical mutation strings are grouped so no exact variant crosses the
  train/test boundary. Residue positions may still overlap.
- **Unseen position:** complete residue positions are held out. Variants spanning both train
  and test positions are excluded, producing zero residue overlap.
- **Higher mutation depth:** training is restricted to single substitutions and testing uses
  variants containing two to five substitutions.

Each training set is further divided into fit and calibration subsets. Reported uncertainty is
an 80% split-conformal interval; its coverage is measured independently on every shifted test
set. See the complete [`docs/METHODS.md`](docs/METHODS.md) for cohort construction, feature
definitions, split invariants, and uncertainty methodology.

The repeated benchmark uses consecutive seeds 42–51 and pairs random-versus-position results
within each seed. Seed ranges measure split sensitivity; they are not treated as independent
biological replicates or confidence intervals.

## Baselines

- `mean`: a no-information control.
- `biophysical_ridge`: mutation count, position, hydropathy, side-chain volume, charge,
  polarity, aromaticity, glycine, and proline deltas.
- `additive_ridge`: biochemical features plus sparse residue-position, substitution-class, and
  exact-mutation effects.

The baselines are intentionally interpretable. Protein language model comparisons are kept in a
separate audited experiment because their scores are fixed rather than trained on each assay and
their pretraining corpora may include related sequences.

The original optional ESM-2 8M scorer remains available as a local smoke test. The published
multi-protein comparison instead uses ProteinGym's official ESM-1v ensemble and complete ESM-2
scaling series, with one-to-one mutation joins, DMS-value agreement, score completeness, and
archive provenance audited before evaluation.

```bash
pip install -e '.[plm]'
variantshift esm-score data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
```

## Data

VariantShift never vendors the source measurements. Downloading the dataset requires
explicitly accepting the provider's data-use agreement.

Dataset: [TEV Protease — Pilot SSVL and epPCR Libraries](https://data.alignbio.org/groqseq/groqseq-014/)

The published release contains 18,486 rows and 151 columns. The default analysis removes indels,
nonsense mutations, and variants with fewer than 1,000 total barcode reads, leaving 9,514 rows.

## Repository structure

```text
src/variantshift/
  data.py          # download gate, schema validation, quality filtering
  mutations.py     # mutation parser and reference-sequence checks
  features.py      # biochemical and sparse additive encoders
  splits.py        # split construction and leakage audits
  models.py        # transparent supervised baselines
  metrics.py       # ranking, error, and conformal coverage
  evaluate.py      # benchmark orchestration
  robustness.py    # repeated splits and paired generalization gaps
  transfer.py      # source-to-target assay-condition transfer
  proteingym.py    # public assay ingestion and eligibility auditing
  multiprotein.py # repeated cross-protein supervised validation
  zero_shot.py     # official fixed-score join and evaluation audit
  provenance.py    # data/source/artifact integrity manifests
  visualize.py     # dependency-free shift-analysis SVG
  report.py        # standalone HTML result report
tests/             # unit and invariant tests
results/           # aggregate, reproducible benchmark outputs
```

## Result integrity

[`results/run-manifest.json`](results/run-manifest.json) binds the dataset SHA-256, source commit,
filter and evaluation configuration, dependency versions, and ten committed artifacts. CI verifies
every artifact byte-for-byte without requiring the licensed raw measurements:

```bash
variantshift verify-artifacts results/run-manifest.json
```

When the dataset is available locally, the same command verifies its hash as well.

## Limitations

- The primary robustness result covers two fitted EC50 endpoints; the transfer matrix covers 20
  complete `mean_y` condition readouts rather than every raw measurement column.
- Multi-protein validation covers 169 proteins but still evaluates models independently within
  each assay; it does not train on one protein and transfer labels to another family.
- Repeated seeds characterize split sensitivity on this cohort rather than uncertainty across
  proteins, assays, or biological replicates.
- Conformal coverage is guaranteed only under exchangeability; its breakdown under shift is a
  diagnostic, not a surprising violation of the method.
- Aggregate performance does not establish that a model is ready to prioritize wet-lab
  experiments.

## License

Code is released under the MIT License. The TEV dataset has separate terms from its provider.
