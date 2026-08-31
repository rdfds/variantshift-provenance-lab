.PHONY: install test lint benchmark report robustness transfer figure verify proteingym-download proteingym-audit proteingym-benchmark proteingym-zero-shot proteingym-official-supervised proteingym-esm2-embeddings proteingym-embedding-probe proteingym-heldout-protein proteingym-family-clusters proteingym-heldout-family proteingym-structure-clusters proteingym-heldout-structure-family proteingym-curated-families proteingym-heldout-curated-family proteingym-heldout-curated-family-ablation proteingym-modern-zero-shot proteingym-crossover proteingym-figure proteingym-extended-figure proteingym-research-figure mavedb-freeze-external mavedb-download-external mavedb-evaluate-external mavedb-external-figure model-preflight transport-features transport-fit confirmation-overlap budget-check site workflow-local workflow-slurm clean

DATASET := data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
PROTEINGYM_DIR := data/raw/proteingym
PROTEINGYM_ASSAYS := $(PROTEINGYM_DIR)/DMS_ProteinGym_substitutions.zip
PROTEINGYM_SCORES := $(PROTEINGYM_DIR)/zero_shot_substitutions_scores.zip
PROTEINGYM_SUPERVISED := $(PROTEINGYM_DIR)/DMS_supervised_substitutions_scores.zip
PROTEINGYM_STRUCTURES := $(PROTEINGYM_DIR)/ProteinGym_AF2_structures.zip
PROTEINGYM_REFERENCE := $(PROTEINGYM_DIR)/DMS_substitutions.csv
PROTEINGYM_EMBEDDINGS := $(PROTEINGYM_DIR)/esm2_t6_8M_embeddings
MAVEDB_PROTOCOL := protocols/mavedb-external-v1
MAVEDB_RAW := data/raw/mavedb-external-v1
MAVEDB_WORK := data/processed/mavedb-external-v1
MAVEDB_RESULTS := results/mavedb-external-v1

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=variantshift --cov-report=term-missing

lint:
	ruff check src tests

benchmark:
	variantshift benchmark $(DATASET) --output artifacts/benchmark.csv

report: benchmark
	variantshift report artifacts/benchmark.csv --filtered-rows 9514 --output artifacts/report.html

robustness:
	variantshift robustness $(DATASET) --output-dir artifacts/robustness

transfer:
	variantshift condition-transfer $(DATASET) --output-dir artifacts/transfer

figure: robustness transfer
	variantshift shift-figure artifacts/robustness/generalization-gaps.csv artifacts/transfer/condition-transfer.csv --output artifacts/shift-analysis.svg

verify:
	variantshift verify-artifacts results/run-manifest.json

proteingym-download:
	variantshift proteingym-download $(PROTEINGYM_DIR) --include-zero-shot-scores --include-supervised-scores --include-structures

proteingym-audit:
	variantshift proteingym-audit $(PROTEINGYM_ASSAYS) $(PROTEINGYM_REFERENCE) --output artifacts/proteingym/eligibility.csv

proteingym-benchmark:
	variantshift proteingym-benchmark $(PROTEINGYM_ASSAYS) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym --workers 4

proteingym-zero-shot:
	variantshift proteingym-zero-shot $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym

proteingym-official-supervised:
	variantshift proteingym-official-supervised $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SUPERVISED) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym

proteingym-esm2-embeddings:
	variantshift proteingym-esm2-embeddings $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv $(PROTEINGYM_EMBEDDINGS)

proteingym-embedding-probe:
	variantshift proteingym-embedding-probe $(PROTEINGYM_ASSAYS) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv $(PROTEINGYM_EMBEDDINGS)/index.csv --output-dir artifacts/proteingym/extended --workers 4

proteingym-heldout-protein:
	variantshift proteingym-heldout-protein $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym/extended --repeats 5

proteingym-family-clusters:
	variantshift proteingym-family-clusters $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym/extended

proteingym-heldout-family:
	variantshift proteingym-heldout-family $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/sequence-family-assignments.csv --protein-assays artifacts/proteingym/extended/heldout-protein-assays.csv --output-dir artifacts/proteingym/extended --repeats 5

proteingym-structure-clusters:
	variantshift proteingym-structure-clusters $(PROTEINGYM_STRUCTURES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/sequence-family-assignments.csv --output-dir artifacts/proteingym/extended

proteingym-heldout-structure-family:
	variantshift proteingym-heldout-structure-family $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/sequence-structure-family-assignments.csv --protein-assays artifacts/proteingym/extended/heldout-protein-assays.csv --sequence-family-assays artifacts/proteingym/extended/heldout-family-assays.csv --output-dir artifacts/proteingym/extended --repeats 5

proteingym-curated-families:
	variantshift proteingym-curated-families $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/sequence-structure-family-assignments.csv $(PROTEINGYM_DIR)/interpro-cache --output-dir artifacts/proteingym/extended --workers 4

proteingym-heldout-curated-family:
	variantshift proteingym-heldout-curated-family $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/curated-family-assignments.csv --protein-assays artifacts/proteingym/extended/heldout-protein-assays.csv --structure-family-assays artifacts/proteingym/extended/heldout-structure-family-assays.csv --output-dir artifacts/proteingym/extended --repeats 5

proteingym-heldout-curated-family-ablation:
	variantshift proteingym-heldout-curated-family $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/curated-family-assignments.csv --output-dir artifacts/proteingym/extended/feature-ablation --repeats 5 --feature-ablation

proteingym-modern-zero-shot:
	variantshift proteingym-modern-zero-shot $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym/extended

proteingym-crossover:
	variantshift proteingym-crossover $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv artifacts/proteingym/extended/embedding-probe-runs.csv artifacts/proteingym/esm-subset-runs.csv --supervised-model esm2_residue_ridge_probe --output-dir artifacts/proteingym/extended

proteingym-extended-figure:
	variantshift proteingym-extended-figure artifacts/proteingym/extended/official-supervised-summary.csv artifacts/proteingym/extended/embedding-probe-summary.csv artifacts/proteingym/extended/heldout-protein-summary.csv artifacts/proteingym/extended/crossover-summary.csv --heldout-family-summary artifacts/proteingym/extended/heldout-family-summary.csv --heldout-structure-family-summary artifacts/proteingym/extended/heldout-structure-family-summary.csv --output artifacts/proteingym-extended.svg

proteingym-research-figure:
	variantshift proteingym-research-figure artifacts/proteingym/extended/modern-zero-shot-summary.csv artifacts/proteingym/extended/sequence-family-audit.csv artifacts/proteingym/extended/structure-family-audit.csv artifacts/proteingym/extended/curated-family-audit.csv artifacts/proteingym/extended/heldout-protein-summary.csv artifacts/proteingym/extended/heldout-family-summary.csv artifacts/proteingym/extended/heldout-structure-family-summary.csv artifacts/proteingym/extended/heldout-curated-family-summary.csv --output artifacts/proteingym-research.svg

proteingym-figure:
	variantshift proteingym-figure artifacts/proteingym/assay-summary.csv artifacts/proteingym/aggregate-summary.csv artifacts/proteingym/esm-aggregate-summary.csv --output artifacts/proteingym-analysis.svg

mavedb-freeze-external:
	variantshift mavedb-freeze-external $(PROTEINGYM_REFERENCE) --output-dir $(MAVEDB_PROTOCOL)

mavedb-download-external:
	variantshift mavedb-download-external $(MAVEDB_PROTOCOL)/protocol.json $(MAVEDB_RAW)

mavedb-evaluate-external:
	variantshift mavedb-evaluate-external $(MAVEDB_PROTOCOL) $(MAVEDB_RAW) --work-dir $(MAVEDB_WORK) --output-dir $(MAVEDB_RESULTS)

mavedb-external-figure:
	variantshift mavedb-external-figure $(MAVEDB_PROTOCOL)/protocol.json $(MAVEDB_RESULTS)/assay-audit.csv $(MAVEDB_RESULTS)/bootstrap-summary.csv $(MAVEDB_RESULTS)/protein-metrics.csv --output docs/mavedb-external-validation.svg

model-preflight:
	variantshift models-preflight configs/model-panel-v1.json --output results/model-panel-v1/metadata-audit.csv

transport-features:
	variantshift transport-features-proteingym results/proteingym/extended/modern-zero-shot-runs.csv results/proteingym/eligibility.csv $(PROTEINGYM_REFERENCE) results/proteingym/extended/curated-family-assignments.csv results/proteingym/extended/sequence-family-alignments.csv results/proteingym/extended/structure-family-alignments.csv results/proteingym/extended/curated-pfam-domain-overlaps.csv $(PROTEINGYM_SCORES) --crossover-predictions results/proteingym/extended/crossover-heldout-predictions.csv --output results/transport-v1/development-task-features.csv

transport-fit: transport-features
	variantshift transport-fit results/transport-v1/development-task-features.csv configs/transport-v1.json --output-dir results/transport-v1

confirmation-overlap:
	variantshift confirmation-overlap-audit $(PROTEINGYM_REFERENCE) results/proteingym/eligibility.csv --confirmation-target protocols/mavedb-complement-v1/frozen/targets.csv --confirmation-target protocols/venusmuthub-v1/frozen/targets.csv --model-config configs/model-panel-v1.json --output-dir results/confirmation-overlap-v1

budget-check:
	variantshift compute-budget-check configs/compute-ledger.csv --planned-cost-usd 0 --output results/compute-budget-status.json

site: model-preflight transport-fit
	variantshift site-build configs/site-v1.json site

workflow-local:
	snakemake --snakefile workflow/Snakefile --profile workflow/profiles/local

workflow-slurm:
	snakemake --snakefile workflow/Snakefile --profile workflow/profiles/slurm

clean:
	rm -rf artifacts .coverage htmlcov .pytest_cache
