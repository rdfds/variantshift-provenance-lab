.PHONY: install test lint benchmark report robustness transfer figure verify proteingym-download proteingym-audit proteingym-benchmark proteingym-zero-shot proteingym-figure clean

DATASET := data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv
PROTEINGYM_DIR := data/raw/proteingym
PROTEINGYM_ASSAYS := $(PROTEINGYM_DIR)/DMS_ProteinGym_substitutions.zip
PROTEINGYM_SCORES := $(PROTEINGYM_DIR)/zero_shot_substitutions_scores.zip
PROTEINGYM_REFERENCE := $(PROTEINGYM_DIR)/DMS_substitutions.csv

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
	variantshift proteingym-download $(PROTEINGYM_DIR) --include-zero-shot-scores

proteingym-audit:
	variantshift proteingym-audit $(PROTEINGYM_ASSAYS) $(PROTEINGYM_REFERENCE) --output artifacts/proteingym/eligibility.csv

proteingym-benchmark:
	variantshift proteingym-benchmark $(PROTEINGYM_ASSAYS) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym --workers 4

proteingym-zero-shot:
	variantshift proteingym-zero-shot $(PROTEINGYM_ASSAYS) $(PROTEINGYM_SCORES) $(PROTEINGYM_REFERENCE) artifacts/proteingym/eligibility.csv --output-dir artifacts/proteingym

proteingym-figure:
	variantshift proteingym-figure artifacts/proteingym/assay-summary.csv artifacts/proteingym/aggregate-summary.csv artifacts/proteingym/esm-aggregate-summary.csv --output artifacts/proteingym-analysis.svg

clean:
	rm -rf artifacts .coverage htmlcov .pytest_cache
