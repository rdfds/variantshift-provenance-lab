.PHONY: install test lint benchmark report robustness transfer figure verify clean

DATASET := data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv

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

clean:
	rm -rf artifacts .coverage htmlcov .pytest_cache
