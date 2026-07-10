.PHONY: install test lint benchmark report clean

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

clean:
	rm -rf artifacts .coverage htmlcov .pytest_cache

