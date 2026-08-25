"""Command-line interface for reproducible VariantShift experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import condition_columns, download_dataset, quality_filter, read_tev_dataset, summarize
from .evaluate import DEFAULT_TARGETS, run_benchmark
from .report import render_report
from .robustness import (
    generalization_gaps,
    run_repeated_benchmark,
    summarize_generalization_gaps,
    summarize_robustness,
)
from .transfer import run_condition_transfer, summarize_condition_transfer


def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path, help="Path to the released TEV CSV")
    parser.add_argument("--min-total-counts", type=int, default=1_000)
    parser.add_argument(
        "--include-stops", action="store_true", help="Retain nonsense mutations"
    )
    parser.add_argument("--include-indels", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="variantshift",
        description="Leakage-aware evaluation for protein variant-effect models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download the Align TEV release")
    download.add_argument("destination", type=Path)
    download.add_argument("--accept-data-use-agreement", action="store_true")

    inspect = subparsers.add_parser("inspect", help="Validate and summarize a dataset")
    _dataset_arguments(inspect)

    benchmark = subparsers.add_parser("benchmark", help="Run all baseline evaluations")
    _dataset_arguments(benchmark)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.csv"))
    benchmark.add_argument("--target", action="append", choices=DEFAULT_TARGETS)
    benchmark.add_argument("--seed", type=int, default=42)

    robustness = subparsers.add_parser(
        "robustness", help="Repeat all split regimes and summarize seed sensitivity"
    )
    _dataset_arguments(robustness)
    robustness.add_argument("--output-dir", type=Path, default=Path("artifacts/robustness"))
    robustness.add_argument("--target", action="append", choices=DEFAULT_TARGETS)
    robustness.add_argument("--start-seed", type=int, default=42)
    robustness.add_argument("--repeats", type=int, default=10)

    transfer = subparsers.add_parser(
        "condition-transfer",
        help="Measure source-to-target assay transfer under biological shifts",
    )
    _dataset_arguments(transfer)
    transfer.add_argument("--output-dir", type=Path, default=Path("artifacts/transfer"))
    transfer.add_argument(
        "--condition",
        action="append",
        help="Condition column to include; defaults to every measured mean_y condition",
    )
    transfer.add_argument(
        "--model",
        choices=("biophysical_ridge", "additive_ridge"),
        default="additive_ridge",
    )
    transfer.add_argument("--seed", type=int, default=42)

    report = subparsers.add_parser("report", help="Render a standalone HTML benchmark report")
    report.add_argument("benchmark", type=Path, help="Benchmark CSV produced by this CLI")
    report.add_argument("--output", type=Path, default=Path("artifacts/report.html"))
    report.add_argument("--filtered-rows", type=int, required=True)

    esm_score = subparsers.add_parser(
        "esm-score", help="Compute optional ESM-2 wild-type-marginal scores"
    )
    _dataset_arguments(esm_score)
    esm_score.add_argument("--output", type=Path, default=Path("artifacts/esm2-scores.csv"))
    return parser


def _load_filtered(arguments: argparse.Namespace):
    frame = read_tev_dataset(arguments.dataset)
    return quality_filter(
        frame,
        min_total_counts=arguments.min_total_counts,
        substitutions_only=not arguments.include_indels,
        exclude_stop=not arguments.include_stops,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if arguments.command == "download":
        path = download_dataset(
            arguments.destination,
            accept_data_use_agreement=arguments.accept_data_use_agreement,
        )
        print(path)
        return 0

    if arguments.command == "inspect":
        frame = _load_filtered(arguments)
        print(json.dumps(summarize(frame).to_dict(), indent=2, sort_keys=True))
        return 0

    if arguments.command == "benchmark":
        frame = _load_filtered(arguments)
        targets = tuple(arguments.target) if arguments.target else DEFAULT_TARGETS
        results = run_benchmark(frame, targets=targets, seed=arguments.seed)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(arguments.output, index=False)
        print(arguments.output)
        return 0

    if arguments.command == "robustness":
        frame = _load_filtered(arguments)
        targets = tuple(arguments.target) if arguments.target else DEFAULT_TARGETS
        runs = run_repeated_benchmark(
            frame,
            targets=targets,
            start_seed=arguments.start_seed,
            repeats=arguments.repeats,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "runs": output_dir / "benchmark-runs.csv",
            "summary": output_dir / "summary.csv",
            "gaps": output_dir / "generalization-gaps.csv",
            "gap_summary": output_dir / "generalization-gap-summary.csv",
        }
        gaps = generalization_gaps(runs)
        runs.to_csv(outputs["runs"], index=False)
        summarize_robustness(runs).to_csv(outputs["summary"], index=False)
        gaps.to_csv(outputs["gaps"], index=False)
        summarize_generalization_gaps(gaps).to_csv(
            outputs["gap_summary"], index=False
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "condition-transfer":
        from .models import baseline_factories

        frame = _load_filtered(arguments)
        conditions = tuple(arguments.condition or condition_columns(frame))
        results = run_condition_transfer(
            frame,
            conditions=conditions,
            seed=arguments.seed,
            model_factory=baseline_factories()[arguments.model],
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "matrix": output_dir / "condition-transfer.csv",
            "summary": output_dir / "condition-transfer-summary.csv",
        }
        results.to_csv(outputs["matrix"], index=False)
        summarize_condition_transfer(results).to_csv(outputs["summary"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "report":
        import pandas as pd

        output = render_report(
            pd.read_csv(arguments.benchmark),
            arguments.output,
            filtered_rows=arguments.filtered_rows,
        )
        print(output)
        return 0

    if arguments.command == "esm-score":
        from .plm import esm2_wild_type_marginals

        frame = _load_filtered(arguments)
        wild_types = frame.loc[
            frame["goi_amino_mutations"].eq(0), "goi_amino_seq"
        ].drop_duplicates()
        if len(wild_types) != 1:
            raise ValueError(f"Expected one wild-type sequence, found {len(wild_types)}")
        scores = esm2_wild_type_marginals(
            wild_types.item(), frame["mutation_codes"].astype(str)
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(arguments.output, index=False)
        print(arguments.output)
        return 0

    raise AssertionError(f"Unhandled command: {arguments.command}")
