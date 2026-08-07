"""Command-line interface for reproducible VariantShift experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import download_dataset, quality_filter, read_tev_dataset, summarize
from .evaluate import DEFAULT_TARGETS, run_benchmark
from .report import render_report


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
