"""Command-line interface for reproducible VariantShift experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import condition_columns, download_dataset, quality_filter, read_tev_dataset, summarize
from .evaluate import DEFAULT_TARGETS, run_benchmark
from .multiprotein import (
    multiprotein_gaps,
    run_multiprotein_benchmark,
    summarize_assays,
    summarize_multiprotein_gaps,
)
from .proteingym import (
    PROTEINGYM_ARCHIVE_URL,
    PROTEINGYM_REFERENCE_URL,
    PROTEINGYM_SCORE_ARCHIVE_URL,
    PROTEINGYM_VERSION,
    EligibilityCriteria,
    audit_archive,
    download_proteingym,
)
from .provenance import (
    build_collection_manifest,
    build_run_manifest,
    git_revision,
    verify_manifest_artifacts,
    write_manifest,
)
from .report import render_report
from .robustness import (
    generalization_gaps,
    run_repeated_benchmark,
    summarize_generalization_gaps,
    summarize_robustness,
)
from .transfer import run_condition_transfer, summarize_condition_transfer
from .visualize import render_shift_figure
from .zero_shot import (
    DEFAULT_ESM_MODELS,
    run_zero_shot_benchmark,
    summarize_zero_shot,
    summarize_zero_shot_assays,
    zero_shot_subset_differences,
)


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

    provenance = subparsers.add_parser(
        "provenance", help="Create a hash-linked manifest for committed results"
    )
    _dataset_arguments(provenance)
    provenance.add_argument("--artifact", type=Path, action="append", required=True)
    provenance.add_argument("--output", type=Path, default=Path("results/run-manifest.json"))
    provenance.add_argument("--seed", type=int, default=42)
    provenance.add_argument("--robustness-start-seed", type=int, default=42)
    provenance.add_argument("--robustness-repeats", type=int, default=10)

    verify = subparsers.add_parser(
        "verify-artifacts", help="Verify committed artifacts against a run manifest"
    )
    verify.add_argument("manifest", type=Path)

    figure = subparsers.add_parser(
        "shift-figure", help="Render repeated-split and condition-transfer results as SVG"
    )
    figure.add_argument("gaps", type=Path)
    figure.add_argument("transfer", type=Path)
    figure.add_argument("--output", type=Path, default=Path("artifacts/shift-analysis.svg"))
    figure.add_argument("--model", default="additive_ridge")
    figure.add_argument("--transfer-split", default="position_holdout")

    report = subparsers.add_parser("report", help="Render a standalone HTML benchmark report")
    report.add_argument("benchmark", type=Path, help="Benchmark CSV produced by this CLI")
    report.add_argument("--output", type=Path, default=Path("artifacts/report.html"))
    report.add_argument("--filtered-rows", type=int, required=True)

    esm_score = subparsers.add_parser(
        "esm-score", help="Compute optional ESM-2 wild-type-marginal scores"
    )
    _dataset_arguments(esm_score)
    esm_score.add_argument("--output", type=Path, default=Path("artifacts/esm2-scores.csv"))

    pg_download = subparsers.add_parser(
        "proteingym-download",
        help="Download the public ProteinGym substitution benchmark",
    )
    pg_download.add_argument("destination", type=Path)

    pg_audit = subparsers.add_parser(
        "proteingym-audit",
        help="Audit every ProteinGym assay against fixed inclusion criteria",
    )
    pg_audit.add_argument("archive", type=Path)
    pg_audit.add_argument("reference", type=Path)
    pg_audit.add_argument("--output", type=Path, default=Path("artifacts/proteingym/eligibility.csv"))
    pg_audit.add_argument("--min-single-variants", type=int, default=500)
    pg_audit.add_argument("--min-positions", type=int, default=20)
    pg_audit.add_argument("--min-unique-scores", type=int, default=10)

    pg_benchmark = subparsers.add_parser(
        "proteingym-benchmark",
        help="Run repeated random and unseen-position evaluation across eligible assays",
    )
    pg_benchmark.add_argument("archive", type=Path)
    pg_benchmark.add_argument("reference", type=Path)
    pg_benchmark.add_argument("eligibility", type=Path)
    pg_benchmark.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym")
    )
    pg_benchmark.add_argument("--start-seed", type=int, default=42)
    pg_benchmark.add_argument("--repeats", type=int, default=10)
    pg_benchmark.add_argument("--bootstrap-repeats", type=int, default=10_000)
    pg_benchmark.add_argument("--workers", type=int, default=1)

    pg_zero_shot = subparsers.add_parser(
        "proteingym-zero-shot",
        help="Audit and evaluate official ProteinGym zero-shot ESM scores",
    )
    pg_zero_shot.add_argument("source_archive", type=Path)
    pg_zero_shot.add_argument("score_archive", type=Path)
    pg_zero_shot.add_argument("reference", type=Path)
    pg_zero_shot.add_argument("eligibility", type=Path)
    pg_zero_shot.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym")
    )
    pg_zero_shot.add_argument(
        "--model",
        action="append",
        help="Official score column; defaults to the complete ESM-1v/ESM-2 series",
    )
    pg_zero_shot.add_argument("--start-seed", type=int, default=42)
    pg_zero_shot.add_argument("--repeats", type=int, default=10)
    pg_zero_shot.add_argument("--min-common-coverage", type=float, default=0.95)
    pg_zero_shot.add_argument("--bootstrap-repeats", type=int, default=10_000)

    pg_provenance = subparsers.add_parser(
        "proteingym-provenance",
        help="Create a multi-input integrity manifest for ProteinGym results",
    )
    pg_provenance.add_argument("source_archive", type=Path)
    pg_provenance.add_argument("score_archive", type=Path)
    pg_provenance.add_argument("reference", type=Path)
    pg_provenance.add_argument("eligibility", type=Path)
    pg_provenance.add_argument("--artifact", type=Path, action="append", required=True)
    pg_provenance.add_argument(
        "--output", type=Path, default=Path("results/proteingym/run-manifest.json")
    )
    pg_provenance.add_argument("--start-seed", type=int, default=42)
    pg_provenance.add_argument("--repeats", type=int, default=10)
    pg_provenance.add_argument("--bootstrap-repeats", type=int, default=10_000)
    pg_provenance.add_argument("--source-revision")
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
            workers=arguments.workers,
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

    if arguments.command == "provenance":
        frame = _load_filtered(arguments)
        root = Path.cwd()
        manifest = build_run_manifest(
            repository_root=root,
            dataset_path=arguments.dataset,
            dataset_name=arguments.dataset.name,
            dataset_source="https://data.alignbio.org/groqseq/groqseq-014/",
            dataset_version="1.1.0",
            rows_after_filtering=len(frame),
            filters={
                "exclude_indels": not arguments.include_indels,
                "exclude_stops": not arguments.include_stops,
                "min_total_counts": arguments.min_total_counts,
            },
            run={
                "seed": arguments.seed,
                "robustness_start_seed": arguments.robustness_start_seed,
                "robustness_repeats": arguments.robustness_repeats,
                "calibration_fraction": 0.2,
                "nominal_coverage": 0.8,
                "targets": list(DEFAULT_TARGETS),
                "condition_transfer": "all mean_y conditions",
            },
            artifact_paths=arguments.artifact,
            source_revision=git_revision(root),
        )
        print(write_manifest(manifest, arguments.output))
        return 0

    if arguments.command == "verify-artifacts":
        print(
            json.dumps(
                verify_manifest_artifacts(arguments.manifest),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "shift-figure":
        import pandas as pd

        output = render_shift_figure(
            pd.read_csv(arguments.gaps),
            pd.read_csv(arguments.transfer),
            arguments.output,
            model=arguments.model,
            transfer_split=arguments.transfer_split,
        )
        print(output)
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

    if arguments.command == "proteingym-download":
        outputs = download_proteingym(arguments.destination)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-audit":
        criteria = EligibilityCriteria(
            min_single_variants=arguments.min_single_variants,
            min_positions=arguments.min_positions,
            min_unique_scores=arguments.min_unique_scores,
        )
        audit = audit_archive(arguments.archive, arguments.reference, criteria=criteria)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(arguments.output, index=False)
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "assays": len(audit),
                    "eligible": int(audit["eligible"].sum()),
                    "criteria": criteria.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "proteingym-benchmark":
        import pandas as pd

        eligibility = pd.read_csv(arguments.eligibility)
        runs = run_multiprotein_benchmark(
            arguments.archive,
            arguments.reference,
            eligibility,
            start_seed=arguments.start_seed,
            repeats=arguments.repeats,
        )
        gaps = multiprotein_gaps(runs)
        assays = summarize_assays(gaps)
        aggregate = summarize_multiprotein_gaps(
            gaps,
            bootstrap_repeats=arguments.bootstrap_repeats,
        )
        if runs["exact_variant_overlap"].ne(0).any():
            raise RuntimeError("Exact-variant leakage detected in ProteinGym benchmark")
        position_runs = runs.loc[runs["split"].eq("position_holdout")]
        if position_runs["shared_position_count"].ne(0).any():
            raise RuntimeError("Residue-position leakage detected in position holdout")

        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "runs": output_dir / "benchmark-runs.csv",
            "gaps": output_dir / "generalization-gaps.csv",
            "assays": output_dir / "assay-summary.csv",
            "aggregate": output_dir / "aggregate-summary.csv",
        }
        runs.to_csv(outputs["runs"], index=False)
        gaps.to_csv(outputs["gaps"], index=False)
        assays.to_csv(outputs["assays"], index=False)
        aggregate.to_csv(outputs["aggregate"], index=False)
        print(
            json.dumps(
                {
                    "outputs": {key: str(path) for key, path in outputs.items()},
                    "assays": int(runs["assay_id"].nunique()),
                    "proteins": int(runs["uniprot_id"].nunique()),
                    "seeds": int(runs["seed"].nunique()),
                    "leakage_checks": "passed",
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "proteingym-zero-shot":
        import pandas as pd

        models = tuple(arguments.model or DEFAULT_ESM_MODELS)
        eligibility = pd.read_csv(arguments.eligibility)
        runs, score_audit = run_zero_shot_benchmark(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            eligibility,
            model_columns=models,
            start_seed=arguments.start_seed,
            repeats=arguments.repeats,
            min_common_coverage=arguments.min_common_coverage,
        )
        differences = zero_shot_subset_differences(runs)
        assays = summarize_zero_shot_assays(differences)
        aggregate = summarize_zero_shot(
            differences,
            bootstrap_repeats=arguments.bootstrap_repeats,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "score_audit": output_dir / "esm-score-audit.csv",
            "runs": output_dir / "esm-subset-runs.csv",
            "differences": output_dir / "esm-subset-differences.csv",
            "assays": output_dir / "esm-assay-summary.csv",
            "aggregate": output_dir / "esm-aggregate-summary.csv",
        }
        score_audit.to_csv(outputs["score_audit"], index=False)
        runs.to_csv(outputs["runs"], index=False)
        differences.to_csv(outputs["differences"], index=False)
        assays.to_csv(outputs["assays"], index=False)
        aggregate.to_csv(outputs["aggregate"], index=False)
        print(
            json.dumps(
                {
                    "outputs": {key: str(path) for key, path in outputs.items()},
                    "models": list(models),
                    "audited_assays": len(score_audit),
                    "eligible_assays": int(score_audit["eligible_for_zero_shot"].sum()),
                    "score_semantics": (
                        "fixed zero-shot scores; split differences are subset sensitivity, "
                        "not supervised generalization gaps"
                    ),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "proteingym-provenance":
        import pandas as pd

        eligibility = pd.read_csv(arguments.eligibility)
        artifact_paths = list(dict.fromkeys([arguments.eligibility, *arguments.artifact]))
        manifest = build_collection_manifest(
            repository_root=Path.cwd(),
            inputs={
                "substitution_assays": {
                    "path": arguments.source_archive,
                    "source": PROTEINGYM_ARCHIVE_URL,
                    "version": PROTEINGYM_VERSION,
                },
                "reference_index": {
                    "path": arguments.reference,
                    "source": PROTEINGYM_REFERENCE_URL,
                    "version": PROTEINGYM_VERSION,
                },
                "official_zero_shot_scores": {
                    "path": arguments.score_archive,
                    "source": PROTEINGYM_SCORE_ARCHIVE_URL,
                    "version": PROTEINGYM_VERSION,
                },
            },
            protocol={
                "cohort": "finite single substitutions",
                "eligibility": EligibilityCriteria().to_dict(),
                "audited_assays": len(eligibility),
                "eligible_assays": int(eligibility["eligible"].sum()),
                "splits": ["random_variant", "position_holdout"],
                "start_seed": arguments.start_seed,
                "repeats": arguments.repeats,
                "calibration_fraction": 0.2,
                "nominal_coverage": 0.8,
                "bootstrap_unit": "UniProt_ID",
                "bootstrap_repeats": arguments.bootstrap_repeats,
                "supervised_models": [
                    "mean",
                    "biophysical_ridge",
                    "additive_ridge",
                ],
                "zero_shot_models": list(DEFAULT_ESM_MODELS),
                "zero_shot_note": (
                    "Fixed-score subset differences are not supervised generalization gaps"
                ),
            },
            artifact_paths=artifact_paths,
            source_revision=arguments.source_revision,
        )
        print(write_manifest(manifest, arguments.output))
        return 0

    raise AssertionError(f"Unhandled command: {arguments.command}")
