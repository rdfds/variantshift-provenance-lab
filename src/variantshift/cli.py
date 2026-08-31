"""Command-line interface for reproducible VariantShift experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import condition_columns, download_dataset, quality_filter, read_tev_dataset, summarize
from .evaluate import DEFAULT_TARGETS, run_benchmark
from .extended_visualize import render_extended_figure
from .multiprotein import (
    multiprotein_gaps,
    run_multiprotein_benchmark,
    summarize_assays,
    summarize_multiprotein_gaps,
)
from .multiprotein_visualize import render_multiprotein_figure
from .official_supervised import (
    run_official_supervised_benchmark,
    summarize_official_supervised,
)
from .proteingym import (
    PROTEINGYM_ARCHIVE_URL,
    PROTEINGYM_REFERENCE_URL,
    PROTEINGYM_SCORE_ARCHIVE_URL,
    PROTEINGYM_STRUCTURE_ARCHIVE_URL,
    PROTEINGYM_SUPERVISED_ARCHIVE_URL,
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
    parser.add_argument("--include-stops", action="store_true", help="Retain nonsense mutations")
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
    pg_download.add_argument("--include-zero-shot-scores", action="store_true")
    pg_download.add_argument("--include-supervised-scores", action="store_true")
    pg_download.add_argument("--include-structures", action="store_true")

    pg_audit = subparsers.add_parser(
        "proteingym-audit",
        help="Audit every ProteinGym assay against fixed inclusion criteria",
    )
    pg_audit.add_argument("archive", type=Path)
    pg_audit.add_argument("reference", type=Path)
    pg_audit.add_argument(
        "--output", type=Path, default=Path("artifacts/proteingym/eligibility.csv")
    )
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
    pg_benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts/proteingym"))
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
    pg_zero_shot.add_argument("--output-dir", type=Path, default=Path("artifacts/proteingym"))
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
    pg_provenance.add_argument("--probe-repeats", type=int, default=5)
    pg_provenance.add_argument("--heldout-folds", type=int, default=5)
    pg_provenance.add_argument("--heldout-repeats", type=int, default=5)
    pg_provenance.add_argument("--family-folds", type=int, default=5)
    pg_provenance.add_argument("--family-identity-threshold", type=float, default=0.30)
    pg_provenance.add_argument("--family-coverage-threshold", type=float, default=0.80)
    pg_provenance.add_argument("--structure-family-folds", type=int, default=5)
    pg_provenance.add_argument("--structure-tm-threshold", type=float, default=0.50)
    pg_provenance.add_argument("--structure-coverage-threshold", type=float, default=0.80)
    pg_provenance.add_argument("--structure-probability-threshold", type=float, default=0.95)
    pg_provenance.add_argument("--curated-family-folds", type=int, default=5)
    pg_provenance.add_argument("--curated-mapping-threshold", type=float, default=0.80)
    pg_provenance.add_argument("--curated-overlap-threshold", type=float, default=0.50)
    pg_provenance.add_argument("--crossover-folds", type=int, default=5)
    pg_provenance.add_argument("--bootstrap-repeats", type=int, default=10_000)
    pg_provenance.add_argument("--source-revision")
    pg_provenance.add_argument("--supervised-archive", type=Path)
    pg_provenance.add_argument("--embedding-index", type=Path)
    pg_provenance.add_argument("--structure-archive", type=Path)

    pg_figure = subparsers.add_parser(
        "proteingym-figure",
        help="Render the multi-protein supervised and ESM result summary",
    )
    pg_figure.add_argument("supervised_assays", type=Path)
    pg_figure.add_argument("supervised_aggregate", type=Path)
    pg_figure.add_argument("esm_aggregate", type=Path)
    pg_figure.add_argument("--output", type=Path, default=Path("artifacts/proteingym-analysis.svg"))

    pg_official = subparsers.add_parser(
        "proteingym-official-supervised",
        help="Audit ProteinNPT, Kermut, and embedding-probe out-of-fold predictions",
    )
    pg_official.add_argument("source_archive", type=Path)
    pg_official.add_argument("supervised_archive", type=Path)
    pg_official.add_argument("reference", type=Path)
    pg_official.add_argument("eligibility", type=Path)
    pg_official.add_argument("--output-dir", type=Path, default=Path("artifacts/proteingym"))
    pg_official.add_argument("--bootstrap-repeats", type=int, default=10_000)

    pg_embeddings = subparsers.add_parser(
        "proteingym-esm2-embeddings",
        help="Cache wild-type residue embeddings for a local ESM-2 probe",
    )
    pg_embeddings.add_argument("reference", type=Path)
    pg_embeddings.add_argument("eligibility", type=Path)
    pg_embeddings.add_argument("output_dir", type=Path)
    pg_embeddings.add_argument(
        "--model",
        choices=("esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D"),
        default="esm2_t6_8M_UR50D",
    )
    pg_embeddings.add_argument("--device", choices=("cpu", "mps", "cuda"))

    pg_probe = subparsers.add_parser(
        "proteingym-embedding-probe",
        help="Evaluate an ESM-2 residue embedding probe under four split regimes",
    )
    pg_probe.add_argument("archive", type=Path)
    pg_probe.add_argument("reference", type=Path)
    pg_probe.add_argument("eligibility", type=Path)
    pg_probe.add_argument("embedding_index", type=Path)
    pg_probe.add_argument("--output-dir", type=Path, default=Path("artifacts/proteingym/extended"))
    pg_probe.add_argument("--start-seed", type=int, default=42)
    pg_probe.add_argument("--repeats", type=int, default=5)
    pg_probe.add_argument("--workers", type=int, default=1)

    pg_heldout = subparsers.add_parser(
        "proteingym-heldout-protein",
        help="Train across assays and evaluate on entirely unseen proteins",
    )
    pg_heldout.add_argument("source_archive", type=Path)
    pg_heldout.add_argument("score_archive", type=Path)
    pg_heldout.add_argument("reference", type=Path)
    pg_heldout.add_argument("eligibility", type=Path)
    pg_heldout.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_heldout.add_argument("--max-variants-per-assay", type=int, default=1_000)
    pg_heldout.add_argument("--folds", type=int, default=5)
    pg_heldout.add_argument("--repeats", type=int, default=1)

    pg_family_clusters = subparsers.add_parser(
        "proteingym-family-clusters",
        help="Cluster eligible proteins by exhaustive assayed-sequence homology",
    )
    pg_family_clusters.add_argument("reference", type=Path)
    pg_family_clusters.add_argument("eligibility", type=Path)
    pg_family_clusters.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_family_clusters.add_argument("--identity-threshold", type=float, default=0.30)
    pg_family_clusters.add_argument("--coverage-threshold", type=float, default=0.80)
    pg_family_clusters.add_argument("--search-identity-floor", type=float, default=0.15)
    pg_family_clusters.add_argument("--search-coverage-floor", type=float, default=0.50)
    pg_family_clusters.add_argument("--mmseqs-binary", default="mmseqs")
    pg_family_clusters.add_argument("--threads", type=int, default=8)

    pg_heldout_family = subparsers.add_parser(
        "proteingym-heldout-family",
        help="Evaluate with complete sequence-family clusters absent from training",
    )
    pg_heldout_family.add_argument("source_archive", type=Path)
    pg_heldout_family.add_argument("score_archive", type=Path)
    pg_heldout_family.add_argument("reference", type=Path)
    pg_heldout_family.add_argument("eligibility", type=Path)
    pg_heldout_family.add_argument("family_assignments", type=Path)
    pg_heldout_family.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_heldout_family.add_argument("--max-variants-per-assay", type=int, default=1_000)
    pg_heldout_family.add_argument("--folds", type=int, default=5)
    pg_heldout_family.add_argument("--repeats", type=int, default=1)
    pg_heldout_family.add_argument("--protein-assays", type=Path)
    pg_heldout_family.add_argument("--bootstrap-repeats", type=int, default=10_000)

    pg_structure_clusters = subparsers.add_parser(
        "proteingym-structure-clusters",
        help="Augment sequence families with reciprocal Foldseek structure homology",
    )
    pg_structure_clusters.add_argument("structure_archive", type=Path)
    pg_structure_clusters.add_argument("reference", type=Path)
    pg_structure_clusters.add_argument("eligibility", type=Path)
    pg_structure_clusters.add_argument("sequence_assignments", type=Path)
    pg_structure_clusters.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_structure_clusters.add_argument("--minimum-tm-score", type=float, default=0.50)
    pg_structure_clusters.add_argument("--minimum-coverage", type=float, default=0.80)
    pg_structure_clusters.add_argument("--minimum-homology-probability", type=float, default=0.95)
    pg_structure_clusters.add_argument("--foldseek-binary", default="foldseek")
    pg_structure_clusters.add_argument("--threads", type=int, default=8)

    pg_heldout_structure = subparsers.add_parser(
        "proteingym-heldout-structure-family",
        help="Evaluate with complete sequence-and-structure families held out",
    )
    pg_heldout_structure.add_argument("source_archive", type=Path)
    pg_heldout_structure.add_argument("score_archive", type=Path)
    pg_heldout_structure.add_argument("reference", type=Path)
    pg_heldout_structure.add_argument("eligibility", type=Path)
    pg_heldout_structure.add_argument("family_assignments", type=Path)
    pg_heldout_structure.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_heldout_structure.add_argument("--max-variants-per-assay", type=int, default=1_000)
    pg_heldout_structure.add_argument("--folds", type=int, default=5)
    pg_heldout_structure.add_argument("--repeats", type=int, default=1)
    pg_heldout_structure.add_argument("--protein-assays", type=Path)
    pg_heldout_structure.add_argument("--sequence-family-assays", type=Path)
    pg_heldout_structure.add_argument("--bootstrap-repeats", type=int, default=10_000)

    pg_curated_clusters = subparsers.add_parser(
        "proteingym-curated-families",
        help="Augment sequence/structure families with curated Pfam family evidence",
    )
    pg_curated_clusters.add_argument("reference", type=Path)
    pg_curated_clusters.add_argument("eligibility", type=Path)
    pg_curated_clusters.add_argument("base_assignments", type=Path)
    pg_curated_clusters.add_argument("cache_dir", type=Path)
    pg_curated_clusters.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_curated_clusters.add_argument("--minimum-mapping-coverage", type=float, default=0.80)
    pg_curated_clusters.add_argument("--minimum-domain-overlap", type=float, default=0.50)
    pg_curated_clusters.add_argument("--workers", type=int, default=4)

    pg_heldout_curated = subparsers.add_parser(
        "proteingym-heldout-curated-family",
        help="Evaluate repeated folds with complete curated combined families held out",
    )
    pg_heldout_curated.add_argument("source_archive", type=Path)
    pg_heldout_curated.add_argument("score_archive", type=Path)
    pg_heldout_curated.add_argument("reference", type=Path)
    pg_heldout_curated.add_argument("eligibility", type=Path)
    pg_heldout_curated.add_argument("family_assignments", type=Path)
    pg_heldout_curated.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_heldout_curated.add_argument("--max-variants-per-assay", type=int, default=1_000)
    pg_heldout_curated.add_argument("--folds", type=int, default=5)
    pg_heldout_curated.add_argument("--repeats", type=int, default=5)
    pg_heldout_curated.add_argument("--protein-assays", type=Path)
    pg_heldout_curated.add_argument("--structure-family-assays", type=Path)
    pg_heldout_curated.add_argument("--bootstrap-repeats", type=int, default=10_000)
    pg_heldout_curated.add_argument("--feature-ablation", action="store_true")

    pg_modern_zero_shot = subparsers.add_parser(
        "proteingym-modern-zero-shot",
        help="Compare modern official zero-shot scores on an exactly paired assay cohort",
    )
    pg_modern_zero_shot.add_argument("source_archive", type=Path)
    pg_modern_zero_shot.add_argument("score_archive", type=Path)
    pg_modern_zero_shot.add_argument("reference", type=Path)
    pg_modern_zero_shot.add_argument("eligibility", type=Path)
    pg_modern_zero_shot.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_modern_zero_shot.add_argument("--minimum-common-coverage", type=float, default=0.95)
    pg_modern_zero_shot.add_argument("--bootstrap-repeats", type=int, default=10_000)

    pg_crossover = subparsers.add_parser(
        "proteingym-crossover",
        help="Predict supervised-versus-zero-shot wins on held-out proteins",
    )
    pg_crossover.add_argument("source_archive", type=Path)
    pg_crossover.add_argument("score_archive", type=Path)
    pg_crossover.add_argument("reference", type=Path)
    pg_crossover.add_argument("eligibility", type=Path)
    pg_crossover.add_argument("supervised_runs", type=Path)
    pg_crossover.add_argument("zero_shot_runs", type=Path)
    pg_crossover.add_argument("--supervised-model", required=True)
    pg_crossover.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/proteingym/extended")
    )
    pg_crossover.add_argument("--folds", type=int, default=5)

    pg_extended_figure = subparsers.add_parser(
        "proteingym-extended-figure",
        help="Render the structured-shift extension summary as SVG",
    )
    pg_extended_figure.add_argument("official_summary", type=Path)
    pg_extended_figure.add_argument("probe_summary", type=Path)
    pg_extended_figure.add_argument("heldout_summary", type=Path)
    pg_extended_figure.add_argument("crossover_summary", type=Path)
    pg_extended_figure.add_argument("--heldout-family-summary", type=Path)
    pg_extended_figure.add_argument("--heldout-structure-family-summary", type=Path)
    pg_extended_figure.add_argument(
        "--output", type=Path, default=Path("artifacts/proteingym-extended.svg")
    )

    pg_research_figure = subparsers.add_parser(
        "proteingym-research-figure",
        help="Render modern zero-shot, curated-family, and repeated-transfer results",
    )
    pg_research_figure.add_argument("modern_summary", type=Path)
    pg_research_figure.add_argument("sequence_audit", type=Path)
    pg_research_figure.add_argument("structure_audit", type=Path)
    pg_research_figure.add_argument("curated_audit", type=Path)
    pg_research_figure.add_argument("heldout_protein", type=Path)
    pg_research_figure.add_argument("heldout_family", type=Path)
    pg_research_figure.add_argument("heldout_structure", type=Path)
    pg_research_figure.add_argument("heldout_curated", type=Path)
    pg_research_figure.add_argument(
        "--output", type=Path, default=Path("artifacts/proteingym-research.svg")
    )

    external_freeze = subparsers.add_parser(
        "mavedb-freeze-external",
        help="Freeze a label-blind post-ProteinGym MaveDB validation panel",
    )
    external_freeze.add_argument("proteingym_reference", type=Path)
    external_freeze.add_argument(
        "--output-dir", type=Path, default=Path("protocols/mavedb-external-v1")
    )
    external_freeze.add_argument("--mmseqs-binary", default="mmseqs")
    external_freeze.add_argument("--threads", type=int, default=8)

    external_download = subparsers.add_parser(
        "mavedb-download-external",
        help="Access score tables for an already committed locked-box panel",
    )
    external_download.add_argument("protocol", type=Path)
    external_download.add_argument("output_dir", type=Path)

    external_evaluate = subparsers.add_parser(
        "mavedb-evaluate-external",
        help="Execute the frozen MaveDB cohort, scoring, and nested evaluation",
    )
    external_evaluate.add_argument("protocol_dir", type=Path)
    external_evaluate.add_argument("score_dir", type=Path)
    external_evaluate.add_argument(
        "--work-dir", type=Path, default=Path("data/processed/mavedb-external-v1")
    )
    external_evaluate.add_argument(
        "--output-dir", type=Path, default=Path("results/mavedb-external-v1")
    )
    external_evaluate.add_argument("--device")
    external_evaluate.add_argument("--batch-size", type=int, default=8)
    external_evaluate.add_argument("--bootstrap-repeats", type=int, default=10_000)
    external_evaluate.add_argument("--bootstrap-seed", type=int, default=2_026_0829)
    external_evaluate.add_argument("--reuse-predictions", action="store_true")

    external_figure = subparsers.add_parser(
        "mavedb-external-figure",
        help="Render the locked-box external-validation summary as SVG",
    )
    external_figure.add_argument("protocol", type=Path)
    external_figure.add_argument("assay_audit", type=Path)
    external_figure.add_argument("bootstrap_summary", type=Path)
    external_figure.add_argument("protein_metrics", type=Path)
    external_figure.add_argument(
        "--output", type=Path, default=Path("docs/mavedb-external-validation.svg")
    )

    panel_freeze = subparsers.add_parser(
        "panel-freeze",
        help="Freeze target-only confirmation inputs and enumerate the complete 19L landscape",
    )
    panel_freeze.add_argument("config", type=Path)
    panel_freeze.add_argument("output_dir", type=Path)

    mavedb_complement = subparsers.add_parser(
        "mavedb-freeze-complement-targets",
        help="Freeze the untouched MaveDB complement from metadata and target sequences only",
    )
    mavedb_complement.add_argument("reference", type=Path)
    mavedb_complement.add_argument("prior_protocol", type=Path)
    mavedb_complement.add_argument("--output-dir", type=Path, required=True)
    mavedb_complement.add_argument("--cutoff", default="2026-08-30")

    venus_targets = subparsers.add_parser(
        "venus-freeze-targets",
        help="Freeze a VenusMutHub target panel without reading mutation files",
    )
    venus_targets.add_argument("reference", type=Path)
    venus_targets.add_argument("mavedb_development_metadata", type=Path)
    venus_targets.add_argument("--output-dir", type=Path, required=True)
    venus_targets.add_argument("--workers", type=int, default=12)

    model_preflight = subparsers.add_parser(
        "models-preflight",
        help="Audit model licenses, execution, coverage, parity, and repeatability",
    )
    model_preflight.add_argument("model_config", type=Path)
    model_preflight.add_argument("--targets", type=Path)
    model_preflight.add_argument("--variants", type=Path)
    model_preflight.add_argument("--output", type=Path, default=Path("artifacts/model-audit.csv"))
    model_preflight.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/model-preflight-cache")
    )
    model_preflight.add_argument("--parity-dir", type=Path)
    model_preflight.add_argument("--execute", action="store_true")

    predict_panel = subparsers.add_parser(
        "predict-panel",
        help="Run outcome-blind model scoring with content-addressed target caches",
    )
    predict_panel.add_argument("model_config", type=Path)
    predict_panel.add_argument("targets", type=Path)
    predict_panel.add_argument("variants", type=Path)
    predict_panel.add_argument("--protocol-id", required=True)
    predict_panel.add_argument("--model", action="append")
    predict_panel.add_argument("--output-dir", type=Path, required=True)
    predict_panel.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/model-prediction-cache")
    )

    transport_fit = subparsers.add_parser(
        "transport-fit",
        help="Fit and freeze the group-calibrated VariantShift Transportability Score",
    )
    transport_fit.add_argument("features", type=Path)
    transport_fit.add_argument("config", type=Path)
    transport_fit.add_argument("--output-dir", type=Path, required=True)
    transport_fit.add_argument("--confirmation-features", type=Path)

    transport_features = subparsers.add_parser(
        "transport-features-proteingym",
        help="Build outcome-free ProteinGym task descriptors for transport fitting",
    )
    transport_features.add_argument("runs", type=Path)
    transport_features.add_argument("eligibility", type=Path)
    transport_features.add_argument("reference", type=Path)
    transport_features.add_argument("families", type=Path)
    transport_features.add_argument("sequence_alignments", type=Path)
    transport_features.add_argument("structure_alignments", type=Path)
    transport_features.add_argument("domain_overlaps", type=Path)
    transport_features.add_argument("score_archive", type=Path)
    transport_features.add_argument("--output", type=Path, required=True)
    transport_features.add_argument("--crossover-predictions", type=Path)

    transport_evaluate = subparsers.add_parser(
        "transport-evaluate",
        help="Evaluate already-frozen transport predictions after authorized outcome reveal",
    )
    transport_evaluate.add_argument("bundle", type=Path)
    transport_evaluate.add_argument("predictions", type=Path)
    transport_evaluate.add_argument("outcomes", type=Path)
    transport_evaluate.add_argument("--output-dir", type=Path, required=True)
    transport_evaluate.add_argument("--outcome-lock", type=Path)
    transport_evaluate.add_argument(
        "--negative-conclusion",
        type=Path,
        help=(
            "JSON record with statement, interpretation_change, and evidence_artifacts; "
            "required to pass the final publication gate"
        ),
    )

    overlap_audit = subparsers.add_parser(
        "confirmation-overlap-audit",
        help="Audit outcome-free confirmation novelty and model exposure strata",
    )
    overlap_audit.add_argument("proteingym_reference", type=Path)
    overlap_audit.add_argument("proteingym_eligibility", type=Path)
    overlap_audit.add_argument(
        "--confirmation-target", type=Path, action="append", required=True
    )
    overlap_audit.add_argument("--model-config", type=Path, required=True)
    overlap_audit.add_argument("--output-dir", type=Path, required=True)
    overlap_audit.add_argument("--confirmation-pfam", type=Path)
    overlap_audit.add_argument("--development-pfam", type=Path)
    overlap_audit.add_argument("--confirmation-structure-families", type=Path)
    overlap_audit.add_argument("--development-structure-families", type=Path)
    overlap_audit.add_argument("--mmseqs-binary", default="mmseqs")
    overlap_audit.add_argument("--threads", type=int, default=8)

    budget = subparsers.add_parser(
        "compute-budget-check",
        help="Block an external-compute job that would exceed the fixed budget",
    )
    budget.add_argument("ledger", type=Path)
    budget.add_argument("--planned-cost-usd", type=float, default=0.0)
    budget.add_argument("--hard-cap-usd", type=float, default=2_000.0)
    budget.add_argument("--output", type=Path, required=True)
    transport_evaluate.add_argument(
        "--development",
        action="store_true",
        help="Evaluate development data without a confirmation lock",
    )

    confirmation_freeze = subparsers.add_parser(
        "confirmation-freeze",
        help="Hash prediction and method artifacts before public preregistration",
    )
    confirmation_freeze.add_argument("outcome_lock", type=Path)
    confirmation_freeze.add_argument("--prediction", type=Path, action="append", required=True)
    confirmation_freeze.add_argument("--method", type=Path, action="append", required=True)

    confirmation_register = subparsers.add_parser(
        "confirmation-register",
        help="Record the public OSF or Zenodo registration URI",
    )
    confirmation_register.add_argument("outcome_lock", type=Path)
    confirmation_register.add_argument("registration_uri")

    confirmation_reveal = subparsers.add_parser(
        "confirmation-reveal",
        help="Record hashes for confirmation outcomes after registration",
    )
    confirmation_reveal.add_argument("outcome_lock", type=Path)
    confirmation_reveal.add_argument("--outcome", type=Path, action="append", required=True)

    preregistration = subparsers.add_parser(
        "preregistration-build",
        help="Build the public outcome-blind registration bundle",
    )
    preregistration.add_argument("protocol", type=Path)
    preregistration.add_argument("outcome_lock", type=Path)
    preregistration.add_argument("model_audit", type=Path)
    preregistration.add_argument("method", type=Path)
    preregistration.add_argument("--output-dir", type=Path, required=True)

    site_build = subparsers.add_parser(
        "site-build",
        help="Build the static benchmark and reliability-results explorer",
    )
    site_build.add_argument("config", type=Path)
    site_build.add_argument("output_dir", type=Path)
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

    if arguments.command == "panel-freeze":
        from .panels import freeze_panel

        outputs = freeze_panel(arguments.config, arguments.output_dir)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "mavedb-freeze-complement-targets":
        from .confirmation_panels import (
            MaveDBComplementCriteria,
            freeze_mavedb_complement_targets,
        )

        outputs = freeze_mavedb_complement_targets(
            arguments.reference,
            arguments.prior_protocol,
            arguments.output_dir,
            criteria=MaveDBComplementCriteria(frozen_on_or_before=arguments.cutoff),
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "venus-freeze-targets":
        from .venus_panel import freeze_venusmuthub_targets

        outputs = freeze_venusmuthub_targets(
            arguments.reference,
            arguments.mavedb_development_metadata,
            arguments.output_dir,
            workers=arguments.workers,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "models-preflight":
        import pandas as pd

        from .model_adapters import load_model_specifications, preflight_models

        if arguments.execute and (arguments.targets is None or arguments.variants is None):
            raise ValueError("--execute requires both --targets and --variants")
        audit = preflight_models(
            load_model_specifications(arguments.model_config),
            targets=pd.read_csv(arguments.targets) if arguments.targets else None,
            variants=pd.read_csv(arguments.variants) if arguments.variants else None,
            cache_dir=arguments.cache_dir,
            parity_dir=arguments.parity_dir,
            execute=arguments.execute,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(arguments.output, index=False)
        print(arguments.output)
        return 0

    if arguments.command == "predict-panel":
        from .model_adapters import write_panel_predictions

        outputs = write_panel_predictions(
            arguments.model_config,
            arguments.targets,
            arguments.variants,
            arguments.output_dir,
            protocol_id=arguments.protocol_id,
            cache_dir=arguments.cache_dir,
            model_ids=set(arguments.model) if arguments.model else None,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "transport-fit":
        from .transportability import fit_transportability

        outputs = fit_transportability(
            arguments.features,
            arguments.config,
            arguments.output_dir,
            confirmation_features_path=arguments.confirmation_features,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "transport-features-proteingym":
        from .transport_features import build_proteingym_transport_features

        summary = build_proteingym_transport_features(
            arguments.runs,
            arguments.eligibility,
            arguments.reference,
            arguments.families,
            arguments.sequence_alignments,
            arguments.structure_alignments,
            arguments.domain_overlaps,
            arguments.score_archive,
            arguments.output,
            crossover_predictions_path=arguments.crossover_predictions,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if arguments.command == "transport-evaluate":
        from .outcome_lock import assert_evaluation_artifacts_locked
        from .transportability import evaluate_frozen_transportability

        if arguments.development == bool(arguments.outcome_lock):
            raise ValueError("Choose exactly one of --development or --outcome-lock")
        if arguments.outcome_lock:
            assert_evaluation_artifacts_locked(
                arguments.outcome_lock,
                prediction_artifact=arguments.predictions,
                method_artifact=arguments.bundle,
                outcome_artifact=arguments.outcomes,
            )
        outputs = evaluate_frozen_transportability(
            arguments.bundle,
            arguments.predictions,
            arguments.outcomes,
            arguments.output_dir,
            negative_conclusion_path=arguments.negative_conclusion,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "confirmation-overlap-audit":
        from .overlap_audit import audit_confirmation_overlap

        outputs = audit_confirmation_overlap(
            arguments.proteingym_reference,
            arguments.proteingym_eligibility,
            arguments.confirmation_target,
            arguments.model_config,
            arguments.output_dir,
            confirmation_pfam_path=arguments.confirmation_pfam,
            development_pfam_path=arguments.development_pfam,
            confirmation_structure_path=arguments.confirmation_structure_families,
            development_structure_path=arguments.development_structure_families,
            mmseqs_binary=arguments.mmseqs_binary,
            threads=arguments.threads,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "compute-budget-check":
        from .compute_budget import check_compute_budget, write_budget_report

        report = check_compute_budget(
            arguments.ledger,
            planned_cost_usd=arguments.planned_cost_usd,
            hard_cap_usd=arguments.hard_cap_usd,
        )
        write_budget_report(report, arguments.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["permitted"] else 2

    if arguments.command == "confirmation-freeze":
        from .outcome_lock import freeze_predictions

        print(
            freeze_predictions(
                arguments.outcome_lock,
                prediction_artifacts=arguments.prediction,
                method_artifacts=arguments.method,
            )
        )
        return 0

    if arguments.command == "confirmation-register":
        from .outcome_lock import register_confirmation

        print(
            register_confirmation(
                arguments.outcome_lock, registration_uri=arguments.registration_uri
            )
        )
        return 0

    if arguments.command == "confirmation-reveal":
        from .outcome_lock import record_outcome_reveal

        print(record_outcome_reveal(arguments.outcome_lock, outcome_artifacts=arguments.outcome))
        return 0

    if arguments.command == "preregistration-build":
        from .preregistration import build_preregistration_bundle

        outputs = build_preregistration_bundle(
            arguments.protocol,
            arguments.outcome_lock,
            arguments.model_audit,
            arguments.method,
            arguments.output_dir,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "site-build":
        from .benchmark_site import build_benchmark_site

        outputs = build_benchmark_site(arguments.config, arguments.output_dir)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "mavedb-freeze-external":
        from .external_validation import freeze_external_panel

        outputs = freeze_external_panel(
            arguments.proteingym_reference,
            arguments.output_dir,
            binary=arguments.mmseqs_binary,
            threads=arguments.threads,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "mavedb-download-external":
        from .external_validation import download_selected_score_tables

        outputs = download_selected_score_tables(arguments.protocol, arguments.output_dir)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "mavedb-evaluate-external":
        from .external_validation import run_external_validation

        outputs = run_external_validation(
            arguments.protocol_dir,
            arguments.score_dir,
            arguments.work_dir,
            arguments.output_dir,
            device=arguments.device,
            batch_size=arguments.batch_size,
            bootstrap_repeats=arguments.bootstrap_repeats,
            bootstrap_seed=arguments.bootstrap_seed,
            reuse_predictions=arguments.reuse_predictions,
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "mavedb-external-figure":
        import pandas as pd

        from .external_visualize import render_external_figure

        protocol = json.loads(arguments.protocol.read_text())
        output = render_external_figure(
            protocol,
            pd.read_csv(arguments.assay_audit),
            pd.read_csv(arguments.bootstrap_summary),
            pd.read_csv(arguments.protein_metrics),
            arguments.output,
        )
        print(output)
        return 0

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
        summarize_generalization_gaps(gaps).to_csv(outputs["gap_summary"], index=False)
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
        scores = esm2_wild_type_marginals(wild_types.item(), frame["mutation_codes"].astype(str))
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(arguments.output, index=False)
        print(arguments.output)
        return 0

    if arguments.command == "proteingym-download":
        outputs = download_proteingym(
            arguments.destination,
            include_zero_shot_scores=arguments.include_zero_shot_scores,
            include_supervised_scores=arguments.include_supervised_scores,
            include_structures=arguments.include_structures,
        )
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
            workers=arguments.workers,
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

        from .modern_zero_shot import MODERN_ZERO_SHOT_MODELS

        eligibility = pd.read_csv(arguments.eligibility)
        artifact_paths = list(dict.fromkeys([arguments.eligibility, *arguments.artifact]))
        inputs = {
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
        }
        if arguments.supervised_archive:
            inputs["official_supervised_predictions"] = {
                "path": arguments.supervised_archive,
                "source": PROTEINGYM_SUPERVISED_ARCHIVE_URL,
                "version": PROTEINGYM_VERSION,
            }
        if arguments.embedding_index:
            inputs["esm2_embedding_index"] = {
                "path": arguments.embedding_index,
                "source": "locally cached frozen ESM-2 residue representations",
                "version": "esm2_t6_8M_UR50D",
            }
        if arguments.structure_archive:
            inputs["official_alphafold_structures"] = {
                "path": arguments.structure_archive,
                "source": PROTEINGYM_STRUCTURE_ARCHIVE_URL,
                "version": PROTEINGYM_VERSION,
            }
        manifest = build_collection_manifest(
            repository_root=Path.cwd(),
            inputs=inputs,
            protocol={
                "cohort": "finite single substitutions",
                "eligibility": EligibilityCriteria().to_dict(),
                "audited_assays": len(eligibility),
                "eligible_assays": int(eligibility["eligible"].sum()),
                "splits": [
                    "random_variant",
                    "position_holdout",
                    "modulo_position",
                    "contiguous_position",
                ],
                "repeated_splits": {
                    "additive_and_zero_shot": {
                        "start_seed": arguments.start_seed,
                        "repeats": arguments.repeats,
                    },
                    "local_embedding_probe": {
                        "start_seed": arguments.start_seed,
                        "repeats": arguments.probe_repeats,
                    },
                },
                "heldout_grouped_evaluation": {
                    "folds": arguments.heldout_folds,
                    "repeats": arguments.heldout_repeats,
                    "start_seed": 2026,
                },
                "heldout_sequence_family": {
                    "folds": arguments.family_folds,
                    "minimum_sequence_identity": arguments.family_identity_threshold,
                    "minimum_bidirectional_coverage": arguments.family_coverage_threshold,
                    "sequence_scope": "ProteinGym MSA_start:MSA_end assayed segment",
                    "clustering": "MMseqs2 exhaustive all-versus-all connected components",
                },
                "heldout_sequence_structure_family": {
                    "folds": arguments.structure_family_folds,
                    "minimum_reciprocal_tm_score": arguments.structure_tm_threshold,
                    "minimum_bidirectional_coverage": arguments.structure_coverage_threshold,
                    "minimum_reciprocal_homology_probability": (
                        arguments.structure_probability_threshold
                    ),
                    "structure_source": "official ProteinGym AlphaFold archive",
                    "clustering": (
                        "sequence-family graph union reciprocal Foldseek structure-homology "
                        "connected components"
                    ),
                },
                "heldout_curated_family": {
                    "folds": arguments.curated_family_folds,
                    "repeats": arguments.heldout_repeats,
                    "minimum_coordinate_mapping_coverage": (arguments.curated_mapping_threshold),
                    "minimum_domain_overlap_fraction_of_shorter": (
                        arguments.curated_overlap_threshold
                    ),
                    "primary_grouping": "Pfam family",
                    "sensitivity_grouping": "Pfam clan",
                    "clustering": (
                        "sequence/structure graph union current InterPro Pfam components"
                    ),
                },
                "crossover_group_folds": arguments.crossover_folds,
                "calibration_fraction": 0.2,
                "nominal_coverage": 0.8,
                "bootstrap_units": {
                    "model_summaries": "UniProt_ID",
                    "holdout_protocol_comparisons": "alternative family_id",
                },
                "bootstrap_repeats": arguments.bootstrap_repeats,
                "supervised_models": [
                    "mean",
                    "biophysical_ridge",
                    "additive_ridge",
                    "esm2_residue_ridge_probe",
                    "ProteinNPT (official out-of-fold)",
                    "Kermut (official out-of-fold)",
                    "cross_protein_ridge",
                    "cross_protein_histgb",
                ],
                "calibration_methods": [
                    "standard_split",
                    "mondrian_substitution",
                    "position_distance_scaled",
                ],
                "zero_shot_models": {
                    "esm_scaling_analysis": list(DEFAULT_ESM_MODELS),
                    "modern_paired_landscape": {
                        name: {"score_column": column, "modality": modality}
                        for name, (column, modality) in MODERN_ZERO_SHOT_MODELS.items()
                    },
                },
                "zero_shot_note": (
                    "Fixed-score subset differences are not supervised generalization gaps"
                ),
            },
            artifact_paths=artifact_paths,
            source_revision=arguments.source_revision,
        )
        print(write_manifest(manifest, arguments.output))
        return 0

    if arguments.command == "proteingym-figure":
        import pandas as pd

        output = render_multiprotein_figure(
            pd.read_csv(arguments.supervised_assays),
            pd.read_csv(arguments.supervised_aggregate),
            pd.read_csv(arguments.esm_aggregate),
            arguments.output,
        )
        print(output)
        return 0

    if arguments.command == "proteingym-official-supervised":
        import pandas as pd

        eligibility = pd.read_csv(arguments.eligibility)
        runs, audit = run_official_supervised_benchmark(
            arguments.source_archive,
            arguments.supervised_archive,
            arguments.reference,
            eligibility,
        )
        summary = summarize_official_supervised(runs, bootstrap_repeats=arguments.bootstrap_repeats)
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "audit": output_dir / "official-supervised-audit.csv",
            "runs": output_dir / "official-supervised-runs.csv",
            "summary": output_dir / "official-supervised-summary.csv",
        }
        audit.to_csv(outputs["audit"], index=False)
        runs.to_csv(outputs["runs"], index=False)
        summary.to_csv(outputs["summary"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-esm2-embeddings":
        import pandas as pd

        from .esm_embeddings import build_embedding_cache

        index = build_embedding_cache(
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            arguments.output_dir,
            model_name=arguments.model,
            device=arguments.device,
        )
        print(
            json.dumps(
                {
                    "index": str(arguments.output_dir / "index.csv"),
                    "assays": int(index["assay_id"].nunique()),
                    "proteins": int(index["uniprot_id"].nunique()),
                    "distinct_sequences": int(index["sequence_sha256"].nunique()),
                    "model": arguments.model,
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "proteingym-embedding-probe":
        import pandas as pd

        from .embedding_probe import (
            run_embedding_probe_benchmark,
            summarize_embedding_probe,
            summarize_probe_risk_coverage,
        )

        metrics, risks = run_embedding_probe_benchmark(
            arguments.archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            pd.read_csv(arguments.embedding_index),
            start_seed=arguments.start_seed,
            repeats=arguments.repeats,
            workers=arguments.workers,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "runs": output_dir / "embedding-probe-runs.csv",
            "summary": output_dir / "embedding-probe-summary.csv",
            "risk_coverage": output_dir / "embedding-probe-risk-coverage.csv",
            "risk_summary": output_dir / "embedding-probe-risk-summary.csv",
        }
        metrics.to_csv(outputs["runs"], index=False)
        summarize_embedding_probe(metrics).to_csv(outputs["summary"], index=False)
        risks.to_csv(outputs["risk_coverage"], index=False)
        summarize_probe_risk_coverage(risks).to_csv(outputs["risk_summary"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-heldout-protein":
        import pandas as pd

        from .cross_protein import (
            build_cross_protein_dataset,
            evaluate_held_out_proteins,
            summarize_grouped_repeat_estimates,
            summarize_held_out_proteins,
            summarize_heldout_risk_coverage,
        )

        dataset = build_cross_protein_dataset(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            max_variants_per_assay=arguments.max_variants_per_assay,
        )
        assays, risks, predictions = evaluate_held_out_proteins(
            dataset, folds=arguments.folds, repeats=arguments.repeats
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assays": output_dir / "heldout-protein-assays.csv",
            "summary": output_dir / "heldout-protein-summary.csv",
            "repeat_estimates": output_dir / "heldout-protein-repeat-estimates.csv",
            "risk_coverage": output_dir / "heldout-protein-risk-coverage.csv",
            "risk_summary": output_dir / "heldout-protein-risk-summary.csv",
            "predictions": output_dir / "heldout-protein-predictions.csv.gz",
        }
        assays.to_csv(outputs["assays"], index=False)
        summarize_held_out_proteins(assays).to_csv(outputs["summary"], index=False)
        summarize_grouped_repeat_estimates(assays).to_csv(outputs["repeat_estimates"], index=False)
        risks.to_csv(outputs["risk_coverage"], index=False)
        summarize_heldout_risk_coverage(risks).to_csv(outputs["risk_summary"], index=False)
        predictions.to_csv(outputs["predictions"], index=False, compression="gzip")
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-family-clusters":
        import pandas as pd

        from .family_clusters import build_sequence_family_clusters

        result = build_sequence_family_clusters(
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            identity_threshold=arguments.identity_threshold,
            coverage_threshold=arguments.coverage_threshold,
            search_identity_floor=arguments.search_identity_floor,
            search_coverage_floor=arguments.search_coverage_floor,
            binary=arguments.mmseqs_binary,
            threads=arguments.threads,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assignments": output_dir / "sequence-family-assignments.csv",
            "alignments": output_dir / "sequence-family-alignments.csv",
            "sensitivity": output_dir / "sequence-family-sensitivity.csv",
            "audit": output_dir / "sequence-family-audit.csv",
        }
        result.assignments.to_csv(outputs["assignments"], index=False)
        result.alignments.to_csv(outputs["alignments"], index=False)
        result.sensitivity.to_csv(outputs["sensitivity"], index=False)
        result.audit.to_csv(outputs["audit"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-heldout-family":
        import pandas as pd

        from .cross_protein import (
            build_cross_protein_dataset,
            compare_holdout_protocols,
            evaluate_held_out_families,
            summarize_grouped_repeat_estimates,
            summarize_held_out_proteins,
            summarize_heldout_risk_coverage,
        )

        dataset = build_cross_protein_dataset(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            max_variants_per_assay=arguments.max_variants_per_assay,
        )
        assays, risks, predictions = evaluate_held_out_families(
            dataset,
            pd.read_csv(arguments.family_assignments),
            folds=arguments.folds,
            repeats=arguments.repeats,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assays": output_dir / "heldout-family-assays.csv",
            "summary": output_dir / "heldout-family-summary.csv",
            "repeat_estimates": output_dir / "heldout-family-repeat-estimates.csv",
            "risk_coverage": output_dir / "heldout-family-risk-coverage.csv",
            "risk_summary": output_dir / "heldout-family-risk-summary.csv",
            "predictions": output_dir / "heldout-family-predictions.csv.gz",
        }
        assays.to_csv(outputs["assays"], index=False)
        summarize_held_out_proteins(assays).to_csv(outputs["summary"], index=False)
        summarize_grouped_repeat_estimates(assays).to_csv(outputs["repeat_estimates"], index=False)
        risks.to_csv(outputs["risk_coverage"], index=False)
        summarize_heldout_risk_coverage(risks).to_csv(outputs["risk_summary"], index=False)
        predictions.to_csv(outputs["predictions"], index=False, compression="gzip")
        if arguments.protein_assays:
            outputs["comparison"] = output_dir / "heldout-family-comparison.csv"
            compare_holdout_protocols(
                pd.read_csv(arguments.protein_assays),
                assays,
                bootstrap_repeats=arguments.bootstrap_repeats,
            ).to_csv(outputs["comparison"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-structure-clusters":
        import pandas as pd

        from .structure_clusters import build_sequence_structure_family_clusters

        result = build_sequence_structure_family_clusters(
            arguments.structure_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            pd.read_csv(arguments.sequence_assignments),
            minimum_tm_score=arguments.minimum_tm_score,
            minimum_coverage=arguments.minimum_coverage,
            minimum_homology_probability=arguments.minimum_homology_probability,
            binary=arguments.foldseek_binary,
            threads=arguments.threads,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assignments": output_dir / "sequence-structure-family-assignments.csv",
            "structure_inputs": output_dir / "structure-input-audit.csv",
            "alignments": output_dir / "structure-family-alignments.csv",
            "sensitivity": output_dir / "structure-family-sensitivity.csv",
            "audit": output_dir / "structure-family-audit.csv",
        }
        result.assignments.to_csv(outputs["assignments"], index=False)
        result.structure_inputs.to_csv(outputs["structure_inputs"], index=False)
        result.alignments.to_csv(outputs["alignments"], index=False)
        result.sensitivity.to_csv(outputs["sensitivity"], index=False)
        result.audit.to_csv(outputs["audit"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-heldout-structure-family":
        import pandas as pd

        from .cross_protein import (
            build_cross_protein_dataset,
            compare_holdout_protocols,
            evaluate_held_out_families,
            summarize_grouped_repeat_estimates,
            summarize_held_out_proteins,
            summarize_heldout_risk_coverage,
        )

        dataset = build_cross_protein_dataset(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            max_variants_per_assay=arguments.max_variants_per_assay,
        )
        assays, risks, predictions = evaluate_held_out_families(
            dataset,
            pd.read_csv(arguments.family_assignments),
            folds=arguments.folds,
            repeats=arguments.repeats,
            group_name="sequence_structure_family",
            evaluation_type="held_out_sequence_structure_family",
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assays": output_dir / "heldout-structure-family-assays.csv",
            "summary": output_dir / "heldout-structure-family-summary.csv",
            "repeat_estimates": output_dir / "heldout-structure-family-repeat-estimates.csv",
            "risk_coverage": output_dir / "heldout-structure-family-risk-coverage.csv",
            "risk_summary": output_dir / "heldout-structure-family-risk-summary.csv",
            "predictions": output_dir / "heldout-structure-family-predictions.csv.gz",
        }
        assays.to_csv(outputs["assays"], index=False)
        summarize_held_out_proteins(assays).to_csv(outputs["summary"], index=False)
        summarize_grouped_repeat_estimates(assays).to_csv(outputs["repeat_estimates"], index=False)
        risks.to_csv(outputs["risk_coverage"], index=False)
        summarize_heldout_risk_coverage(risks).to_csv(outputs["risk_summary"], index=False)
        predictions.to_csv(outputs["predictions"], index=False, compression="gzip")
        if arguments.protein_assays:
            outputs["protein_comparison"] = output_dir / "heldout-structure-family-vs-protein.csv"
            compare_holdout_protocols(
                pd.read_csv(arguments.protein_assays),
                assays,
                bootstrap_repeats=arguments.bootstrap_repeats,
                baseline_label="heldout_protein",
                alternative_label="heldout_structure_family",
            ).to_csv(outputs["protein_comparison"], index=False)
        if arguments.sequence_family_assays:
            outputs["sequence_comparison"] = (
                output_dir / "heldout-structure-family-vs-sequence-family.csv"
            )
            compare_holdout_protocols(
                pd.read_csv(arguments.sequence_family_assays),
                assays,
                bootstrap_repeats=arguments.bootstrap_repeats,
                baseline_label="heldout_sequence_family",
                alternative_label="heldout_structure_family",
            ).to_csv(outputs["sequence_comparison"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-curated-families":
        import pandas as pd

        from .curated_families import build_curated_family_clusters

        result = build_curated_family_clusters(
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            pd.read_csv(arguments.base_assignments),
            arguments.cache_dir,
            minimum_mapping_coverage=arguments.minimum_mapping_coverage,
            minimum_overlap=arguments.minimum_domain_overlap,
            workers=arguments.workers,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assignments": output_dir / "curated-family-assignments.csv",
            "clan_assignments": output_dir / "curated-clan-family-assignments.csv",
            "uniprot_mapping": output_dir / "curated-uniprot-mapping.csv",
            "coordinate_mapping": output_dir / "curated-coordinate-mapping.csv",
            "domain_overlaps": output_dir / "curated-pfam-domain-overlaps.csv",
            "edges": output_dir / "curated-family-edges.csv",
            "clan_edges": output_dir / "curated-clan-family-edges.csv",
            "audit": output_dir / "curated-family-audit.csv",
        }
        result.assignments.to_csv(outputs["assignments"], index=False)
        result.clan_assignments.to_csv(outputs["clan_assignments"], index=False)
        result.uniprot_mapping.to_csv(outputs["uniprot_mapping"], index=False)
        result.coordinate_mapping.to_csv(outputs["coordinate_mapping"], index=False)
        result.domain_overlaps.to_csv(outputs["domain_overlaps"], index=False)
        result.curated_edges.to_csv(outputs["edges"], index=False)
        result.clan_edges.to_csv(outputs["clan_edges"], index=False)
        result.audit.to_csv(outputs["audit"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-heldout-curated-family":
        import pandas as pd

        from .cross_protein import (
            build_cross_protein_dataset,
            compare_feature_ablation,
            compare_holdout_protocols,
            evaluate_held_out_families,
            summarize_grouped_repeat_estimates,
            summarize_held_out_proteins,
            summarize_heldout_risk_coverage,
        )

        dataset = build_cross_protein_dataset(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            max_variants_per_assay=arguments.max_variants_per_assay,
        )
        assays, risks, predictions = evaluate_held_out_families(
            dataset,
            pd.read_csv(arguments.family_assignments),
            folds=arguments.folds,
            repeats=arguments.repeats,
            group_name="curated_sequence_structure_family",
            evaluation_type="held_out_curated_sequence_structure_family",
            include_feature_ablation=arguments.feature_ablation,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "assays": output_dir / "heldout-curated-family-assays.csv",
            "summary": output_dir / "heldout-curated-family-summary.csv",
            "repeat_estimates": output_dir / "heldout-curated-family-repeat-estimates.csv",
            "risk_coverage": output_dir / "heldout-curated-family-risk-coverage.csv",
            "risk_summary": output_dir / "heldout-curated-family-risk-summary.csv",
            "predictions": output_dir / "heldout-curated-family-predictions.csv.gz",
        }
        assays.to_csv(outputs["assays"], index=False)
        summarize_held_out_proteins(assays).to_csv(outputs["summary"], index=False)
        summarize_grouped_repeat_estimates(assays).to_csv(outputs["repeat_estimates"], index=False)
        risks.to_csv(outputs["risk_coverage"], index=False)
        summarize_heldout_risk_coverage(risks).to_csv(outputs["risk_summary"], index=False)
        predictions.to_csv(outputs["predictions"], index=False, compression="gzip")
        if arguments.feature_ablation:
            outputs["feature_ablation"] = output_dir / "heldout-curated-feature-ablation.csv"
            compare_feature_ablation(assays, bootstrap_repeats=arguments.bootstrap_repeats).to_csv(
                outputs["feature_ablation"], index=False
            )
        if arguments.protein_assays:
            outputs["protein_comparison"] = output_dir / "heldout-curated-family-vs-protein.csv"
            compare_holdout_protocols(
                pd.read_csv(arguments.protein_assays),
                assays,
                bootstrap_repeats=arguments.bootstrap_repeats,
                baseline_label="heldout_protein",
                alternative_label="heldout_curated_family",
            ).to_csv(outputs["protein_comparison"], index=False)
        if arguments.structure_family_assays:
            outputs["structure_comparison"] = (
                output_dir / "heldout-curated-family-vs-structure-family.csv"
            )
            compare_holdout_protocols(
                pd.read_csv(arguments.structure_family_assays),
                assays,
                bootstrap_repeats=arguments.bootstrap_repeats,
                baseline_label="heldout_structure_family",
                alternative_label="heldout_curated_family",
            ).to_csv(outputs["structure_comparison"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-modern-zero-shot":
        import pandas as pd

        from .modern_zero_shot import (
            compare_modern_to_baseline,
            run_modern_zero_shot_landscape,
            summarize_modern_zero_shot,
        )

        runs, audit = run_modern_zero_shot_landscape(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            minimum_common_coverage=arguments.minimum_common_coverage,
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "audit": output_dir / "modern-zero-shot-audit.csv",
            "runs": output_dir / "modern-zero-shot-runs.csv",
            "summary": output_dir / "modern-zero-shot-summary.csv",
            "comparison": output_dir / "modern-zero-shot-vs-esm2.csv",
        }
        audit.to_csv(outputs["audit"], index=False)
        runs.to_csv(outputs["runs"], index=False)
        summarize_modern_zero_shot(runs, bootstrap_repeats=arguments.bootstrap_repeats).to_csv(
            outputs["summary"], index=False
        )
        compare_modern_to_baseline(runs, bootstrap_repeats=arguments.bootstrap_repeats).to_csv(
            outputs["comparison"], index=False
        )
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-crossover":
        import pandas as pd

        from .crossover import build_crossover_examples, evaluate_crossover_predictor

        examples = build_crossover_examples(
            arguments.source_archive,
            arguments.score_archive,
            arguments.reference,
            pd.read_csv(arguments.eligibility),
            pd.read_csv(arguments.supervised_runs),
            pd.read_csv(arguments.zero_shot_runs),
            supervised_model=arguments.supervised_model,
        )
        predictions, summary, coefficients = evaluate_crossover_predictor(
            examples, folds=arguments.folds
        )
        output_dir = arguments.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "examples": output_dir / "crossover-examples.csv",
            "predictions": output_dir / "crossover-heldout-predictions.csv",
            "summary": output_dir / "crossover-summary.csv",
            "coefficients": output_dir / "crossover-logistic-coefficients.csv",
        }
        examples.to_csv(outputs["examples"], index=False)
        predictions.to_csv(outputs["predictions"], index=False)
        summary.to_csv(outputs["summary"], index=False)
        coefficients.to_csv(outputs["coefficients"], index=False)
        print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
        return 0

    if arguments.command == "proteingym-extended-figure":
        import pandas as pd

        output = render_extended_figure(
            pd.read_csv(arguments.official_summary),
            pd.read_csv(arguments.probe_summary),
            pd.read_csv(arguments.heldout_summary),
            pd.read_csv(arguments.crossover_summary),
            arguments.output,
            heldout_family=(
                pd.read_csv(arguments.heldout_family_summary)
                if arguments.heldout_family_summary
                else None
            ),
            heldout_structure_family=(
                pd.read_csv(arguments.heldout_structure_family_summary)
                if arguments.heldout_structure_family_summary
                else None
            ),
        )
        print(output)
        return 0

    if arguments.command == "proteingym-research-figure":
        import pandas as pd

        from .research_visualize import render_research_figure

        output = render_research_figure(
            pd.read_csv(arguments.modern_summary),
            pd.read_csv(arguments.sequence_audit),
            pd.read_csv(arguments.structure_audit),
            pd.read_csv(arguments.curated_audit),
            pd.read_csv(arguments.heldout_protein),
            pd.read_csv(arguments.heldout_family),
            pd.read_csv(arguments.heldout_structure),
            pd.read_csv(arguments.heldout_curated),
            arguments.output,
        )
        print(output)
        return 0

    raise AssertionError(f"Unhandled command: {arguments.command}")
