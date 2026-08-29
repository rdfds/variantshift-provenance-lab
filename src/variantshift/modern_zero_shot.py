"""Paired modern zero-shot model landscape on the audited ProteinGym cohort."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .metrics import regression_metrics, top_selection_metrics
from .multiprotein import _bootstrap_mean_interval
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    read_assay_member,
    read_reference_index,
)
from .zero_shot import _align_scores, _read_score_member

MODERN_ZERO_SHOT_MODELS = {
    "esm2_650m": ("ESM2_650M", "single_sequence"),
    "esm3_open": ("ESM3", "sequence_structure_function"),
    "esmc_600m": ("ESMC-600M", "single_sequence"),
    "progen3_3b": ("Progen3_3b", "single_sequence"),
    "xtrimopglm_100b": ("xTrimoPGLM-100B-int4", "single_sequence"),
    "saprot_650m": ("SaProt_650M_AF2", "sequence_structure"),
    "prosst_2048": ("ProSST-2048", "sequence_structure"),
    "s3f_msa": ("S3F_MSA", "structure_msa"),
    "venusrem": ("VenusREM", "structure_msa"),
    "siterm": ("SiteRM", "structure_msa"),
    "gemme": ("GEMME", "msa"),
    "tranception_l": ("Tranception_L", "msa_retrieval"),
}


def run_modern_zero_shot_landscape(
    source_archive_path: Path,
    score_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    *,
    models: dict[str, tuple[str, str]] | None = None,
    minimum_common_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score predeclared modern models on exactly matched finite variants per assay."""
    models = models or MODERN_ZERO_SHOT_MODELS
    if not models:
        raise ValueError("At least one modern zero-shot model is required")
    if not 0 < minimum_common_coverage <= 1:
        raise ValueError("Minimum common coverage must be in (0, 1]")
    columns = tuple(column for column, _ in models.values())
    if len(columns) != len(set(columns)):
        raise ValueError("Modern zero-shot score columns must be unique")
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    audit_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    with (
        ZipFile(source_archive_path) as source_archive,
        ZipFile(score_archive_path) as score_archive,
    ):
        source_members = _archive_members(source_archive)
        score_members = _archive_members(score_archive)
        for assay_id in eligible_ids:
            metadata = reference.loc[assay_id]
            filename = str(metadata["DMS_filename"])
            source = canonicalize_assay(
                read_assay_member(source_archive, source_members[filename]), metadata
            )
            scores = _read_score_member(score_archive, score_members[filename], columns)
            aligned, audit = _align_scores(source, scores, columns)
            common = aligned["common_finite_scores"].to_numpy(dtype=bool)
            common_coverage = float(common.mean())
            reasons = []
            if audit["duplicate_score_variants"]:
                reasons.append("duplicate_score_variants")
            if audit["matched_dms_scores"] != len(source):
                reasons.append("incomplete_variant_join")
            if not np.isclose(audit["dms_score_max_abs_difference"], 0.0, atol=1e-10):
                reasons.append("dms_score_mismatch")
            if common_coverage < minimum_common_coverage:
                reasons.append("insufficient_common_model_coverage")
            audit_rows.append(
                {
                    "assay_id": assay_id,
                    "uniprot_id": str(metadata["UniProt_ID"]),
                    **audit,
                    "common_model_coverage": common_coverage,
                    "eligible_for_paired_landscape": not reasons,
                    "exclusion_reasons": ";".join(reasons),
                }
            )
            if reasons:
                continue
            selected = aligned.loc[common]
            observed = selected["DMS_score"].to_numpy(dtype=float)
            for model_name, (column, modality) in models.items():
                predicted = selected[column].to_numpy(dtype=float)
                point = regression_metrics(observed, predicted)
                selection = top_selection_metrics(observed, predicted)
                result_rows.append(
                    {
                        "assay_id": assay_id,
                        "uniprot_id": str(metadata["UniProt_ID"]),
                        "taxon": str(metadata["taxon"]),
                        "coarse_selection_type": str(metadata["coarse_selection_type"]),
                        "model": model_name,
                        "score_column": column,
                        "modality": modality,
                        "evaluation_type": "official_zero_shot_paired_complete_case",
                        **point.to_dict(),
                        **selection,
                        "test_rows": len(selected),
                    }
                )
    audit_frame = pd.DataFrame(audit_rows).sort_values("assay_id").reset_index(drop=True)
    if not result_rows:
        raise ValueError("No assays passed the paired modern-model coverage audit")
    runs = pd.DataFrame(result_rows).sort_values(["model", "assay_id"]).reset_index(drop=True)
    return runs, audit_frame


def summarize_modern_zero_shot(
    runs: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Aggregate within UniProt and bootstrap proteins for each modern model."""
    metrics = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "best_variant_regret_sd",
    ]
    protein_level = runs.groupby(
        ["model", "score_column", "modality", "uniprot_id"], as_index=False
    )[metrics].mean()
    rows = []
    for (model, score_column, modality), group in protein_level.groupby(
        ["model", "score_column", "modality"], sort=True
    ):
        row: dict[str, object] = {
            "model": model,
            "score_column": score_column,
            "modality": modality,
            "n_assays": int(runs.loc[runs["model"].eq(model), "assay_id"].nunique()),
            "n_proteins": len(group),
        }
        for offset, metric in enumerate(metrics):
            values = group[metric].to_numpy(dtype=float)
            low, high = _bootstrap_mean_interval(
                values,
                repeats=bootstrap_repeats,
                rng=np.random.default_rng(seed + offset),
            )
            row[f"mean_{metric}"] = float(values.mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_spearman", ascending=False).reset_index(drop=True)


def compare_modern_to_baseline(
    runs: pd.DataFrame,
    *,
    baseline: str = "esm2_650m",
    bootstrap_repeats: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Paired UniProt bootstrap of every model against the declared baseline."""
    protein = runs.groupby(["model", "uniprot_id"], as_index=False)["spearman"].mean()
    pivot = protein.pivot(index="uniprot_id", columns="model", values="spearman")
    if baseline not in pivot:
        raise ValueError(f"Missing baseline model: {baseline}")
    if pivot.isna().any().any():
        raise ValueError("Modern model comparison is not exactly paired by UniProt")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(pivot), size=(bootstrap_repeats, len(pivot)))
    baseline_values = pivot[baseline].to_numpy(dtype=float)
    models = sorted(pivot.columns)
    bootstrap_means = np.column_stack(
        [pivot[model].to_numpy(dtype=float)[sample_indices].mean(axis=1) for model in models]
    )
    baseline_index = models.index(baseline)
    bootstrap_deltas = bootstrap_means - bootstrap_means[:, [baseline_index]]
    point_deltas = np.asarray(
        [pivot[model].mean() - pivot[baseline].mean() for model in models], dtype=float
    )
    nonbaseline = [index for index, model in enumerate(models) if model != baseline]
    maximum_centered_delta = np.max(
        np.abs(bootstrap_deltas[:, nonbaseline] - point_deltas[nonbaseline]), axis=1
    )
    simultaneous_radius = float(np.quantile(maximum_centered_delta, 0.95))
    bootstrap_ranks = np.argsort(np.argsort(-bootstrap_means, axis=1), axis=1) + 1
    best_indices = np.argmax(bootstrap_means, axis=1)
    rows = []
    for model_index, model in enumerate(models):
        values = pivot[model].to_numpy(dtype=float)
        delta = values - baseline_values
        bootstrap = bootstrap_deltas[:, model_index]
        point_delta = float(delta.mean())
        rows.append(
            {
                "model": model,
                "baseline": baseline,
                "n_proteins": len(pivot),
                "model_mean_spearman": float(values.mean()),
                "baseline_mean_spearman": float(baseline_values.mean()),
                "mean_paired_delta": point_delta,
                "delta_ci_low": float(np.quantile(bootstrap, 0.025)),
                "delta_ci_high": float(np.quantile(bootstrap, 0.975)),
                "simultaneous_delta_ci_low": point_delta - simultaneous_radius,
                "simultaneous_delta_ci_high": point_delta + simultaneous_radius,
                "probability_delta_above_zero": float(np.mean(bootstrap > 0)),
                "bootstrap_probability_best": float(np.mean(best_indices == model_index)),
                "mean_bootstrap_rank": float(bootstrap_ranks[:, model_index].mean()),
                "multiplicity_control": "95% paired-bootstrap max absolute centered delta",
                "bootstrap_unit": "UniProt_ID",
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("mean_paired_delta", ascending=False).reset_index(drop=True)
    )
