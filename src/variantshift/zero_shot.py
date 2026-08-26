"""Audited evaluation of official ProteinGym zero-shot model scores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    read_assay_member,
    read_reference_index,
)
from .robustness import seed_schedule
from .splits import leakage_audit, position_holdout_split, random_variant_split

DEFAULT_ESM_MODELS = (
    "ESM1v_ensemble",
    "ESM2_8M",
    "ESM2_35M",
    "ESM2_150M",
    "ESM2_650M",
    "ESM2_3B",
    "ESM2_15B",
)


def _read_score_member(
    archive: ZipFile,
    member: str,
    model_columns: Sequence[str],
) -> pd.DataFrame:
    required = ["mutant", "DMS_score", *model_columns]
    with archive.open(member) as handle:
        try:
            frame = pd.read_csv(handle, usecols=required, low_memory=False)
        except ValueError as error:
            raise ValueError(f"Score file {member} is missing a requested column") from error
    return frame


def _align_scores(
    source: pd.DataFrame,
    scores: pd.DataFrame,
    model_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    score_depth = scores["mutant"].astype(str).str.count(":") + 1
    selected = scores.loc[score_depth.eq(1), ["mutant", "DMS_score", *model_columns]].copy()
    selected["mutation_codes"] = selected.pop("mutant").astype(str).str.replace(
        ":", "/", regex=False
    )
    selected = selected.rename(columns={"DMS_score": "score_archive_dms_score"})
    duplicates = int(selected["mutation_codes"].duplicated().sum())
    aligned = source.merge(selected, on="mutation_codes", how="left", validate="one_to_one")
    observed = aligned["DMS_score"].to_numpy(dtype=float)
    archived = pd.to_numeric(aligned["score_archive_dms_score"], errors="coerce").to_numpy(
        dtype=float
    )
    finite_dms = np.isfinite(archived)
    maximum_difference = (
        float(np.max(np.abs(observed[finite_dms] - archived[finite_dms])))
        if finite_dms.any()
        else float("nan")
    )
    common_finite = np.ones(len(aligned), dtype=bool)
    for model in model_columns:
        aligned[model] = pd.to_numeric(aligned[model], errors="coerce")
        common_finite &= np.isfinite(aligned[model].to_numpy(dtype=float))
    audit = {
        "source_single_variants": len(source),
        "score_single_variants": len(selected),
        "matched_dms_scores": int(finite_dms.sum()),
        "duplicate_score_variants": duplicates,
        "dms_score_max_abs_difference": maximum_difference,
        "common_finite_scores": int(common_finite.sum()),
        "common_score_coverage": float(common_finite.mean()),
    }
    aligned["common_finite_scores"] = common_finite
    return aligned, audit


def evaluate_zero_shot_assay(
    frame: pd.DataFrame,
    *,
    model_columns: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Evaluate fixed zero-shot scores on matched random and position subsets."""
    target = frame["DMS_score"].to_numpy(dtype=float)
    target_scale = float(np.std(target))
    results: list[dict[str, object]] = []
    for seed in seeds:
        splits = (
            random_variant_split(frame, seed=seed),
            position_holdout_split(frame, seed=seed),
        )
        for split in splits:
            audit = leakage_audit(frame, split)
            observed = target[split.test_indices]
            for model in model_columns:
                prediction = frame.iloc[split.test_indices][model].to_numpy(dtype=float)
                metrics = regression_metrics(observed, prediction)
                results.append(
                    {
                        "assay_id": str(frame["assay_id"].iat[0]),
                        "uniprot_id": str(frame["uniprot_id"].iat[0]),
                        "taxon": str(frame["taxon"].iat[0]),
                        "coarse_selection_type": str(
                            frame["coarse_selection_type"].iat[0]
                        ),
                        "seed": int(seed),
                        "split": split.name,
                        "model": model,
                        "evaluation_type": "zero_shot_fixed_scores",
                        **metrics.to_dict(),
                        "normalized_rmse": metrics.rmse / target_scale
                        if target_scale > 1e-12
                        else np.nan,
                        "test_rows": len(split.test_indices),
                        "exact_variant_overlap": audit["exact_variant_overlap"],
                        "shared_position_count": audit["shared_position_count"],
                    }
                )
    return pd.DataFrame(results)


def run_zero_shot_benchmark(
    source_archive_path: Path,
    score_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    *,
    model_columns: Sequence[str] = DEFAULT_ESM_MODELS,
    start_seed: int = 42,
    repeats: int = 10,
    min_common_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit and evaluate official scores for every supervised-eligible assay."""
    if not 0 < min_common_coverage <= 1:
        raise ValueError("Minimum common score coverage must lie in (0, 1]")
    if not model_columns:
        raise ValueError("At least one zero-shot model column is required")
    if len(model_columns) != len(set(model_columns)):
        raise ValueError("Zero-shot model columns must be unique")

    seeds = seed_schedule(start_seed=start_seed, repeats=repeats)
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    audit_rows: list[dict[str, object]] = []
    runs: list[pd.DataFrame] = []

    with ZipFile(source_archive_path) as source_archive, ZipFile(
        score_archive_path
    ) as score_archive:
        source_members = _archive_members(source_archive)
        score_members = _archive_members(score_archive)
        for assay_id in eligible_ids:
            metadata = reference.loc[assay_id]
            filename = str(metadata["DMS_filename"])
            reasons: list[str] = []
            if filename not in source_members:
                reasons.append("missing_source_assay")
            if filename not in score_members:
                reasons.append("missing_score_assay")
            if reasons:
                audit_rows.append(
                    {
                        "assay_id": assay_id,
                        "uniprot_id": str(metadata["UniProt_ID"]),
                        "eligible_for_zero_shot": False,
                        "exclusion_reasons": ";".join(reasons),
                    }
                )
                continue

            source = canonicalize_assay(
                read_assay_member(source_archive, source_members[filename]), metadata
            )
            score_frame = _read_score_member(
                score_archive, score_members[filename], model_columns
            )
            aligned, audit = _align_scores(source, score_frame, model_columns)
            if audit["duplicate_score_variants"]:
                reasons.append("duplicate_score_variants")
            if audit["matched_dms_scores"] != audit["source_single_variants"]:
                reasons.append("incomplete_variant_join")
            if not np.isclose(audit["dms_score_max_abs_difference"], 0.0, atol=1e-10):
                reasons.append("dms_score_mismatch")
            if audit["common_score_coverage"] < min_common_coverage:
                reasons.append("insufficient_common_score_coverage")

            audit_rows.append(
                {
                    "assay_id": assay_id,
                    "uniprot_id": str(metadata["UniProt_ID"]),
                    **audit,
                    **{
                        f"{model}_finite_scores": int(
                            np.isfinite(aligned[model].to_numpy(dtype=float)).sum()
                        )
                        for model in model_columns
                    },
                    "eligible_for_zero_shot": not reasons,
                    "exclusion_reasons": ";".join(reasons),
                }
            )
            if reasons:
                continue
            complete = aligned.loc[aligned["common_finite_scores"]].reset_index(drop=True)
            runs.append(
                evaluate_zero_shot_assay(
                    complete,
                    model_columns=model_columns,
                    seeds=seeds,
                )
            )

    audit_frame = pd.DataFrame(audit_rows).sort_values("assay_id").reset_index(drop=True)
    if not runs:
        raise ValueError("No assays passed the zero-shot score audit")
    run_frame = pd.concat(runs, ignore_index=True).sort_values(
        ["assay_id", "seed", "split", "model"]
    ).reset_index(drop=True)
    return run_frame, audit_frame


def zero_shot_subset_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair zero-shot performance across random and unseen-position test subsets."""
    identifiers = (
        "assay_id",
        "uniprot_id",
        "taxon",
        "coarse_selection_type",
        "seed",
        "model",
    )
    metrics = ("spearman", "normalized_rmse")
    pivot = runs.pivot(index=list(identifiers), columns="split", values=list(metrics))
    output = pivot.index.to_frame(index=False)
    for metric in metrics:
        output[f"random_{metric}"] = pivot[(metric, "random_variant")].to_numpy()
        output[f"position_{metric}"] = pivot[(metric, "position_holdout")].to_numpy()
        output[f"subset_{metric}_difference"] = (
            output[f"random_{metric}"] - output[f"position_{metric}"]
        )
    return output.sort_values(["model", "uniprot_id", "assay_id", "seed"]).reset_index(
        drop=True
    )


def summarize_zero_shot(
    differences: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 2026,
) -> pd.DataFrame:
    """Aggregate fixed-score subset sensitivity with UniProt-level bootstrap intervals."""
    assay_level = (
        differences.groupby(["model", "uniprot_id", "assay_id"], as_index=False)
        .agg(
            random_spearman=("random_spearman", "mean"),
            position_spearman=("position_spearman", "mean"),
            subset_spearman_difference=("subset_spearman_difference", "mean"),
            subset_normalized_rmse_difference=(
                "subset_normalized_rmse_difference",
                "mean",
            ),
        )
    )
    protein_level = (
        assay_level.groupby(["model", "uniprot_id"], as_index=False)
        .agg(
            random_spearman=("random_spearman", "mean"),
            position_spearman=("position_spearman", "mean"),
            subset_spearman_difference=("subset_spearman_difference", "mean"),
            subset_normalized_rmse_difference=(
                "subset_normalized_rmse_difference",
                "mean",
            ),
        )
    )

    rows: list[dict[str, object]] = []
    for offset, (model, group) in enumerate(protein_level.groupby("model", sort=True)):
        values = group["subset_spearman_difference"].to_numpy(dtype=float)
        rng = np.random.default_rng(bootstrap_seed + offset)
        indices = rng.integers(0, len(values), size=(bootstrap_repeats, len(values)))
        bootstrapped = values[indices].mean(axis=1)
        rows.append(
            {
                "model": model,
                "evaluation_type": "zero_shot_fixed_scores",
                "n_assays": int(
                    assay_level.loc[assay_level["model"].eq(model), "assay_id"].nunique()
                ),
                "n_proteins": int(group["uniprot_id"].nunique()),
                "random_spearman_mean": float(group["random_spearman"].mean()),
                "position_spearman_mean": float(group["position_spearman"].mean()),
                "subset_spearman_difference_mean": float(values.mean()),
                "subset_spearman_difference_ci_low": float(
                    np.quantile(bootstrapped, 0.025)
                ),
                "subset_spearman_difference_ci_high": float(
                    np.quantile(bootstrapped, 0.975)
                ),
                "subset_normalized_rmse_difference_mean": float(
                    group["subset_normalized_rmse_difference"].mean()
                ),
                "bootstrap_unit": "UniProt_ID",
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def summarize_zero_shot_assays(differences: pd.DataFrame) -> pd.DataFrame:
    """Produce one repeated-subset summary row per assay and zero-shot model."""
    return (
        differences.groupby(
            ["assay_id", "uniprot_id", "taxon", "coarse_selection_type", "model"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            random_spearman=("random_spearman", "mean"),
            position_spearman=("position_spearman", "mean"),
            subset_spearman_difference=("subset_spearman_difference", "mean"),
            random_normalized_rmse=("random_normalized_rmse", "mean"),
            position_normalized_rmse=("position_normalized_rmse", "mean"),
            subset_normalized_rmse_difference=(
                "subset_normalized_rmse_difference",
                "mean",
            ),
        )
        .sort_values(
            ["model", "subset_spearman_difference"], ascending=[True, False]
        )
        .reset_index(drop=True)
    )
