"""Audit ProteinGym's official out-of-fold supervised predictions.

These predictions were trained by the model authors under ProteinGym's published
five-fold protocols. VariantShift evaluates them as external artifacts and never
describes them as locally fitted models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
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

OFFICIAL_SUPERVISED_MODELS = {
    "esm1v_embedding_probe": "Embeddings - Augmented - ESM1v_predictions",
    "protein_npt": "ProteinNPT_predictions",
    "kermut": "Kermut_predictions",
}

OFFICIAL_SPLIT_DIRECTORIES = {
    "fold_random_5": "random_variant",
    "fold_modulo_5": "modulo_position",
    "fold_contiguous_5": "contiguous_position",
}


def _supervised_members(archive: ZipFile) -> dict[tuple[str, str], str]:
    members: dict[tuple[str, str], str] = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() != ".csv" or len(path.parts) < 2:
            continue
        split_directory = path.parts[-2]
        if split_directory not in OFFICIAL_SPLIT_DIRECTORIES:
            continue
        key = (split_directory, path.name)
        if key in members:
            raise ValueError(f"Duplicate supervised score member for {key}")
        members[key] = name
    return members


def _read_supervised_member(
    archive: ZipFile,
    member: str,
    model_columns: Sequence[str],
) -> pd.DataFrame:
    required = ["mutant", "DMS_score", "normalized_targets", *model_columns]
    with archive.open(member) as handle:
        try:
            return pd.read_csv(handle, usecols=required, low_memory=False)
        except ValueError as error:
            raise ValueError(f"Supervised score file {member} is missing a column") from error


def _align_supervised_scores(
    source: pd.DataFrame,
    scores: pd.DataFrame,
    model_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = scores.copy()
    selected["mutation_codes"] = selected.pop("mutant").astype(str).str.replace(
        ":", "/", regex=False
    )
    duplicates = int(selected["mutation_codes"].duplicated().sum())
    selected = selected.rename(columns={"DMS_score": "score_archive_dms_score"})
    aligned = source.merge(selected, on="mutation_codes", how="left", validate="one_to_one")
    source_score = aligned["DMS_score"].to_numpy(dtype=float)
    archived_score = pd.to_numeric(
        aligned["score_archive_dms_score"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_dms = np.isfinite(archived_score)
    maximum_difference = (
        float(np.max(np.abs(source_score[finite_dms] - archived_score[finite_dms])))
        if finite_dms.any()
        else np.nan
    )
    aligned["normalized_targets"] = pd.to_numeric(
        aligned["normalized_targets"], errors="coerce"
    )
    audit: dict[str, object] = {
        "source_variants": len(source),
        "score_variants": len(selected),
        "matched_variants": int(finite_dms.sum()),
        "duplicate_score_variants": duplicates,
        "dms_score_max_abs_difference": maximum_difference,
    }
    for column in model_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        audit[f"{column}_finite"] = int(np.isfinite(aligned[column]).sum())
    return aligned, audit


def run_official_supervised_benchmark(
    source_archive_path: Path,
    supervised_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    *,
    models: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit and score official ProteinGym out-of-fold prediction files."""
    models = models or OFFICIAL_SUPERVISED_MODELS
    if not models:
        raise ValueError("At least one official supervised model is required")
    model_columns = tuple(models.values())
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    audit_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    with ZipFile(source_archive_path) as source_archive, ZipFile(
        supervised_archive_path
    ) as supervised_archive:
        source_members = _archive_members(source_archive)
        supervised_members = _supervised_members(supervised_archive)
        for assay_id in eligible_ids:
            metadata = reference.loc[assay_id]
            filename = str(metadata["DMS_filename"])
            source = canonicalize_assay(
                read_assay_member(source_archive, source_members[filename]), metadata
            )
            for split_directory, split_name in OFFICIAL_SPLIT_DIRECTORIES.items():
                member = supervised_members.get((split_directory, filename))
                if member is None:
                    audit_rows.append(
                        {
                            "assay_id": assay_id,
                            "uniprot_id": str(metadata["UniProt_ID"]),
                            "split": split_name,
                            "eligible": False,
                            "exclusion_reasons": "missing_score_member",
                        }
                    )
                    continue
                scores = _read_supervised_member(
                    supervised_archive, member, model_columns
                )
                aligned, audit = _align_supervised_scores(source, scores, model_columns)
                reasons: list[str] = []
                if audit["duplicate_score_variants"]:
                    reasons.append("duplicate_score_variants")
                if audit["matched_variants"] != audit["source_variants"]:
                    reasons.append("incomplete_variant_join")
                if not np.isclose(
                    audit["dms_score_max_abs_difference"], 0.0, atol=1e-10
                ):
                    reasons.append("dms_score_mismatch")
                if not np.isfinite(aligned["normalized_targets"]).all():
                    reasons.append("nonfinite_normalized_targets")
                for column in model_columns:
                    if audit[f"{column}_finite"] != len(aligned):
                        reasons.append(f"nonfinite_predictions:{column}")
                audit_rows.append(
                    {
                        "assay_id": assay_id,
                        "uniprot_id": str(metadata["UniProt_ID"]),
                        "split": split_name,
                        **audit,
                        "eligible": not reasons,
                        "exclusion_reasons": ";".join(reasons),
                    }
                )
                if reasons:
                    continue
                observed = aligned["normalized_targets"].to_numpy(dtype=float)
                experimental = aligned["DMS_score"].to_numpy(dtype=float)
                for model_name, column in models.items():
                    predicted = aligned[column].to_numpy(dtype=float)
                    metrics = regression_metrics(observed, predicted)
                    selection = top_selection_metrics(experimental, predicted)
                    result_rows.append(
                        {
                            "assay_id": assay_id,
                            "uniprot_id": str(metadata["UniProt_ID"]),
                            "taxon": str(metadata["taxon"]),
                            "coarse_selection_type": str(
                                metadata["coarse_selection_type"]
                            ),
                            "split": split_name,
                            "model": model_name,
                            "evaluation_type": "official_out_of_fold_predictions",
                            "target_space": "ProteinGym_normalized_targets",
                            **metrics.to_dict(),
                            **selection,
                            "test_rows": len(aligned),
                        }
                    )

    audit_frame = pd.DataFrame(audit_rows).sort_values(
        ["assay_id", "split"]
    ).reset_index(drop=True)
    if not result_rows:
        raise ValueError("No official supervised prediction files passed the audit")
    runs = pd.DataFrame(result_rows).sort_values(
        ["model", "split", "assay_id"]
    ).reset_index(drop=True)
    return runs, audit_frame


def summarize_official_supervised(
    runs: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 2026,
) -> pd.DataFrame:
    """Aggregate assays within protein and bootstrap across UniProt identifiers."""
    required = {"assay_id", "uniprot_id", "split", "model", "spearman", "top_recall"}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Official supervised results are missing: {', '.join(missing)}")
    assay_level = runs.groupby(
        ["model", "split", "uniprot_id", "assay_id"], as_index=False
    ).agg(spearman=("spearman", "mean"), top_recall=("top_recall", "mean"))
    protein_level = assay_level.groupby(
        ["model", "split", "uniprot_id"], as_index=False
    ).agg(spearman=("spearman", "mean"), top_recall=("top_recall", "mean"))
    rows: list[dict[str, object]] = []
    for offset, ((model, split), group) in enumerate(
        protein_level.groupby(["model", "split"], sort=True)
    ):
        rng = np.random.default_rng(bootstrap_seed + offset)
        values = group["spearman"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_interval(
            values, repeats=bootstrap_repeats, rng=rng
        )
        rows.append(
            {
                "model": model,
                "split": split,
                "evaluation_type": "official_out_of_fold_predictions",
                "n_assays": int(
                    assay_level.loc[
                        assay_level["model"].eq(model) & assay_level["split"].eq(split),
                        "assay_id",
                    ].nunique()
                ),
                "n_proteins": int(group["uniprot_id"].nunique()),
                "mean_spearman": float(values.mean()),
                "spearman_ci_low": low,
                "spearman_ci_high": high,
                "mean_top_recall": float(group["top_recall"].mean()),
                "bootstrap_unit": "UniProt_ID",
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
