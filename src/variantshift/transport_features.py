"""Outcome-free task descriptors for the VariantShift transport model.

The development builder intentionally reads model-score columns from the ProteinGym
prediction archive and never reads ``DMS_score`` or ``DMS_score_bin`` from that
archive. Outcomes enter only through the already-audited task-level run table.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .modern_zero_shot import MODERN_ZERO_SHOT_MODELS
from .proteingym import _archive_members
from .schemas import TASK_METRIC_SCHEMA, stable_frame_sha256, write_table

MODEL_FAMILIES = {
    "esm2_650m": "masked-protein-language-model",
    "esmc_600m": "masked-protein-language-model",
    "esm3_open": "multimodal-generative-model",
    "progen3_3b": "autoregressive-protein-language-model",
    "xtrimopglm_100b": "autoregressive-protein-language-model",
    "saprot_650m": "structure-aware-protein-language-model",
    "prosst_2048": "structure-aware-protein-language-model",
    "s3f_msa": "structure-msa-model",
    "venusrem": "retrieval-structure-ensemble",
    "siterm": "structure-msa-model",
    "gemme": "evolutionary-sequence-model",
    "tranception_l": "retrieval-augmented-protein-language-model",
}


def _other_family_sequence_identity(
    alignments: pd.DataFrame, families: dict[str, str]
) -> dict[str, float]:
    if alignments.empty:
        return {}
    working = alignments.copy()
    working["query_family"] = working["query_uniprot_id"].astype(str).map(families)
    working["target_family"] = working["target_uniprot_id"].astype(str).map(families)
    working = working.loc[
        working["query_uniprot_id"].astype(str).ne(working["target_uniprot_id"].astype(str))
        & working["query_family"].notna()
        & working["target_family"].notna()
        & working["query_family"].ne(working["target_family"])
    ]
    return (
        pd.to_numeric(working["sequence_identity"], errors="coerce")
        .groupby(working["query_uniprot_id"].astype(str))
        .max()
        .to_dict()
    )


def _other_family_structure_similarity(
    alignments: pd.DataFrame, families: dict[str, str]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in alignments.itertuples(index=False):
        protein_a = str(row.protein_a)
        protein_b = str(row.protein_b)
        if families.get(protein_a) is None or families.get(protein_b) is None:
            continue
        if families[protein_a] == families[protein_b]:
            continue
        similarity = float(row.reciprocal_minimum_tm_score)
        values[protein_a] = max(values.get(protein_a, 0.0), similarity)
        values[protein_b] = max(values.get(protein_b, 0.0), similarity)
    return values


def _score_shape_features(
    archive: ZipFile,
    member: str,
    model_columns: dict[str, str],
) -> pd.DataFrame:
    usecols = ["mutant", *model_columns.values()]
    if any(column.lower().startswith("dms_") for column in usecols):
        raise RuntimeError("Outcome columns are forbidden in transport feature extraction")
    scores = pd.read_csv(archive.open(member), usecols=usecols)
    numeric = scores.loc[:, list(model_columns.values())].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corr(method="spearman", min_periods=50)
    rows: list[dict[str, object]] = []
    for model_id, column in model_columns.items():
        values = numeric[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        dispersion = float(np.std(finite, ddof=0)) if len(finite) else np.nan
        if len(finite) and dispersion > 0:
            threshold = float(np.quantile(finite, 0.9))
            tail_mean = float(finite[finite >= threshold].mean())
            tail_separation = float((tail_mean - np.median(finite)) / dispersion)
        else:
            tail_separation = np.nan
        peer_correlations = correlations.loc[column].drop(index=column, errors="ignore")
        peer_correlations = peer_correlations[np.isfinite(peer_correlations)]
        median_correlation = (
            float(peer_correlations.median()) if len(peer_correlations) else np.nan
        )
        agreement = (
            (median_correlation + 1.0) / 2.0
            if np.isfinite(median_correlation)
            else np.nan
        )
        rows.append(
            {
                "model_id": model_id,
                "score_dispersion": dispersion,
                "score_tail_separation": tail_separation,
                "missing_fraction": float(1.0 - len(finite) / len(values)),
                "ensemble_agreement": agreement,
                "ensemble_disagreement": (
                    1.0 - agreement if np.isfinite(agreement) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_proteingym_transport_features(
    runs_path: Path,
    eligibility_path: Path,
    reference_path: Path,
    family_assignments_path: Path,
    sequence_alignments_path: Path,
    structure_alignments_path: Path,
    domain_overlaps_path: Path,
    score_archive_path: Path,
    output_path: Path,
    crossover_predictions_path: Path | None = None,
) -> dict[str, object]:
    """Build the real ProteinGym development table used by transport fitting."""
    runs = pd.read_csv(runs_path)
    eligibility = pd.read_csv(eligibility_path)
    reference = pd.read_csv(reference_path)
    families = pd.read_csv(family_assignments_path)
    sequence_alignments = pd.read_csv(sequence_alignments_path)
    structure_alignments = pd.read_csv(structure_alignments_path)
    domains = pd.read_csv(domain_overlaps_path)

    required_runs = {
        "assay_id",
        "uniprot_id",
        "model",
        "modality",
        "selection_gain_sd",
        "taxon",
        "coarse_selection_type",
    }
    missing = sorted(required_runs.difference(runs.columns))
    if missing:
        raise ValueError(f"ProteinGym runs are missing columns: {missing}")
    if runs.duplicated(["assay_id", "model"]).any():
        raise ValueError("ProteinGym runs must be unique by assay and model")

    family_map = families.set_index("uniprot_id")["family_id"].astype(str).to_dict()
    sequence_similarity = _other_family_sequence_identity(sequence_alignments, family_map)
    structure_similarity = _other_family_structure_similarity(
        structure_alignments, family_map
    )
    domain_coverage = (
        domains.loc[domains["qualifies_curated_domain"].astype(bool)]
        .groupby("assay_id")["overlap_fraction_of_shorter"]
        .max()
        .to_dict()
    )

    metadata = reference.loc[
        :, ["DMS_id", "MSA_num_seqs", "MSA_N_eff", "MSA_perc_cov", "pdb_file"]
    ].rename(columns={"DMS_id": "assay_id"})
    base = (
        runs.merge(
            eligibility.loc[
                :, ["assay_id", "protein_length", "mutated_positions", "single_variants"]
            ],
            on="assay_id",
            how="left",
            validate="many_to_one",
        )
        .merge(metadata, on="assay_id", how="left", validate="many_to_one")
        .merge(
            families.loc[:, ["uniprot_id", "family_id", "family_size"]],
            on="uniprot_id",
            how="left",
            validate="many_to_one",
        )
    )
    if base[["protein_length", "mutated_positions", "family_id"]].isna().any().any():
        raise ValueError("Every development task requires length, position, and family metadata")

    present_models = set(base["model"].astype(str))
    configured_columns = {
        model_id: score_column
        for model_id, (score_column, _modality) in MODERN_ZERO_SHOT_MODELS.items()
        if model_id in present_models
    }
    score_rows = []
    reference_index = reference.set_index("DMS_id")
    with ZipFile(score_archive_path) as archive:
        members = _archive_members(archive)
        for assay_id in sorted(base["assay_id"].astype(str).unique()):
            filename = str(reference_index.loc[assay_id, "DMS_filename"])
            features = _score_shape_features(
                archive, members[filename], configured_columns
            )
            features.insert(0, "assay_id", assay_id)
            score_rows.append(features)
    score_features = pd.concat(score_rows, ignore_index=True)
    base = base.merge(
        score_features,
        left_on=["assay_id", "model"],
        right_on=["assay_id", "model_id"],
        how="left",
        validate="one_to_one",
    )

    base["protocol_id"] = "variantshift-development-v1"
    base["panel_id"] = "proteingym-v1.3"
    base["dataset_id"] = "proteingym-substitutions-v1.3"
    base["task_id"] = base["assay_id"].astype(str)
    base["target_id"] = base["uniprot_id"].astype(str)
    base["protein_id"] = base["uniprot_id"].astype(str)
    base["assayed_fraction"] = base["mutated_positions"] / base["protein_length"]
    base["msa_depth"] = pd.to_numeric(base["MSA_num_seqs"], errors="coerce")
    base["msa_neff"] = pd.to_numeric(base["MSA_N_eff"], errors="coerce")
    base["alignment_coverage"] = pd.to_numeric(base["MSA_perc_cov"], errors="coerce")
    base["sequence_identity_to_development"] = (
        base["uniprot_id"].astype(str).map(sequence_similarity).fillna(0.0)
    )
    base["structure_similarity_to_development"] = (
        base["uniprot_id"].astype(str).map(structure_similarity).fillna(0.0)
    )
    base["domain_coverage"] = base["assay_id"].map(domain_coverage).fillna(0.0)
    base["structure_available"] = base["pdb_file"].notna().map(
        {True: "available", False: "missing"}
    )
    base["assay_modality"] = base["coarse_selection_type"].astype(str)
    base["model_family"] = base["model"].astype(str).map(MODEL_FAMILIES).fillna("other")
    base["model_modalities"] = base["modality"].astype(str)
    base["exposure_status"] = "known-development-benchmark"
    if crossover_predictions_path is not None:
        crossover = pd.read_csv(crossover_predictions_path)
        required = {"assay_id", "model", "predicted_probability"}
        missing = sorted(required.difference(crossover.columns))
        if missing:
            raise ValueError(f"Crossover predictions are missing columns: {missing}")
        crossover = crossover.loc[crossover["model"].astype(str).eq("histgb")]
        confidence = crossover.groupby("assay_id")["predicted_probability"].mean()
        # This predictor uses within-assay training outcomes. Retain it only as an explicitly
        # quarantined development diagnostic; it is forbidden in the deployable transport model.
        base["development_crossover_probability_supervised_wins"] = base[
            "assay_id"
        ].map(confidence)
    else:
        base["development_crossover_probability_supervised_wins"] = np.nan

    columns = [
        "protocol_id",
        "panel_id",
        "dataset_id",
        "assay_id",
        "task_id",
        "target_id",
        "protein_id",
        "family_id",
        "model_id",
        "selection_gain_sd",
        "protein_length",
        "assayed_fraction",
        "mutated_positions",
        "msa_depth",
        "msa_neff",
        "alignment_coverage",
        "sequence_identity_to_development",
        "structure_similarity_to_development",
        "domain_coverage",
        "score_dispersion",
        "score_tail_separation",
        "missing_fraction",
        "ensemble_disagreement",
        "ensemble_agreement",
        "taxon",
        "assay_modality",
        "model_family",
        "model_modalities",
        "exposure_status",
        "structure_available",
        "development_crossover_probability_supervised_wins",
    ]
    output = (
        base.loc[:, columns]
        .sort_values(["assay_id", "model_id"])
        .reset_index(drop=True)
    )
    TASK_METRIC_SCHEMA.validate(output)
    write_table(output, output_path)
    return {
        "output": str(output_path),
        "rows": len(output),
        "assays": int(output["assay_id"].nunique()),
        "proteins": int(output["protein_id"].nunique()),
        "families": int(output["family_id"].nunique()),
        "models": int(output["model_id"].nunique()),
        "sha256": stable_frame_sha256(output),
        "outcome_columns_read_from_prediction_archive": [],
    }
