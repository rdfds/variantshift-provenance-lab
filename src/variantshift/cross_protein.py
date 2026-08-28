"""Held-out-protein transfer using assay-independent mutation features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import mondrian_intervals, standard_intervals
from .features import HYDROPATHY, biophysical_matrix
from .metrics import (
    interval_metrics,
    position_conditional_coverage,
    regression_metrics,
    risk_coverage_curve,
    top_selection_metrics,
)
from .mutations import parse_variant
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    read_assay_member,
    read_reference_index,
)
from .zero_shot import DEFAULT_ESM_MODELS, _align_scores, _read_score_member

AMINO_ACIDS = tuple(sorted(HYDROPATHY))
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class CrossProteinDataset:
    metadata: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]


def _stable_assay_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    ordered = frame.sort_values("mutation_codes", kind="stable").reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return ordered.iloc[indices].reset_index(drop=True)


def _cross_protein_matrix(
    frame: pd.DataFrame,
    target_sequence: str,
    score_columns: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    codes = frame["mutation_codes"].astype(str).tolist()
    biochemical = biophysical_matrix(codes).astype(np.float32)
    categorical = np.zeros((len(frame), 2 * len(AMINO_ACIDS) + 2), dtype=np.float32)
    for row, code in enumerate(codes):
        mutation = parse_variant(code)[0]
        categorical[row, AA_INDEX[mutation.reference]] = 1.0
        categorical[row, len(AMINO_ACIDS) + AA_INDEX[mutation.alternate]] = 1.0
        categorical[row, -2] = mutation.position / len(target_sequence)
        categorical[row, -1] = np.log1p(len(target_sequence))
    scores = frame[list(score_columns)].to_numpy(dtype=np.float32)
    names = (
        *[f"reference={residue}" for residue in AMINO_ACIDS],
        *[f"alternate={residue}" for residue in AMINO_ACIDS],
        "normalized_position",
        "log_protein_length",
        *[f"biophysical:{index}" for index in range(biochemical.shape[1])],
        *score_columns,
    )
    return np.concatenate([categorical, biochemical, scores], axis=1), tuple(names)


def build_cross_protein_dataset(
    source_archive_path: Path,
    score_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    *,
    score_columns: Sequence[str] = DEFAULT_ESM_MODELS,
    max_variants_per_assay: int = 1_000,
) -> CrossProteinDataset:
    """Build a label-balanced table for strict held-out-protein prediction."""
    if max_variants_per_assay < 100:
        raise ValueError("At least 100 variants per assay are required")
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = eligibility.loc[
        eligibility["eligible"].astype(bool), "assay_id"
    ].astype(str)
    metadata_frames: list[pd.DataFrame] = []
    feature_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    with ZipFile(source_archive_path) as source_archive, ZipFile(
        score_archive_path
    ) as score_archive:
        source_members = _archive_members(source_archive)
        score_members = _archive_members(score_archive)
        for assay_id in eligible_ids:
            assay_metadata = reference.loc[assay_id]
            filename = str(assay_metadata["DMS_filename"])
            source = canonicalize_assay(
                read_assay_member(source_archive, source_members[filename]), assay_metadata
            )
            scores = _read_score_member(score_archive, score_members[filename], score_columns)
            aligned, audit = _align_scores(source, scores, score_columns)
            if (
                audit["duplicate_score_variants"]
                or audit["matched_dms_scores"] != len(source)
                or audit["common_score_coverage"] < 1.0
            ):
                raise ValueError(f"Zero-shot feature audit failed for {assay_id}")
            selected = _stable_assay_sample(
                aligned.loc[aligned["common_finite_scores"]].reset_index(drop=True),
                max_variants_per_assay,
            )
            matrix, current_names = _cross_protein_matrix(
                selected, str(assay_metadata["target_seq"]), score_columns
            )
            if feature_names is None:
                feature_names = current_names
            elif feature_names != current_names:
                raise RuntimeError("Cross-protein feature schema changed across assays")
            experimental = selected["DMS_score"].to_numpy(dtype=float)
            rank_target = pd.Series(experimental).rank(method="average", pct=True).to_numpy()
            metadata_frames.append(
                pd.DataFrame(
                    {
                        "assay_id": assay_id,
                        "uniprot_id": str(assay_metadata["UniProt_ID"]),
                        "mutation_codes": selected["mutation_codes"].astype(str),
                        "taxon": str(assay_metadata["taxon"]),
                        "coarse_selection_type": str(
                            assay_metadata["coarse_selection_type"]
                        ),
                        "experimental_score": experimental,
                    }
                )
            )
            feature_frames.append(matrix)
            target_frames.append(rank_target)
    if feature_names is None:
        raise ValueError("No eligible assays were available")
    return CrossProteinDataset(
        metadata=pd.concat(metadata_frames, ignore_index=True),
        features=np.concatenate(feature_frames, axis=0),
        targets=np.concatenate(target_frames),
        feature_names=feature_names,
    )


def _model_factories(seed: int):
    return {
        "cross_protein_ridge": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=10.0, solver="lsqr")
        ),
        "cross_protein_histgb": lambda: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def _assay_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("assay_id")["assay_id"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights / np.mean(weights)


def evaluate_held_out_proteins(
    dataset: CrossProteinDataset,
    *,
    folds: int = 5,
    seed: int = 2026,
    coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on some proteins, calibrate on new proteins, and test on disjoint proteins."""
    metadata = dataset.metadata.reset_index(drop=True)
    groups = metadata["uniprot_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds + 2:
        raise ValueError("Held-out-protein evaluation requires more protein groups")
    outer = GroupKFold(n_splits=folds)
    assay_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold, (outer_train, test_indices) in enumerate(
        outer.split(dataset.features, dataset.targets, groups)
    ):
        inner_groups = groups[outer_train]
        inner_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed + fold
        )
        fit_relative, calibration_relative = next(
            inner_splitter.split(outer_train, groups=inner_groups)
        )
        fit_indices = outer_train[fit_relative]
        calibration_indices = outer_train[calibration_relative]
        fit_groups = set(groups[fit_indices])
        calibration_groups = set(groups[calibration_indices])
        test_groups = set(groups[test_indices])
        if (
            fit_groups.intersection(calibration_groups)
            or fit_groups.intersection(test_groups)
            or calibration_groups.intersection(test_groups)
        ):
            raise RuntimeError("Protein-group leakage detected in transfer evaluation")
        fit_weights = _assay_balanced_weights(metadata.iloc[fit_indices])
        for model_name, factory in _model_factories(seed + fold).items():
            model = factory()
            fit_arguments = (
                {"ridge__sample_weight": fit_weights}
                if model_name == "cross_protein_ridge"
                else {"sample_weight": fit_weights}
            )
            model.fit(
                dataset.features[fit_indices],
                dataset.targets[fit_indices],
                **fit_arguments,
            )
            calibration_prediction = np.asarray(
                model.predict(dataset.features[calibration_indices]), dtype=float
            )
            test_prediction = np.asarray(
                model.predict(dataset.features[test_indices]), dtype=float
            )
            standard = standard_intervals(
                dataset.targets[calibration_indices],
                calibration_prediction,
                test_prediction,
                coverage=coverage,
            )
            mondrian = mondrian_intervals(
                metadata.iloc[calibration_indices]["mutation_codes"].astype(str).tolist(),
                dataset.targets[calibration_indices],
                calibration_prediction,
                metadata.iloc[test_indices]["mutation_codes"].astype(str).tolist(),
                test_prediction,
                coverage=coverage,
                min_group_size=100,
            )
            fold_predictions = metadata.iloc[test_indices][
                ["assay_id", "uniprot_id", "mutation_codes", "experimental_score"]
            ].copy()
            fold_predictions["fold"] = fold
            fold_predictions["model"] = model_name
            fold_predictions["rank_target"] = dataset.targets[test_indices]
            fold_predictions["prediction"] = test_prediction
            prediction_rows.append(fold_predictions)
            for interval in (standard, mondrian):
                test_metadata = metadata.iloc[test_indices].reset_index(drop=True)
                test_target = dataset.targets[test_indices]
                for assay_id, local_indices in test_metadata.groupby("assay_id").indices.items():
                    local = np.asarray(local_indices, dtype=int)
                    local_target = test_target[local]
                    local_prediction = test_prediction[local]
                    codes = test_metadata.iloc[local]["mutation_codes"].astype(str).tolist()
                    point = regression_metrics(local_target, local_prediction).to_dict()
                    selection = top_selection_metrics(local_target, local_prediction)
                    intervals = interval_metrics(
                        local_target, interval.lower[local], interval.upper[local]
                    )
                    conditional = position_conditional_coverage(
                        codes,
                        local_target,
                        interval.lower[local],
                        interval.upper[local],
                    )
                    assay_rows.append(
                        {
                            "fold": fold,
                            "assay_id": assay_id,
                            "uniprot_id": str(test_metadata.iloc[local[0]]["uniprot_id"]),
                            "model": model_name,
                            "calibration_method": interval.method,
                            "evaluation_type": "held_out_protein",
                            "nominal_coverage": coverage,
                            **point,
                            **selection,
                            **intervals,
                            **conditional,
                            "test_rows": len(local),
                            "fit_proteins": int(np.unique(groups[fit_indices]).size),
                            "calibration_proteins": int(
                                np.unique(groups[calibration_indices]).size
                            ),
                            "test_proteins": int(np.unique(groups[test_indices]).size),
                        }
                    )
                    for risk in risk_coverage_curve(
                        local_target, local_prediction, interval.uncertainty[local]
                    ):
                        risk_rows.append(
                            {
                                "fold": fold,
                                "assay_id": assay_id,
                                "uniprot_id": str(
                                    test_metadata.iloc[local[0]]["uniprot_id"]
                                ),
                                "model": model_name,
                                "calibration_method": interval.method,
                                **risk,
                            }
                        )
    assays = pd.DataFrame(assay_rows).sort_values(
        ["model", "calibration_method", "fold", "assay_id"]
    ).reset_index(drop=True)
    risks = pd.DataFrame(risk_rows).sort_values(
        ["model", "calibration_method", "retained_fraction", "fold", "assay_id"]
    ).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["model", "fold", "assay_id", "mutation_codes"]
    ).reset_index(drop=True)
    return assays, risks, predictions


def summarize_held_out_proteins(assays: pd.DataFrame) -> pd.DataFrame:
    values = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "observed_coverage",
        "position_coverage_mean",
        "mean_interval_width",
    ]
    protein_level = assays.groupby(
        ["model", "calibration_method", "uniprot_id"], as_index=False
    )[values].mean()
    return (
        protein_level.groupby(["model", "calibration_method"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            **{f"mean_{value}": (value, "mean") for value in values},
        )
        .sort_values(["model", "calibration_method"])
        .reset_index(drop=True)
    )


def summarize_heldout_risk_coverage(risks: pd.DataFrame) -> pd.DataFrame:
    protein_level = risks.groupby(
        ["model", "calibration_method", "retained_fraction", "uniprot_id"],
        as_index=False,
    ).agg(normalized_mae=("normalized_mae", "mean"))
    return (
        protein_level.groupby(
            ["model", "calibration_method", "retained_fraction"], as_index=False
        )
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_normalized_mae=("normalized_mae", "mean"),
        )
        .sort_values(["model", "calibration_method", "retained_fraction"])
        .reset_index(drop=True)
    )
