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
    score_feature_count: int = 0


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
    eligible_ids = eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    metadata_frames: list[pd.DataFrame] = []
    feature_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    with (
        ZipFile(source_archive_path) as source_archive,
        ZipFile(score_archive_path) as score_archive,
    ):
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
                        "coarse_selection_type": str(assay_metadata["coarse_selection_type"]),
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
        score_feature_count=len(score_columns),
    )


def _model_factories(seed: int, *, include_feature_ablation: bool):
    factories = {
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
    if include_feature_ablation:
        factories.update(
            {
                "cross_protein_ridge_mutation_only": lambda: make_pipeline(
                    StandardScaler(), Ridge(alpha=10.0, solver="lsqr")
                ),
                "cross_protein_histgb_mutation_only": lambda: HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            }
        )
    return factories


def _assay_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("assay_id")["assay_id"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights / np.mean(weights)


def _evaluate_group_holdout(
    dataset: CrossProteinDataset,
    metadata: pd.DataFrame,
    groups: np.ndarray,
    *,
    group_name: str,
    evaluation_type: str,
    folds: int = 5,
    repeats: int = 1,
    seed: int = 2026,
    coverage: float = 0.8,
    include_feature_ablation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit, calibrate, and test on three disjoint sets of grouped proteins."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds + 2:
        raise ValueError(f"Held-out-{group_name} evaluation requires more groups")
    if repeats < 1:
        raise ValueError("Grouped holdout repeats must be positive")
    assay_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for repeat in range(repeats):
        outer = GroupKFold(
            n_splits=folds,
            shuffle=repeats > 1,
            random_state=seed + repeat if repeats > 1 else None,
        )
        for fold, (outer_train, test_indices) in enumerate(
            outer.split(dataset.features, dataset.targets, groups)
        ):
            _evaluate_one_group_fold(
                dataset,
                metadata,
                groups,
                outer_train,
                test_indices,
                group_name=group_name,
                evaluation_type=evaluation_type,
                repeat=repeat,
                fold=fold,
                seed=seed + repeat * folds + fold,
                coverage=coverage,
                include_feature_ablation=include_feature_ablation,
                assay_rows=assay_rows,
                risk_rows=risk_rows,
                prediction_rows=prediction_rows,
            )
    assays = (
        pd.DataFrame(assay_rows)
        .sort_values(["model", "calibration_method", "repeat", "fold", "assay_id"])
        .reset_index(drop=True)
    )
    risks = (
        pd.DataFrame(risk_rows)
        .sort_values(
            [
                "model",
                "calibration_method",
                "retained_fraction",
                "repeat",
                "fold",
                "assay_id",
            ]
        )
        .reset_index(drop=True)
    )
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        .sort_values(["model", "repeat", "fold", "assay_id", "mutation_codes"])
        .reset_index(drop=True)
    )
    return assays, risks, predictions


def _evaluate_one_group_fold(
    dataset: CrossProteinDataset,
    metadata: pd.DataFrame,
    groups: np.ndarray,
    outer_train: np.ndarray,
    test_indices: np.ndarray,
    *,
    group_name: str,
    evaluation_type: str,
    repeat: int,
    fold: int,
    seed: int,
    coverage: float,
    include_feature_ablation: bool,
    assay_rows: list[dict[str, object]],
    risk_rows: list[dict[str, object]],
    prediction_rows: list[pd.DataFrame],
) -> None:
    """Evaluate one fit/calibration/test partition of complete groups."""
    inner_groups = groups[outer_train]
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
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
        raise RuntimeError(f"{group_name.title()} leakage detected in transfer evaluation")
    fit_weights = _assay_balanced_weights(metadata.iloc[fit_indices])
    if include_feature_ablation and dataset.score_feature_count < 1:
        raise ValueError("Feature ablation requires declared zero-shot score features")
    for model_name, factory in _model_factories(
        seed, include_feature_ablation=include_feature_ablation
    ).items():
        if model_name.endswith("_mutation_only"):
            features = dataset.features[:, : -dataset.score_feature_count]
            feature_set = "mutation_descriptors_only"
        else:
            features = dataset.features
            feature_set = "mutation_descriptors_plus_fixed_esm_scores"
        model = factory()
        fit_arguments = (
            {"ridge__sample_weight": fit_weights}
            if model_name.startswith("cross_protein_ridge")
            else {"sample_weight": fit_weights}
        )
        model.fit(
            features[fit_indices],
            dataset.targets[fit_indices],
            **fit_arguments,
        )
        calibration_prediction = np.asarray(
            model.predict(features[calibration_indices]), dtype=float
        )
        test_prediction = np.asarray(model.predict(features[test_indices]), dtype=float)
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
        prediction_columns = [
            "assay_id",
            "uniprot_id",
            "mutation_codes",
            "experimental_score",
        ]
        if group_name != "protein":
            prediction_columns.append("family_id")
        fold_predictions = metadata.iloc[test_indices][prediction_columns].copy()
        fold_predictions["fold"] = fold
        fold_predictions["repeat"] = repeat
        fold_predictions["model"] = model_name
        if include_feature_ablation:
            fold_predictions["feature_set"] = feature_set
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
                assay_result = {
                    "fold": fold,
                    "repeat": repeat,
                    "assay_id": assay_id,
                    "uniprot_id": str(test_metadata.iloc[local[0]]["uniprot_id"]),
                    "model": model_name,
                    "calibration_method": interval.method,
                    "evaluation_type": evaluation_type,
                    "nominal_coverage": coverage,
                    **point,
                    **selection,
                    **intervals,
                    **conditional,
                    "test_rows": len(local),
                    "fit_proteins": int(metadata.iloc[fit_indices]["uniprot_id"].nunique()),
                    "calibration_proteins": int(
                        metadata.iloc[calibration_indices]["uniprot_id"].nunique()
                    ),
                    "test_proteins": int(metadata.iloc[test_indices]["uniprot_id"].nunique()),
                }
                if include_feature_ablation:
                    assay_result["feature_set"] = feature_set
                if group_name != "protein":
                    assay_result.update(
                        {
                            "family_id": str(test_metadata.iloc[local[0]]["family_id"]),
                            "fit_families": len(fit_groups),
                            "calibration_families": len(calibration_groups),
                            "test_families": len(test_groups),
                        }
                    )
                assay_rows.append(assay_result)
                for risk in risk_coverage_curve(
                    local_target, local_prediction, interval.uncertainty[local]
                ):
                    risk_result = {
                        "fold": fold,
                        "repeat": repeat,
                        "assay_id": assay_id,
                        "uniprot_id": str(test_metadata.iloc[local[0]]["uniprot_id"]),
                        "model": model_name,
                        "calibration_method": interval.method,
                        **risk,
                    }
                    if include_feature_ablation:
                        risk_result["feature_set"] = feature_set
                    if group_name != "protein":
                        risk_result["family_id"] = str(test_metadata.iloc[local[0]]["family_id"])
                    risk_rows.append(risk_result)


def evaluate_held_out_proteins(
    dataset: CrossProteinDataset,
    *,
    folds: int = 5,
    repeats: int = 1,
    seed: int = 2026,
    coverage: float = 0.8,
    include_feature_ablation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on some proteins, calibrate on new proteins, and test on disjoint proteins."""
    metadata = dataset.metadata.reset_index(drop=True)
    groups = metadata["uniprot_id"].astype(str).to_numpy()
    return _evaluate_group_holdout(
        dataset,
        metadata,
        groups,
        group_name="protein",
        evaluation_type="held_out_protein",
        folds=folds,
        repeats=repeats,
        seed=seed,
        coverage=coverage,
        include_feature_ablation=include_feature_ablation,
    )


def evaluate_held_out_families(
    dataset: CrossProteinDataset,
    family_assignments: pd.DataFrame,
    *,
    folds: int = 5,
    repeats: int = 1,
    seed: int = 2026,
    coverage: float = 0.8,
    group_name: str = "sequence_family",
    evaluation_type: str = "held_out_sequence_family",
    include_feature_ablation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate with complete precomputed family clusters absent from training."""
    required_columns = ["uniprot_id", "family_id"]
    missing_columns = set(required_columns).difference(family_assignments.columns)
    if missing_columns:
        raise ValueError(f"Family assignments are missing: {', '.join(sorted(missing_columns))}")
    mapping = family_assignments[required_columns].drop_duplicates()
    if mapping["uniprot_id"].duplicated().any():
        raise ValueError("Each UniProt ID must map to exactly one family")
    metadata = dataset.metadata.reset_index(drop=True).merge(
        mapping, on="uniprot_id", how="left", validate="many_to_one"
    )
    if metadata["family_id"].isna().any():
        missing = sorted(metadata.loc[metadata["family_id"].isna(), "uniprot_id"].unique())
        raise ValueError(f"Missing family assignments for: {', '.join(missing)}")
    groups = metadata["family_id"].astype(str).to_numpy()
    return _evaluate_group_holdout(
        dataset,
        metadata,
        groups,
        group_name=group_name,
        evaluation_type=evaluation_type,
        folds=folds,
        repeats=repeats,
        seed=seed,
        coverage=coverage,
        include_feature_ablation=include_feature_ablation,
    )


def summarize_held_out_proteins(assays: pd.DataFrame) -> pd.DataFrame:
    values = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "observed_coverage",
        "position_coverage_mean",
        "mean_interval_width",
    ]
    protein_level = assays.groupby(["model", "calibration_method", "uniprot_id"], as_index=False)[
        values
    ].mean()
    return (
        protein_level.groupby(["model", "calibration_method"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            **{f"mean_{value}": (value, "mean") for value in values},
        )
        .sort_values(["model", "calibration_method"])
        .reset_index(drop=True)
    )


def summarize_grouped_repeat_estimates(assays: pd.DataFrame) -> pd.DataFrame:
    """Expose protein-balanced estimates for every randomized grouped split repeat."""
    if "repeat" not in assays:
        raise ValueError("Grouped holdout assays do not contain repeat identifiers")
    values = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "observed_coverage",
        "position_coverage_mean",
        "mean_interval_width",
    ]
    protein_level = assays.groupby(
        ["model", "calibration_method", "repeat", "uniprot_id"], as_index=False
    )[values].mean()
    return (
        protein_level.groupby(["model", "calibration_method", "repeat"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            **{f"mean_{value}": (value, "mean") for value in values},
        )
        .sort_values(["model", "calibration_method", "repeat"])
        .reset_index(drop=True)
    )


def summarize_heldout_risk_coverage(risks: pd.DataFrame) -> pd.DataFrame:
    protein_level = risks.groupby(
        ["model", "calibration_method", "retained_fraction", "uniprot_id"],
        as_index=False,
    ).agg(normalized_mae=("normalized_mae", "mean"))
    return (
        protein_level.groupby(["model", "calibration_method", "retained_fraction"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_normalized_mae=("normalized_mae", "mean"),
        )
        .sort_values(["model", "calibration_method", "retained_fraction"])
        .reset_index(drop=True)
    )


def compare_holdout_protocols(
    protein_assays: pd.DataFrame,
    family_assays: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 2026,
    baseline_label: str = "heldout_protein",
    alternative_label: str = "heldout_family",
) -> pd.DataFrame:
    """Paired grouped bootstrap comparison of two holdout protocols."""
    if bootstrap_repeats < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    keys = ["model", "calibration_method", "assay_id", "uniprot_id"]
    if "repeat" in protein_assays.columns and "repeat" in family_assays.columns:
        keys.append("repeat")
    metrics = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "observed_coverage",
        "position_coverage_mean",
    ]
    required = set(keys + metrics)
    for label, frame in (
        (baseline_label, protein_assays),
        (alternative_label, family_assays),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label.title()} holdout results are missing: {', '.join(sorted(missing))}"
            )
        if frame.duplicated(keys).any():
            raise ValueError(f"{label.title()} holdout results contain duplicate assays")
    baseline_suffix = f"_{baseline_label}"
    alternative_suffix = f"_{alternative_label}"
    alternative_columns = keys + metrics
    bootstrap_group = None
    if "family_id" in family_assays.columns:
        alternative_columns.append("family_id")
        bootstrap_group = "family_id"
    paired = protein_assays[keys + metrics].merge(
        family_assays[alternative_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=(baseline_suffix, alternative_suffix),
    )
    if len(paired) != len(protein_assays) or len(paired) != len(family_assays):
        raise ValueError("Protein and family holdout results are not exactly paired")
    rng = np.random.default_rng(seed)
    rows = []
    for (model, calibration), group in paired.groupby(["model", "calibration_method"], sort=True):
        aggregate = {
            **{f"{metric}{baseline_suffix}": "mean" for metric in metrics},
            **{f"{metric}{alternative_suffix}": "mean" for metric in metrics},
        }
        if bootstrap_group is not None:
            aggregate[bootstrap_group] = "first"
            if group.groupby("uniprot_id")[bootstrap_group].nunique().max() != 1:
                raise ValueError("Each protein must have one alternative holdout family")
        protein_level = group.groupby("uniprot_id", as_index=False).agg(aggregate)
        n_proteins = len(protein_level)
        if bootstrap_group is None:
            group_codes = np.arange(n_proteins)
            bootstrap_unit = "UniProt_ID"
        else:
            group_codes, _ = pd.factorize(protein_level[bootstrap_group], sort=True)
            bootstrap_unit = f"{alternative_label}:{bootstrap_group}"
        n_groups = int(np.max(group_codes)) + 1
        sampled_groups = rng.integers(0, n_groups, size=(bootstrap_repeats, n_groups))
        group_counts = np.bincount(group_codes).astype(float)
        for metric in metrics:
            protein_values = protein_level[f"{metric}{baseline_suffix}"].to_numpy(dtype=float)
            family_values = protein_level[f"{metric}{alternative_suffix}"].to_numpy(dtype=float)
            protein_sums = np.bincount(group_codes, weights=protein_values)
            family_sums = np.bincount(group_codes, weights=family_values)
            bootstrap_counts = group_counts[sampled_groups].sum(axis=1)
            protein_bootstrap = protein_sums[sampled_groups].sum(axis=1) / bootstrap_counts
            family_bootstrap = family_sums[sampled_groups].sum(axis=1) / bootstrap_counts
            delta_bootstrap = family_bootstrap - protein_bootstrap
            rows.append(
                {
                    "model": model,
                    "calibration_method": calibration,
                    "metric": metric,
                    "n_proteins": n_proteins,
                    "n_bootstrap_groups": n_groups,
                    f"{baseline_label}_mean": float(protein_values.mean()),
                    f"{baseline_label}_ci_low": float(np.quantile(protein_bootstrap, 0.025)),
                    f"{baseline_label}_ci_high": float(np.quantile(protein_bootstrap, 0.975)),
                    f"{alternative_label}_mean": float(family_values.mean()),
                    f"{alternative_label}_ci_low": float(np.quantile(family_bootstrap, 0.025)),
                    f"{alternative_label}_ci_high": float(np.quantile(family_bootstrap, 0.975)),
                    f"{alternative_label}_minus_{baseline_label}": float(
                        family_values.mean() - protein_values.mean()
                    ),
                    "delta_ci_low": float(np.quantile(delta_bootstrap, 0.025)),
                    "delta_ci_high": float(np.quantile(delta_bootstrap, 0.975)),
                    "bootstrap_unit": bootstrap_unit,
                    "bootstrap_repeats": bootstrap_repeats,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "calibration_method", "metric"])
        .reset_index(drop=True)
    )


def compare_feature_ablation(
    assays: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compare full and mutation-only transfer with a paired family bootstrap."""
    if bootstrap_repeats < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    required = {
        "model",
        "calibration_method",
        "assay_id",
        "uniprot_id",
        "family_id",
        "repeat",
        "spearman",
        "top_recall",
        "selection_gain_sd",
    }
    if missing := required.difference(assays.columns):
        raise ValueError(f"Feature-ablation results are missing: {', '.join(sorted(missing))}")
    keys = [
        "calibration_method",
        "assay_id",
        "uniprot_id",
        "family_id",
        "repeat",
    ]
    metrics = ["spearman", "top_recall", "selection_gain_sd"]
    pairs = (
        ("cross_protein_ridge", "cross_protein_ridge_mutation_only"),
        ("cross_protein_histgb", "cross_protein_histgb_mutation_only"),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for full_model, mutation_model in pairs:
        full = assays.loc[assays["model"].eq(full_model), keys + metrics]
        mutation = assays.loc[assays["model"].eq(mutation_model), keys + metrics]
        paired = full.merge(
            mutation,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_full", "_mutation_only"),
        )
        if len(paired) != len(full) or len(paired) != len(mutation):
            raise ValueError(f"Feature sets are not exactly paired for {full_model}")
        for calibration, group in paired.groupby("calibration_method", sort=True):
            aggregate = {
                "family_id": "first",
                **{f"{metric}_full": "mean" for metric in metrics},
                **{f"{metric}_mutation_only": "mean" for metric in metrics},
            }
            protein = group.groupby("uniprot_id", as_index=False).agg(aggregate)
            if group.groupby("uniprot_id")["family_id"].nunique().max() != 1:
                raise ValueError("Each protein must have one curated family")
            family_codes, families = pd.factorize(protein["family_id"], sort=True)
            family_count = len(families)
            sampled = rng.integers(0, family_count, size=(bootstrap_repeats, family_count))
            counts = np.bincount(family_codes).astype(float)
            sampled_counts = counts[sampled].sum(axis=1)
            for metric in metrics:
                full_values = protein[f"{metric}_full"].to_numpy(dtype=float)
                mutation_values = protein[f"{metric}_mutation_only"].to_numpy(dtype=float)
                full_sums = np.bincount(family_codes, weights=full_values)
                mutation_sums = np.bincount(family_codes, weights=mutation_values)
                full_bootstrap = full_sums[sampled].sum(axis=1) / sampled_counts
                mutation_bootstrap = mutation_sums[sampled].sum(axis=1) / sampled_counts
                delta_bootstrap = full_bootstrap - mutation_bootstrap
                rows.append(
                    {
                        "model": full_model,
                        "calibration_method": calibration,
                        "metric": metric,
                        "n_proteins": len(protein),
                        "n_families": family_count,
                        "full_feature_mean": float(full_values.mean()),
                        "mutation_only_mean": float(mutation_values.mean()),
                        "full_minus_mutation_only": float(
                            full_values.mean() - mutation_values.mean()
                        ),
                        "delta_ci_low": float(np.quantile(delta_bootstrap, 0.025)),
                        "delta_ci_high": float(np.quantile(delta_bootstrap, 0.975)),
                        "bootstrap_unit": "curated_family_id",
                        "bootstrap_repeats": bootstrap_repeats,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["model", "calibration_method", "metric"])
        .reset_index(drop=True)
    )
