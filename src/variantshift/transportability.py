"""VariantShift Transportability Score and task-level selective evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import TASK_METRIC_SCHEMA, stable_frame_sha256, write_table

OUTCOME_DERIVED_MARKERS = (
    "selection_gain",
    "spearman",
    "top_recall",
    "ndcg",
    "regret",
    "fitness",
    "effect",
    "outcome",
    "dms_score",
    "failure",
)


@dataclass(frozen=True)
class TransportConfig:
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    group_column: str = "family_id"
    task_column: str = "task_id"
    target_column: str = "selection_gain_sd"
    model_column: str = "model_id"
    protein_column: str = "protein_id"
    assay_column: str = "assay_id"
    coverage: float = 0.90
    outer_folds: int = 5
    calibration_fraction: float = 0.20
    bootstrap_repeats: int = 10_000
    seed: int = 20260830

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransportConfig:
        return cls(
            numeric_features=tuple(map(str, payload["numeric_features"])),
            categorical_features=tuple(map(str, payload.get("categorical_features", []))),
            group_column=str(payload.get("group_column", "family_id")),
            task_column=str(payload.get("task_column", "task_id")),
            target_column=str(payload.get("target_column", "selection_gain_sd")),
            model_column=str(payload.get("model_column", "model_id")),
            protein_column=str(payload.get("protein_column", "protein_id")),
            assay_column=str(payload.get("assay_column", "assay_id")),
            coverage=float(payload.get("coverage", 0.90)),
            outer_folds=int(payload.get("outer_folds", 5)),
            calibration_fraction=float(payload.get("calibration_fraction", 0.20)),
            bootstrap_repeats=int(payload.get("bootstrap_repeats", 10_000)),
            seed=int(payload.get("seed", 20260830)),
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def validate(self) -> None:
        if not self.feature_columns:
            raise ValueError("At least one transport feature is required")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("Transport feature names must be unique")
        leaked = sorted(
            feature
            for feature in self.feature_columns
            if any(marker in feature.lower() for marker in OUTCOME_DERIVED_MARKERS)
        )
        if leaked:
            raise ValueError(f"Outcome-derived columns cannot be transport features: {leaked}")
        if not 0 < self.coverage < 1:
            raise ValueError("Conformal coverage must lie in (0, 1)")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("Calibration fraction must lie in (0, 0.5)")
        if self.outer_folds < 3:
            raise ValueError("At least three outer group folds are required")
        if self.bootstrap_repeats < 1:
            raise ValueError("Bootstrap repeats must be positive")


def load_transport_config(path: Path) -> TransportConfig:
    config = TransportConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    config.validate()
    return config


def _preprocessor(config: TransportConfig, *, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), list(config.numeric_features))
    ]
    if config.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(config.categorical_features),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


HISTGB_CANDIDATES: tuple[dict[str, float | int], ...] = (
    {
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 500,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 2.0,
    },
)


def _pipeline(
    config: TransportConfig,
    *,
    estimator: str,
    parameters: dict[str, float | int] | None = None,
    seed_offset: int = 0,
) -> Pipeline:
    if estimator == "histgb":
        model = HistGradientBoostingRegressor(
            **(parameters or HISTGB_CANDIDATES[0]),
            random_state=config.seed + seed_offset,
        )
        return Pipeline(
            [("preprocess", _preprocessor(config, scale_numeric=False)), ("model", model)]
        )
    if estimator == "elastic_net":
        model = ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10_000, random_state=config.seed)
        return Pipeline(
            [("preprocess", _preprocessor(config, scale_numeric=True)), ("model", model)]
        )
    raise ValueError(f"Unsupported transport estimator: {estimator}")


def _group_weighted_mae(observed: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame(
        {"group": groups.astype(str), "error": np.abs(observed - predicted)}
    )
    return float(frame.groupby("group", sort=False)["error"].mean().mean())


def select_histgb_parameters(
    frame: pd.DataFrame,
    config: TransportConfig,
) -> dict[str, float | int]:
    """Choose a small preregistered grid using only inner held-out groups."""
    groups = frame[config.group_column].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return dict(HISTGB_CANDIDATES[0])
    splits = min(4, len(unique_groups))
    splitter = GroupKFold(n_splits=splits)
    target = frame[config.target_column].to_numpy(dtype=float)
    scores: list[tuple[float, int]] = []
    for index, parameters in enumerate(HISTGB_CANDIDATES):
        fold_scores = []
        for train_indices, test_indices in splitter.split(frame, target, groups):
            model = _pipeline(config, estimator="histgb", parameters=dict(parameters))
            model.fit(frame.iloc[train_indices], target[train_indices])
            predicted = model.predict(frame.iloc[test_indices])
            fold_scores.append(
                _group_weighted_mae(
                    target[test_indices], predicted, groups[test_indices]
                )
            )
        scores.append((float(np.mean(fold_scores)), index))
    _, best = min(scores)
    return dict(HISTGB_CANDIDATES[best])


def group_conformal_quantile(
    residuals: np.ndarray,
    groups: np.ndarray,
    *,
    coverage: float,
) -> float:
    """Conservative one-sided quantile over worst residuals within calibration groups."""
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie in (0, 1)")
    values = pd.DataFrame(
        {"residual": np.asarray(residuals, dtype=float), "group": groups.astype(str)}
    ).groupby("group", sort=False)["residual"].max()
    if values.empty:
        raise ValueError("At least one calibration group is required")
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(np.sort(values.to_numpy(dtype=float))[rank - 1])


def _fit_calibration_indices(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame[config.group_column].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Training data must contain at least two family groups")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.calibration_fraction,
        random_state=config.seed + seed_offset,
    )
    fit_indices, calibration_indices = next(splitter.split(frame, groups=groups))
    return fit_indices, calibration_indices


def cross_fitted_transport_predictions(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    estimator: str = "histgb",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each row from a model trained and calibrated on disjoint families."""
    config.validate()
    required = {
        *config.feature_columns,
        config.group_column,
        config.task_column,
        config.target_column,
        config.model_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Transport table is missing columns: {missing}")
    target = frame[config.target_column].to_numpy(dtype=float)
    groups = frame[config.group_column].astype(str).to_numpy()
    group_count = len(np.unique(groups))
    if group_count < config.outer_folds:
        raise ValueError(
            f"Need at least {config.outer_folds} distinct groups; found {group_count}"
        )
    splitter = GroupKFold(n_splits=config.outer_folds)
    predicted = np.full(len(frame), np.nan, dtype=float)
    lower = np.full(len(frame), np.nan, dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)
    parameter_rows: list[dict[str, object]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(frame, target, groups)):
        train = frame.iloc[train_indices]
        fit_local, calibration_local = _fit_calibration_indices(
            train, config, seed_offset=fold
        )
        fit_indices = train_indices[fit_local]
        calibration_indices = train_indices[calibration_local]
        parameters = (
            select_histgb_parameters(frame.iloc[fit_indices], config)
            if estimator == "histgb"
            else {}
        )
        model = _pipeline(
            config,
            estimator=estimator,
            parameters=parameters,
            seed_offset=fold,
        )
        model.fit(frame.iloc[fit_indices], target[fit_indices])
        calibration_predictions = model.predict(frame.iloc[calibration_indices])
        quantile = group_conformal_quantile(
            calibration_predictions - target[calibration_indices],
            groups[calibration_indices],
            coverage=config.coverage,
        )
        predicted[test_indices] = model.predict(frame.iloc[test_indices])
        lower[test_indices] = predicted[test_indices] - quantile
        fold_ids[test_indices] = fold
        parameter_rows.append(
            {
                "fold": fold,
                "estimator": estimator,
                "fit_rows": len(fit_indices),
                "calibration_rows": len(calibration_indices),
                "test_rows": len(test_indices),
                "fit_groups": len(np.unique(groups[fit_indices])),
                "calibration_groups": len(np.unique(groups[calibration_indices])),
                "test_groups": len(np.unique(groups[test_indices])),
                "conformal_quantile": quantile,
                "parameters": json.dumps(parameters, sort_keys=True),
            }
        )
    if not np.isfinite(predicted).all() or (fold_ids < 0).any():
        raise RuntimeError("Cross-fitting did not generate every held-out prediction")
    output = frame.copy()
    output["transport_estimator"] = estimator
    output["fold"] = fold_ids
    output["predicted_selection_gain_sd"] = predicted
    output["lower_selection_gain_sd"] = lower
    output["trusted"] = lower > 0
    output["prediction_error"] = predicted - target
    output["lower_bound_covers"] = target >= lower
    return output, pd.DataFrame(parameter_rows)


def fit_frozen_transport_model(
    frame: pd.DataFrame,
    config: TransportConfig,
) -> dict[str, object]:
    """Fit the deployable model and keep a disjoint family calibration set."""
    fit_indices, calibration_indices = _fit_calibration_indices(frame, config, seed_offset=991)
    parameters = select_histgb_parameters(frame.iloc[fit_indices], config)
    target = frame[config.target_column].to_numpy(dtype=float)
    groups = frame[config.group_column].astype(str).to_numpy()
    model = _pipeline(config, estimator="histgb", parameters=parameters, seed_offset=991)
    model.fit(frame.iloc[fit_indices], target[fit_indices])
    calibration_predictions = model.predict(frame.iloc[calibration_indices])
    quantile = group_conformal_quantile(
        calibration_predictions - target[calibration_indices],
        groups[calibration_indices],
        coverage=config.coverage,
    )
    best_average_model = (
        frame.groupby(config.model_column)[config.target_column].mean().idxmax()
    )
    return {
        "schema_version": 1,
        "pipeline": model,
        "config": asdict(config),
        "parameters": parameters,
        "conformal_quantile": quantile,
        "fit_rows": fit_indices.tolist(),
        "calibration_rows": calibration_indices.tolist(),
        "best_average_model": str(best_average_model),
        "training_frame_sha256": stable_frame_sha256(frame),
    }


def predict_with_frozen_transport_model(
    bundle: dict[str, object],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    config = TransportConfig.from_dict(dict(bundle["config"]))
    config.validate()
    missing = sorted(set(config.feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Confirmation features are missing columns: {missing}")
    forbidden = [
        column
        for column in frame.columns
        if any(marker in column.lower() for marker in OUTCOME_DERIVED_MARKERS)
    ]
    if forbidden:
        raise ValueError(
            f"Frozen confirmation features contain outcome-derived columns: {sorted(forbidden)}"
        )
    model = bundle["pipeline"]
    predicted = model.predict(frame)
    output = frame.copy()
    output["predicted_selection_gain_sd"] = predicted
    output["lower_selection_gain_sd"] = predicted - float(bundle["conformal_quantile"])
    output["trusted"] = output["lower_selection_gain_sd"] > 0
    return output


def _choose_task_rows(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    policy: str,
    seed: int,
    best_average_model: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    working = frame.copy()
    if policy == "variantshift":
        index = working.groupby(config.task_column)["lower_selection_gain_sd"].idxmax()
        selected = working.loc[index].copy()
        selected["confidence"] = selected["lower_selection_gain_sd"]
    elif policy == "uncalibrated":
        index = working.groupby(config.task_column)["predicted_selection_gain_sd"].idxmax()
        selected = working.loc[index].copy()
        selected["confidence"] = selected["predicted_selection_gain_sd"]
    elif policy == "elastic_net":
        index = working.groupby(config.task_column)[
            "elastic_predicted_selection_gain_sd"
        ].idxmax()
        selected = working.loc[index].copy()
        selected["confidence"] = selected["elastic_predicted_selection_gain_sd"]
    elif policy == "oracle":
        index = working.groupby(config.task_column)[config.target_column].idxmax()
        selected = working.loc[index].copy()
        selected["confidence"] = selected[config.target_column]
    elif policy == "random":
        working["_random_choice"] = rng.random(len(working))
        index = working.groupby(config.task_column)["_random_choice"].idxmax()
        selected = working.loc[index].copy()
        selected["confidence"] = rng.random(len(selected))
    elif policy in {
        "always_best",
        "msa_depth",
        "score_dispersion",
        "ensemble_agreement",
        "crossover_classifier",
    }:
        if not best_average_model:
            raise ValueError(f"{policy} policy requires the development best model")
        candidates = working.loc[
            working[config.model_column].astype(str).eq(best_average_model)
        ]
        if candidates.empty:
            raise ValueError(f"Best development model {best_average_model} is unavailable")
        selected = candidates.copy()
        confidence_column = {
            "always_best": None,
            "msa_depth": "msa_neff",
            "score_dispersion": "score_dispersion",
            "ensemble_agreement": "ensemble_agreement",
            "crossover_classifier": "crossover_probability_supervised_wins",
        }[policy]
        confidence = (
            np.zeros(len(selected))
            if confidence_column is None
            else selected[confidence_column].to_numpy(dtype=float)
        )
        selected["confidence"] = (
            1.0 - confidence if policy == "crossover_classifier" else confidence
        )
    else:
        raise ValueError(f"Unsupported selective policy: {policy}")
    return selected.sort_values(config.task_column).reset_index(drop=True)


def selective_policy_curve(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    policy: str,
    best_average_model: str | None = None,
    coverages: tuple[float, ...] = tuple(np.linspace(0.1, 1.0, 10)),
) -> pd.DataFrame:
    selected = _choose_task_rows(
        frame,
        config,
        policy=policy,
        seed=config.seed,
        best_average_model=best_average_model,
    ).sort_values("confidence", ascending=False, kind="stable")
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(selected) * coverage)))
        retained = selected.iloc[:count]
        gains = retained[config.target_column].to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "coverage": float(coverage),
                "retained_tasks": count,
                "failure_rate": float(np.mean(gains <= 0)),
                "mean_selection_gain_sd": float(np.mean(gains)),
                "median_selection_gain_sd": float(np.median(gains)),
                "confidence_threshold": float(retained["confidence"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in curves.groupby("policy", sort=True):
        group = group.sort_values("coverage")
        at_half = group.iloc[(group["coverage"] - 0.5).abs().argsort()[:1]].iloc[0]
        rows.append(
            {
                "policy": policy,
                "risk_coverage_auc": float(
                    np.trapz(group["failure_rate"], group["coverage"])
                ),
                "failure_rate_at_50pct": float(at_half["failure_rate"]),
                "mean_selection_gain_at_50pct": float(at_half["mean_selection_gain_sd"]),
                "tasks": int(group["retained_tasks"].max()),
            }
        )
    return pd.DataFrame(rows)


def _risk_auc(selected: pd.DataFrame, target_column: str) -> float:
    ordered = selected.sort_values("confidence", ascending=False, kind="stable")
    risks = []
    coverages = np.linspace(0.1, 1.0, 10)
    for coverage in coverages:
        count = max(1, int(np.ceil(len(ordered) * coverage)))
        risks.append(float(np.mean(ordered.iloc[:count][target_column] <= 0)))
    return float(np.trapz(risks, coverages))


def _task_hierarchy(
    tasks: pd.DataFrame, config: TransportConfig
) -> tuple[tuple[tuple[np.ndarray, ...], ...], ...]:
    hierarchy = []
    for _family, family_frame in tasks.groupby(config.group_column, sort=True):
        proteins = []
        for _protein, protein_frame in family_frame.groupby(
            config.protein_column, sort=True
        ):
            assays = tuple(
                assay_frame[config.task_column].astype(str).to_numpy()
                for _assay, assay_frame in protein_frame.groupby(
                    config.assay_column, sort=True
                )
            )
            proteins.append(assays)
        hierarchy.append(tuple(proteins))
    return tuple(hierarchy)


def _nested_task_sample(
    hierarchy: tuple[tuple[tuple[np.ndarray, ...], ...], ...],
    rng: np.random.Generator,
) -> list[str]:
    sampled: list[str] = []
    for family_index in rng.integers(0, len(hierarchy), size=len(hierarchy)):
        proteins = hierarchy[int(family_index)]
        for protein_index in rng.integers(0, len(proteins), size=len(proteins)):
            assays = proteins[int(protein_index)]
            for assay_index in rng.integers(0, len(assays), size=len(assays)):
                tasks = assays[int(assay_index)]
                sampled.extend(
                    tasks[rng.integers(0, len(tasks), size=len(tasks))].tolist()
                )
    return sampled


def bootstrap_policy_difference(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    comparator: str,
    best_average_model: str,
    repeats: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Nested family/protein/assay bootstrap of comparator minus VariantShift AURC."""
    repeats = repeats or config.bootstrap_repeats
    variantshift = _choose_task_rows(
        frame,
        config,
        policy="variantshift",
        seed=config.seed,
        best_average_model=best_average_model,
    ).set_index(config.task_column, drop=False)
    comparison = _choose_task_rows(
        frame,
        config,
        policy=comparator,
        seed=config.seed,
        best_average_model=best_average_model,
    ).set_index(config.task_column, drop=False)
    shared = variantshift.index.intersection(comparison.index)
    if len(shared) != len(variantshift) or len(shared) != len(comparison):
        raise ValueError("Selective policies must cover exactly the same tasks")
    task_metadata = variantshift.loc[
        shared,
        [
            config.task_column,
            config.group_column,
            config.protein_column,
            config.assay_column,
        ],
    ]
    hierarchy = _task_hierarchy(task_metadata, config)
    rng = np.random.default_rng(config.seed + 73)
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = _nested_task_sample(hierarchy, rng)
        variantshift_auc = _risk_auc(variantshift.loc[sampled], config.target_column)
        comparator_auc = _risk_auc(comparison.loc[sampled], config.target_column)
        estimates[repeat] = comparator_auc - variantshift_auc
    replicates = pd.DataFrame(
        {
            "repeat": np.arange(repeats),
            "comparator": comparator,
            "risk_coverage_auc_improvement": estimates,
        }
    )
    point = _risk_auc(comparison, config.target_column) - _risk_auc(
        variantshift, config.target_column
    )
    summary = {
        "comparator": comparator,
        "risk_coverage_auc_improvement": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_improvement_above_zero": float(np.mean(estimates > 0)),
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "family, then protein, then assay",
    }
    return replicates, summary


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    family_column: str,
    protein_column: str,
    assay_column: str,
    repeats: int = 10_000,
    seed: int = 20260830,
) -> pd.DataFrame:
    """Resample families, proteins, and assays while preserving nested clustering."""
    rng = np.random.default_rng(seed)
    family_values = frame[family_column].astype(str).unique()
    if not len(family_values):
        raise ValueError("Hierarchical bootstrap requires at least one family")
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        values: list[float] = []
        sampled_families = rng.choice(family_values, size=len(family_values), replace=True)
        for family in sampled_families:
            family_frame = frame.loc[frame[family_column].astype(str).eq(family)]
            proteins = family_frame[protein_column].astype(str).unique()
            for protein in rng.choice(proteins, size=len(proteins), replace=True):
                protein_frame = family_frame.loc[
                    family_frame[protein_column].astype(str).eq(protein)
                ]
                assays = protein_frame[assay_column].astype(str).unique()
                for assay in rng.choice(assays, size=len(assays), replace=True):
                    assay_values = protein_frame.loc[
                        protein_frame[assay_column].astype(str).eq(assay), value_column
                    ].to_numpy(dtype=float)
                    values.append(float(np.mean(assay_values)))
        estimates[repeat] = float(np.mean(values))
    return pd.DataFrame({"repeat": np.arange(repeats), "estimate": estimates})


def fit_transportability(
    features_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    confirmation_features_path: Path | None = None,
) -> dict[str, Path]:
    frame = pd.read_csv(features_path)
    config = load_transport_config(config_path)
    TASK_METRIC_SCHEMA.validate(frame)
    if config.task_column not in frame:
        frame[config.task_column] = (
            frame["panel_id"].astype(str)
            + "::"
            + frame["dataset_id"].astype(str)
            + "::"
            + frame["assay_id"].astype(str)
            + "::"
            + frame["target_id"].astype(str)
        )
    crossfit, fold_audit = cross_fitted_transport_predictions(frame, config)
    elastic, elastic_audit = cross_fitted_transport_predictions(
        frame, config, estimator="elastic_net"
    )
    crossfit["elastic_predicted_selection_gain_sd"] = elastic[
        "predicted_selection_gain_sd"
    ].to_numpy(dtype=float)
    bundle = fit_frozen_transport_model(frame, config)
    best_model = str(bundle["best_average_model"])
    policies = [
        "variantshift",
        "uncalibrated",
        "elastic_net",
        "always_best",
        "random",
        "msa_depth",
        "score_dispersion",
        "ensemble_agreement",
        "oracle",
    ]
    if "crossover_probability_supervised_wins" in crossfit:
        policies.append("crossover_classifier")
    curves = pd.concat(
        [
            selective_policy_curve(
                crossfit,
                config,
                policy=policy,
                best_average_model=best_model,
            )
            for policy in policies
        ],
        ignore_index=True,
    )
    summary = summarize_policy_curves(curves)
    comparator_candidates = summary.loc[
        summary["policy"].isin(
            [
                "uncalibrated",
                "elastic_net",
                "msa_depth",
                "score_dispersion",
                "ensemble_agreement",
                "crossover_classifier",
            ]
        )
    ]
    best_comparator = str(
        comparator_candidates.sort_values("risk_coverage_auc").iloc[0]["policy"]
    )
    bootstrap, bootstrap_summary = bootstrap_policy_difference(
        crossfit,
        config,
        comparator=best_comparator,
        best_average_model=best_model,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "crossfit": output_dir / "transport-crossfit.csv",
        "elastic_crossfit": output_dir / "elastic-net-crossfit.csv",
        "fold_audit": output_dir / "transport-fold-audit.csv",
        "elastic_fold_audit": output_dir / "elastic-net-fold-audit.csv",
        "curves": output_dir / "risk-coverage.csv",
        "summary": output_dir / "transport-summary.csv",
        "bootstrap": output_dir / "transport-bootstrap.csv.gz",
        "bootstrap_summary": output_dir / "transport-bootstrap-summary.csv",
        "bundle": output_dir / "transport-model.joblib",
        "method": output_dir / "transport-method.json",
    }
    write_table(crossfit, outputs["crossfit"])
    write_table(elastic, outputs["elastic_crossfit"])
    write_table(fold_audit, outputs["fold_audit"])
    write_table(elastic_audit, outputs["elastic_fold_audit"])
    write_table(curves, outputs["curves"])
    write_table(summary, outputs["summary"])
    bootstrap.to_csv(outputs["bootstrap"], index=False, compression="gzip")
    write_table(pd.DataFrame([bootstrap_summary]), outputs["bootstrap_summary"])
    joblib.dump(bundle, outputs["bundle"])
    method = {
        "schema_version": 1,
        "name": "VariantShift Transportability Score",
        "config": asdict(config),
        "parameters": bundle["parameters"],
        "conformal_quantile": bundle["conformal_quantile"],
        "best_average_model": best_model,
        "best_label_free_comparator": best_comparator,
        "training_frame_sha256": bundle["training_frame_sha256"],
        "formal_scope": "one-sided group conformal under exchangeable protein-family tasks",
        "primary_failure": "selection_gain_sd <= 0",
        "primary_endpoint": "task-level failure risk-coverage AUC",
    }
    outputs["method"].write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if confirmation_features_path is not None:
        confirmation = pd.read_csv(confirmation_features_path)
        predictions = predict_with_frozen_transport_model(bundle, confirmation)
        outputs["confirmation_predictions"] = (
            output_dir / "confirmation-transport-predictions.csv"
        )
        write_table(predictions, outputs["confirmation_predictions"])
    return outputs


def evaluate_frozen_transportability(
    bundle_path: Path,
    frozen_predictions_path: Path,
    outcomes_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    bundle = joblib.load(bundle_path)
    config = TransportConfig.from_dict(dict(bundle["config"]))
    predictions = pd.read_csv(frozen_predictions_path)
    outcomes = pd.read_csv(outcomes_path)
    TASK_METRIC_SCHEMA.validate(outcomes)
    join_columns = [
        config.task_column,
        config.model_column,
        config.group_column,
        config.protein_column,
        config.assay_column,
    ]
    join_columns = [column for column in join_columns if column in predictions.columns]
    merged = predictions.merge(
        outcomes.loc[:, [*join_columns, config.target_column]],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Frozen predictions and confirmation outcomes have no shared tasks")
    best_model = str(bundle["best_average_model"])
    policies = ["variantshift", "uncalibrated", "always_best", "random", "oracle"]
    for optional_policy, column in (
        ("msa_depth", "msa_neff"),
        ("score_dispersion", "score_dispersion"),
        ("ensemble_agreement", "ensemble_agreement"),
    ):
        if column in merged:
            policies.append(optional_policy)
    curves = pd.concat(
        [
            selective_policy_curve(
                merged,
                config,
                policy=policy,
                best_average_model=best_model,
            )
            for policy in policies
        ],
        ignore_index=True,
    )
    selected = _choose_task_rows(
        merged,
        config,
        policy="variantshift",
        seed=config.seed,
        best_average_model=best_model,
    )
    coverage = float(
        np.mean(
            selected[config.target_column].to_numpy(dtype=float)
            >= selected["lower_selection_gain_sd"].to_numpy(dtype=float)
        )
    )
    summary = summarize_policy_curves(curves)
    summary["observed_lower_bound_coverage"] = coverage
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "merged": output_dir / "confirmation-task-predictions.csv",
        "curves": output_dir / "confirmation-risk-coverage.csv",
        "summary": output_dir / "confirmation-transport-summary.csv",
    }
    write_table(merged, outputs["merged"])
    write_table(curves, outputs["curves"])
    write_table(summary, outputs["summary"])
    return outputs
