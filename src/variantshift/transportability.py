"""VariantShift selective transport auditing and task-level evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import (
    RISK_COVERAGE_SCHEMA,
    TASK_METRIC_SCHEMA,
    TRANSPORT_FEATURE_SCHEMA,
    stable_frame_sha256,
    write_table,
)

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
    "supervised_wins",
    "crossover_probability",
)

CONFIRMATION_PRIMARY_PANELS = (
    "human-domainome-v1",
    "mavedb-complement-v1",
)

MSA_FEATURE_MARKERS = ("msa", "alignment", "entropy")
ENSEMBLE_FEATURE_MARKERS = ("ensemble", "disagreement", "agreement")


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

# Fixed before confirmation. The value is selected inside family-held-out folds, so the
# deployment rule can borrow strength from stable model-level performance without reporting a
# post-selection development estimate as if the shrinkage had been known in advance.
DECISION_SHRINKAGE_CANDIDATES: tuple[float, ...] = (0.5, 0.75, 1.0)
OVERRIDE_MARGIN_CANDIDATES: tuple[float, ...] = (0.05, 0.075, 0.1, 0.15)
# Frozen after development. Every component is available from sequence, MSA, and model predictions
# before confirmation assay outcomes are opened.
PRIORITY_WEIGHTS = {
    "log_msa_neff": 1.0,
    "log_protein_length": 0.5,
    "decision_score": 0.25,
}


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


def _scale_pipeline(config: TransportConfig, *, seed_offset: int) -> Pipeline:
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=config.seed + 10_000 + seed_offset,
    )
    return Pipeline(
        [("preprocess", _preprocessor(config, scale_numeric=False)), ("model", model)]
    )


def _out_of_group_predictions(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    estimator: str,
    parameters: dict[str, float | int],
    seed_offset: int,
    decision_shrinkage: float = 1.0,
) -> np.ndarray:
    target = frame[config.target_column].to_numpy(dtype=float)
    groups = frame[config.group_column].astype(str).to_numpy()
    splits = min(4, len(np.unique(groups)))
    if splits < 2:
        raise ValueError("Error-scale fitting requires at least two family groups")
    predicted = np.full(len(frame), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, target, groups)
    ):
        model = _pipeline(
            config,
            estimator=estimator,
            parameters=parameters,
            seed_offset=seed_offset * 10 + fold,
        )
        model.fit(frame.iloc[train_indices], target[train_indices])
        raw_prediction = model.predict(frame.iloc[test_indices])
        predicted[test_indices] = _decision_scores(
            frame.iloc[test_indices],
            raw_prediction,
            config,
            priors=_model_priors(frame.iloc[train_indices], config),
            shrinkage=decision_shrinkage,
        )
    if not np.isfinite(predicted).all():
        raise RuntimeError("Group-held-out error scale did not cover every fit row")
    return predicted


def _fit_error_scale(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    estimator: str,
    parameters: dict[str, float | int],
    seed_offset: int,
    decision_shrinkage: float = 1.0,
) -> tuple[Pipeline, float]:
    target = frame[config.target_column].to_numpy(dtype=float)
    out_of_group = _out_of_group_predictions(
        frame,
        config,
        estimator=estimator,
        parameters=parameters,
        seed_offset=seed_offset,
        decision_shrinkage=decision_shrinkage,
    )
    absolute_error = np.abs(out_of_group - target)
    positive = absolute_error[absolute_error > 0]
    scale_floor = max(
        1e-3,
        0.05 * float(np.median(positive)) if len(positive) else 1e-3,
    )
    scale_model = _scale_pipeline(config, seed_offset=seed_offset)
    scale_model.fit(frame, absolute_error)
    return scale_model, scale_floor


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


def hierarchical_conformal_quantile(
    residuals: np.ndarray,
    groups: np.ndarray,
    *,
    coverage: float,
) -> float:
    """Hierarchical split-conformal quantile for a new task from a new group.

    Every calibration family receives equal total mass, and tasks within a family share that
    mass. The remaining 1 / (n_families + 1) mass is placed at infinity. This is the finite-sample
    construction for two-level hierarchical exchangeability; unlike a family maximum, it targets
    marginal coverage for a randomly sampled task in an unseen family rather than simultaneous
    coverage of every task in that family.
    """
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie in (0, 1)")
    values = pd.DataFrame(
        {"residual": np.asarray(residuals, dtype=float), "group": groups.astype(str)}
    )
    if values.empty or not np.isfinite(values["residual"]).all():
        raise ValueError("Finite residuals from at least one calibration group are required")
    group_count = int(values["group"].nunique())
    weighted_values: list[float] = []
    weights: list[float] = []
    for _group, selected in values.groupby("group", sort=False):
        group_size = len(selected)
        weighted_values.extend(selected["residual"].tolist())
        weights.extend([1.0 / ((group_count + 1) * group_size)] * group_size)
    weighted_values.append(float("inf"))
    weights.append(1.0 / (group_count + 1))
    order = np.argsort(np.asarray(weighted_values, dtype=float), kind="stable")
    ordered_values = np.asarray(weighted_values, dtype=float)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    index = min(len(ordered_values) - 1, int(np.searchsorted(cumulative, coverage)))
    return float(ordered_values[index])


def _model_priors(frame: pd.DataFrame, config: TransportConfig) -> dict[str, float]:
    return {
        str(model): float(value)
        for model, value in frame.groupby(config.model_column, sort=True)[
            config.target_column
        ].mean().items()
    }


def _decision_scores(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    config: TransportConfig,
    *,
    priors: dict[str, float],
    shrinkage: float,
) -> np.ndarray:
    if not 0 <= shrinkage <= 1:
        raise ValueError("Decision shrinkage must lie in [0, 1]")
    model_ids = frame[config.model_column].astype(str)
    prior_values = model_ids.map(priors)
    if prior_values.isna().any():
        missing = sorted(model_ids.loc[prior_values.isna()].unique())
        raise ValueError(f"Decision priors are missing models: {missing}")
    return (
        shrinkage * np.asarray(predicted, dtype=float)
        + (1.0 - shrinkage) * prior_values.to_numpy(dtype=float)
    )


def _mean_selected_regret(
    frame: pd.DataFrame,
    scores: np.ndarray,
    config: TransportConfig,
) -> float:
    working = frame.loc[:, [config.task_column, config.target_column]].copy()
    working["_decision_score"] = np.asarray(scores, dtype=float)
    selected = working.loc[working.groupby(config.task_column)["_decision_score"].idxmax()]
    oracle = working.groupby(config.task_column)[config.target_column].max()
    observed = selected.set_index(config.task_column)[config.target_column]
    return float((oracle - observed).mean())


def _selected_with_override(
    frame: pd.DataFrame,
    scores: np.ndarray,
    config: TransportConfig,
    *,
    baseline_models: np.ndarray,
    margin: float,
) -> pd.DataFrame:
    working = frame.copy()
    working["_source_index"] = np.arange(len(working), dtype=int)
    working["_decision_score"] = np.asarray(scores, dtype=float)
    working["_baseline_model"] = np.asarray(baseline_models, dtype=str)
    top_indices = working.groupby(config.task_column)["_decision_score"].idxmax()
    top = working.loc[top_indices].set_index(config.task_column, drop=False)
    baseline_candidates = working.loc[
        working[config.model_column].astype(str).eq(working["_baseline_model"])
    ]
    baseline = baseline_candidates.set_index(config.task_column, drop=False)
    if len(baseline) != working[config.task_column].nunique():
        raise ValueError("Every task must contain its development-baseline model")
    use_override = (
        top["_decision_score"] - baseline["_decision_score"]
    ) > margin
    selected_rows = []
    for task_id in top.index:
        selected_rows.append(
            top.loc[task_id] if bool(use_override.loc[task_id]) else baseline.loc[task_id]
        )
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["_overrode_baseline"] = use_override.to_numpy(dtype=bool)
    return selected


def _apply_auditor_policy(
    frame: pd.DataFrame,
    scores: np.ndarray,
    config: TransportConfig,
    policy: dict[str, object],
) -> pd.DataFrame:
    baseline_models = np.repeat(str(policy["baseline_model"]), len(frame))
    selected = _selected_with_override(
        frame.reset_index(drop=True),
        scores,
        config,
        baseline_models=baseline_models,
        margin=float(policy["override_margin"]),
    )
    selected["confidence"] = _priority_values(
        selected,
        decision_mean=float(policy["decision_mean"]),
        decision_std=float(policy["decision_std"]),
        log_msa_neff_mean=float(policy["log_msa_neff_mean"]),
        log_msa_neff_std=float(policy["log_msa_neff_std"]),
        log_protein_length_mean=float(policy["log_protein_length_mean"]),
        log_protein_length_std=float(policy["log_protein_length_std"]),
    )
    return selected


def _safe_standard_deviation(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0
    standard_deviation = float(np.std(finite, ddof=0))
    return standard_deviation if standard_deviation > 1e-12 else 1.0


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _log1p_nonnegative(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
    numeric[numeric < 0] = np.nan
    return np.log1p(numeric)


def _priority_values(
    selected: pd.DataFrame,
    *,
    decision_mean: float,
    decision_std: float,
    log_msa_neff_mean: float,
    log_msa_neff_std: float,
    log_protein_length_mean: float,
    log_protein_length_std: float,
) -> np.ndarray:
    decision = selected["_decision_score"].to_numpy(dtype=float)
    if not np.isfinite(decision).all():
        raise ValueError("Auditor decision scores must be finite")
    log_msa_neff = _log1p_nonnegative(selected["msa_neff"])
    log_protein_length = _log1p_nonnegative(selected["protein_length"])
    # Missing deployment metadata is neutral rather than silently promoted or rejected. The
    # imputation constants are learned on development data and serialized with the policy.
    log_msa_neff = np.where(
        np.isfinite(log_msa_neff), log_msa_neff, log_msa_neff_mean
    )
    log_protein_length = np.where(
        np.isfinite(log_protein_length),
        log_protein_length,
        log_protein_length_mean,
    )
    return (
        PRIORITY_WEIGHTS["log_msa_neff"]
        * (log_msa_neff - log_msa_neff_mean)
        / log_msa_neff_std
        + PRIORITY_WEIGHTS["log_protein_length"]
        * (log_protein_length - log_protein_length_mean)
        / log_protein_length_std
        + PRIORITY_WEIGHTS["decision_score"]
        * (decision - decision_mean)
        / decision_std
    )


def select_auditor_policy(
    frame: pd.DataFrame,
    config: TransportConfig,
    *,
    parameters: dict[str, float | int],
    seed_offset: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Choose the complete selector and ranking rule with family-held-out predictions."""
    target = frame[config.target_column].to_numpy(dtype=float)
    groups = frame[config.group_column].astype(str).to_numpy()
    splits = min(4, len(np.unique(groups)))
    if splits < 2:
        priors = _model_priors(frame, config)
        return (
            {
                "decision_shrinkage": 1.0,
                "override_margin": 0.0,
                "decision_mean": 0.0,
                "decision_std": 1.0,
                "log_msa_neff_mean": 0.0,
                "log_msa_neff_std": 1.0,
                "log_protein_length_mean": 0.0,
                "log_protein_length_std": 1.0,
                "baseline_model": max(priors, key=priors.get),
            },
            {"candidate_count": 1, "best_regret_coverage_auc": float("nan")},
        )
    raw_prediction = np.full(len(frame), np.nan, dtype=float)
    prior_prediction: dict[float, np.ndarray] = {
        value: np.full(len(frame), np.nan, dtype=float)
        for value in DECISION_SHRINKAGE_CANDIDATES
    }
    baseline_models = np.full(len(frame), "", dtype=object)
    splitter = GroupKFold(n_splits=splits)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, target, groups)
    ):
        model = _pipeline(
            config,
            estimator="histgb",
            parameters=parameters,
            seed_offset=30_000 + seed_offset * 10 + fold,
        )
        model.fit(frame.iloc[train_indices], target[train_indices])
        predicted = model.predict(frame.iloc[test_indices])
        raw_prediction[test_indices] = predicted
        priors = _model_priors(frame.iloc[train_indices], config)
        baseline_models[test_indices] = max(priors, key=priors.get)
        for shrinkage in DECISION_SHRINKAGE_CANDIDATES:
            prior_prediction[shrinkage][test_indices] = _decision_scores(
                frame.iloc[test_indices],
                predicted,
                config,
                priors=priors,
                shrinkage=shrinkage,
            )
    if not np.isfinite(raw_prediction).all():
        raise RuntimeError("Shrinkage selection did not cover every held-out family")
    oracle = frame.groupby(config.task_column)[config.target_column].max()
    candidates: list[tuple[tuple[float, float, float, float], dict[str, object]]] = []
    for shrinkage, scores in prior_prediction.items():
        for margin in OVERRIDE_MARGIN_CANDIDATES:
            selected = _selected_with_override(
                frame,
                scores,
                config,
                baseline_models=baseline_models,
                margin=margin,
            )
            decision_values = selected["_decision_score"].to_numpy(dtype=float)
            log_msa_neff = _log1p_nonnegative(selected["msa_neff"])
            log_protein_length = _log1p_nonnegative(selected["protein_length"])
            decision_mean = _finite_mean(decision_values)
            decision_std = _safe_standard_deviation(decision_values)
            log_msa_neff_mean = _finite_mean(log_msa_neff)
            log_msa_neff_std = _safe_standard_deviation(log_msa_neff)
            log_protein_length_mean = _finite_mean(log_protein_length)
            log_protein_length_std = _safe_standard_deviation(log_protein_length)
            mean_regret = float(
                (
                    selected[config.task_column].map(oracle).to_numpy(dtype=float)
                    - selected[config.target_column].to_numpy(dtype=float)
                ).mean()
            )
            selected["confidence"] = _priority_values(
                selected,
                decision_mean=decision_mean,
                decision_std=decision_std,
                log_msa_neff_mean=log_msa_neff_mean,
                log_msa_neff_std=log_msa_neff_std,
                log_protein_length_mean=log_protein_length_mean,
                log_protein_length_std=log_protein_length_std,
            )
            regret_auc = _regret_auc(
                selected,
                oracle,
                task_column=config.task_column,
                target_column=config.target_column,
            )
            policy = {
                "decision_shrinkage": float(shrinkage),
                "override_margin": float(margin),
                "decision_mean": decision_mean,
                "decision_std": decision_std,
                "log_msa_neff_mean": log_msa_neff_mean,
                "log_msa_neff_std": log_msa_neff_std,
                "log_protein_length_mean": log_protein_length_mean,
                "log_protein_length_std": log_protein_length_std,
            }
            candidates.append(
                (
                    (regret_auc, mean_regret, -margin, -shrinkage),
                    policy,
                )
            )
    objective, policy = min(candidates, key=lambda item: item[0])
    priors = _model_priors(frame, config)
    policy["baseline_model"] = max(priors, key=priors.get)
    audit = {
        "candidate_count": len(candidates),
        "best_regret_coverage_auc": float(objective[0]),
        "best_mean_regret": float(objective[1]),
        "selected_policy": dict(policy),
    }
    return policy, audit


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
    decision_score = np.full(len(frame), np.nan, dtype=float)
    lower = np.full(len(frame), np.nan, dtype=float)
    predicted_scale = np.full(len(frame), np.nan, dtype=float)
    auditor_selected = np.zeros(len(frame), dtype=bool)
    auditor_confidence = np.full(len(frame), np.nan, dtype=float)
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
        auditor_policy, policy_audit = (
            select_auditor_policy(
                frame.iloc[fit_indices],
                config,
                parameters=parameters,
                seed_offset=fold,
            )
            if estimator == "histgb"
            else (
                {
                    "decision_shrinkage": 1.0,
                    "override_margin": 0.0,
                    "decision_mean": 0.0,
                    "decision_std": 1.0,
                    "log_msa_neff_mean": float(
                        np.log1p(frame.iloc[fit_indices]["msa_neff"]).mean()
                    ),
                    "log_msa_neff_std": _safe_standard_deviation(
                        np.log1p(frame.iloc[fit_indices]["msa_neff"])
                    ),
                    "log_protein_length_mean": float(
                        np.log1p(frame.iloc[fit_indices]["protein_length"]).mean()
                    ),
                    "log_protein_length_std": _safe_standard_deviation(
                        np.log1p(frame.iloc[fit_indices]["protein_length"])
                    ),
                    "baseline_model": max(
                        _model_priors(frame.iloc[fit_indices], config),
                        key=_model_priors(frame.iloc[fit_indices], config).get,
                    ),
                },
                {"candidate_count": 1},
            )
        )
        decision_shrinkage = float(auditor_policy["decision_shrinkage"])
        priors = _model_priors(frame.iloc[fit_indices], config)
        model = _pipeline(
            config,
            estimator=estimator,
            parameters=parameters,
            seed_offset=fold,
        )
        model.fit(frame.iloc[fit_indices], target[fit_indices])
        scale_model, scale_floor = _fit_error_scale(
            frame.iloc[fit_indices],
            config,
            estimator=estimator,
            parameters=parameters,
            seed_offset=fold,
            decision_shrinkage=decision_shrinkage,
        )
        calibration_predictions = model.predict(frame.iloc[calibration_indices])
        calibration_decisions = _decision_scores(
            frame.iloc[calibration_indices],
            calibration_predictions,
            config,
            priors=priors,
            shrinkage=decision_shrinkage,
        )
        calibration_scale = np.maximum(
            scale_model.predict(frame.iloc[calibration_indices]), scale_floor
        )
        calibration = frame.iloc[calibration_indices].reset_index(drop=True).copy()
        calibration["_decision_score"] = calibration_decisions
        calibration["_predicted_scale"] = calibration_scale
        calibration["_target"] = target[calibration_indices]
        selected_calibration = _apply_auditor_policy(
            calibration,
            calibration_decisions,
            config,
            auditor_policy,
        )
        quantile = hierarchical_conformal_quantile(
            (
                selected_calibration["_decision_score"]
                - selected_calibration["_target"]
            ).to_numpy(dtype=float)
            / selected_calibration["_predicted_scale"].to_numpy(dtype=float),
            selected_calibration[config.group_column].astype(str).to_numpy(),
            coverage=config.coverage,
        )
        predicted[test_indices] = model.predict(frame.iloc[test_indices])
        decision_score[test_indices] = _decision_scores(
            frame.iloc[test_indices],
            predicted[test_indices],
            config,
            priors=priors,
            shrinkage=decision_shrinkage,
        )
        predicted_scale[test_indices] = np.maximum(
            scale_model.predict(frame.iloc[test_indices]), scale_floor
        )
        lower[test_indices] = (
            decision_score[test_indices] - quantile * predicted_scale[test_indices]
        )
        selected_test = _apply_auditor_policy(
            frame.iloc[test_indices].reset_index(drop=True),
            decision_score[test_indices],
            config,
            auditor_policy,
        )
        selected_local = selected_test["_source_index"].to_numpy(dtype=int)
        selected_global = test_indices[selected_local]
        auditor_selected[selected_global] = True
        auditor_confidence[selected_global] = selected_test["confidence"].to_numpy(
            dtype=float
        )
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
                "calibration_tasks": int(
                    selected_calibration[config.task_column].nunique()
                ),
                "test_groups": len(np.unique(groups[test_indices])),
                "conformal_quantile": quantile,
                "decision_shrinkage": decision_shrinkage,
                "override_margin": auditor_policy["override_margin"],
                "priority_weights": json.dumps(PRIORITY_WEIGHTS, sort_keys=True),
                "policy_inner_audit": json.dumps(policy_audit, sort_keys=True),
                "median_predicted_error_scale": float(
                    np.median(predicted_scale[test_indices])
                ),
                "error_scale_floor": scale_floor,
                "parameters": json.dumps(parameters, sort_keys=True),
            }
        )
    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(decision_score).all()
        or (fold_ids < 0).any()
    ):
        raise RuntimeError("Cross-fitting did not generate every held-out prediction")
    output = frame.copy()
    output["transport_estimator"] = estimator
    output["fold"] = fold_ids
    output["predicted_selection_gain_sd"] = predicted
    output["decision_selection_gain_sd"] = decision_score
    output["predicted_error_scale"] = predicted_scale
    output["lower_selection_gain_sd"] = lower
    output["auditor_selected"] = auditor_selected
    output["auditor_confidence"] = auditor_confidence
    output["trusted"] = auditor_selected & (lower > 0)
    output["prediction_error"] = predicted - target
    output["decision_prediction_error"] = decision_score - target
    output["lower_bound_covers"] = target >= lower
    return output, pd.DataFrame(parameter_rows)


def fit_frozen_transport_model(
    frame: pd.DataFrame,
    config: TransportConfig,
) -> dict[str, object]:
    """Fit the deployable model and keep a disjoint family calibration set."""
    fit_indices, calibration_indices = _fit_calibration_indices(frame, config, seed_offset=991)
    parameters = select_histgb_parameters(frame.iloc[fit_indices], config)
    auditor_policy, policy_audit = select_auditor_policy(
        frame.iloc[fit_indices],
        config,
        parameters=parameters,
        seed_offset=991,
    )
    decision_shrinkage = float(auditor_policy["decision_shrinkage"])
    priors = _model_priors(frame.iloc[fit_indices], config)
    target = frame[config.target_column].to_numpy(dtype=float)
    model = _pipeline(config, estimator="histgb", parameters=parameters, seed_offset=991)
    model.fit(frame.iloc[fit_indices], target[fit_indices])
    scale_model, scale_floor = _fit_error_scale(
        frame.iloc[fit_indices],
        config,
        estimator="histgb",
        parameters=parameters,
        seed_offset=991,
        decision_shrinkage=decision_shrinkage,
    )
    calibration_predictions = model.predict(frame.iloc[calibration_indices])
    calibration_decisions = _decision_scores(
        frame.iloc[calibration_indices],
        calibration_predictions,
        config,
        priors=priors,
        shrinkage=decision_shrinkage,
    )
    calibration_scale = np.maximum(
        scale_model.predict(frame.iloc[calibration_indices]), scale_floor
    )
    calibration = frame.iloc[calibration_indices].reset_index(drop=True).copy()
    calibration["_decision_score"] = calibration_decisions
    calibration["_predicted_scale"] = calibration_scale
    calibration["_target"] = target[calibration_indices]
    selected_calibration = _apply_auditor_policy(
        calibration,
        calibration_decisions,
        config,
        auditor_policy,
    )
    quantile = hierarchical_conformal_quantile(
        (
            selected_calibration["_decision_score"]
            - selected_calibration["_target"]
        ).to_numpy(dtype=float)
        / selected_calibration["_predicted_scale"].to_numpy(dtype=float),
        selected_calibration[config.group_column].astype(str).to_numpy(),
        coverage=config.coverage,
    )
    best_average_model = (
        frame.groupby(config.model_column)[config.target_column].mean().idxmax()
    )
    elastic_model = _pipeline(config, estimator="elastic_net")
    elastic_model.fit(frame, target)
    ablations: dict[str, dict[str, object]] = {}
    for name, ablation_config in _ablation_configs(config).items():
        ablation_fit, ablation_calibration = _fit_calibration_indices(
            frame, ablation_config, seed_offset=2_000 + len(ablations)
        )
        ablation_parameters = select_histgb_parameters(
            frame.iloc[ablation_fit], ablation_config
        )
        ablation_policy, ablation_policy_audit = select_auditor_policy(
            frame.iloc[ablation_fit],
            ablation_config,
            parameters=ablation_parameters,
            seed_offset=2_000 + len(ablations),
        )
        ablation_shrinkage = float(ablation_policy["decision_shrinkage"])
        ablation_priors = _model_priors(frame.iloc[ablation_fit], ablation_config)
        ablation_model = _pipeline(
            ablation_config,
            estimator="histgb",
            parameters=ablation_parameters,
            seed_offset=2_000 + len(ablations),
        )
        ablation_model.fit(frame.iloc[ablation_fit], target[ablation_fit])
        ablation_scale_model, ablation_scale_floor = _fit_error_scale(
            frame.iloc[ablation_fit],
            ablation_config,
            estimator="histgb",
            parameters=ablation_parameters,
            seed_offset=2_000 + len(ablations),
            decision_shrinkage=ablation_shrinkage,
        )
        ablation_calibration_prediction = ablation_model.predict(
            frame.iloc[ablation_calibration]
        )
        ablation_calibration_decision = _decision_scores(
            frame.iloc[ablation_calibration],
            ablation_calibration_prediction,
            ablation_config,
            priors=ablation_priors,
            shrinkage=ablation_shrinkage,
        )
        ablation_calibration_scale = np.maximum(
            ablation_scale_model.predict(frame.iloc[ablation_calibration]),
            ablation_scale_floor,
        )
        ablation_calibration_frame = (
            frame.iloc[ablation_calibration].reset_index(drop=True).copy()
        )
        ablation_calibration_frame["_decision_score"] = (
            ablation_calibration_decision
        )
        ablation_calibration_frame["_predicted_scale"] = (
            ablation_calibration_scale
        )
        ablation_calibration_frame["_target"] = target[ablation_calibration]
        ablation_selected = _apply_auditor_policy(
            ablation_calibration_frame,
            ablation_calibration_decision,
            ablation_config,
            ablation_policy,
        )
        ablation_quantile = hierarchical_conformal_quantile(
            (
                ablation_selected["_decision_score"]
                - ablation_selected["_target"]
            ).to_numpy(dtype=float)
            / ablation_selected["_predicted_scale"].to_numpy(dtype=float),
            ablation_selected[ablation_config.group_column].astype(str).to_numpy(),
            coverage=ablation_config.coverage,
        )
        ablations[name] = {
            "config": asdict(ablation_config),
            "pipeline": ablation_model,
            "scale_pipeline": ablation_scale_model,
            "scale_floor": ablation_scale_floor,
            "conformal_quantile": ablation_quantile,
            "parameters": ablation_parameters,
            "decision_shrinkage": ablation_shrinkage,
            "decision_priors": ablation_priors,
            "auditor_policy": ablation_policy,
            "policy_inner_audit": ablation_policy_audit,
        }
    return {
        "schema_version": 2,
        "pipeline": model,
        "scale_pipeline": scale_model,
        "elastic_pipeline": elastic_model,
        "scale_floor": scale_floor,
        "config": asdict(config),
        "parameters": parameters,
        "conformal_quantile": quantile,
        "decision_shrinkage": decision_shrinkage,
        "decision_priors": priors,
        "auditor_policy": auditor_policy,
        "policy_inner_audit": policy_audit,
        "calibration_target": "selected task-model pair",
        "fit_rows": fit_indices.tolist(),
        "calibration_rows": calibration_indices.tolist(),
        "best_average_model": str(best_average_model),
        "ablations": ablations,
        "confirmation_primary_panels": list(CONFIRMATION_PRIMARY_PANELS),
        "training_frame_sha256": stable_frame_sha256(frame),
    }


def _ablation_configs(config: TransportConfig) -> dict[str, TransportConfig]:
    def contains_any(feature: str, markers: tuple[str, ...]) -> bool:
        normalized = feature.lower()
        return any(marker in normalized for marker in markers)

    msa_features = tuple(
        feature
        for feature in config.numeric_features
        if contains_any(feature, MSA_FEATURE_MARKERS)
    )
    ensemble_features = tuple(
        feature
        for feature in config.numeric_features
        if contains_any(feature, ENSEMBLE_FEATURE_MARKERS)
    )
    common_categorical = tuple(
        feature
        for feature in config.categorical_features
        if feature in {"model_family", "model_modalities", "exposure_status"}
    )
    candidates = {
        "msa_only": replace(
            config,
            numeric_features=msa_features,
            categorical_features=common_categorical,
        ),
        "ensemble_only": replace(
            config,
            numeric_features=ensemble_features,
            categorical_features=common_categorical,
        ),
        "without_msa": replace(
            config,
            numeric_features=tuple(
                feature
                for feature in config.numeric_features
                if not contains_any(feature, MSA_FEATURE_MARKERS)
            ),
        ),
        "without_ensemble": replace(
            config,
            numeric_features=tuple(
                feature
                for feature in config.numeric_features
                if not contains_any(feature, ENSEMBLE_FEATURE_MARKERS)
            ),
        ),
    }
    output = {}
    for name, candidate in candidates.items():
        if candidate.feature_columns and candidate.feature_columns != config.feature_columns:
            candidate.validate()
            output[name] = candidate
    return output


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
    decision = _decision_scores(
        frame,
        predicted,
        config,
        priors={
            str(name): float(value)
            for name, value in dict(bundle["decision_priors"]).items()
        },
        shrinkage=float(bundle["decision_shrinkage"]),
    )
    predicted_scale = np.maximum(
        bundle["scale_pipeline"].predict(frame), float(bundle["scale_floor"])
    )
    output = frame.copy()
    output["predicted_selection_gain_sd"] = predicted
    output["decision_selection_gain_sd"] = decision
    output["predicted_error_scale"] = predicted_scale
    output["lower_selection_gain_sd"] = (
        decision - float(bundle["conformal_quantile"]) * predicted_scale
    )
    selected = _apply_auditor_policy(
        output,
        decision,
        config,
        dict(bundle["auditor_policy"]),
    )
    output["auditor_selected"] = False
    output["auditor_confidence"] = np.nan
    selected_indices = selected["_source_index"].to_numpy(dtype=int)
    output.loc[selected_indices, "auditor_selected"] = True
    output.loc[selected_indices, "auditor_confidence"] = selected[
        "confidence"
    ].to_numpy(dtype=float)
    output["trusted"] = output["auditor_selected"] & (
        output["lower_selection_gain_sd"] > 0
    )
    if "elastic_pipeline" in bundle:
        output["elastic_predicted_selection_gain_sd"] = bundle[
            "elastic_pipeline"
        ].predict(frame)
    for name, ablation in dict(bundle.get("ablations", {})).items():
        ablation_config = TransportConfig.from_dict(dict(ablation["config"]))
        ablation_config.validate()
        ablation_missing = sorted(
            set(ablation_config.feature_columns).difference(frame.columns)
        )
        if ablation_missing:
            raise ValueError(
                f"Confirmation features are missing {name} ablation columns: "
                f"{ablation_missing}"
            )
        ablation_prediction = ablation["pipeline"].predict(frame)
        ablation_decision = _decision_scores(
            frame,
            ablation_prediction,
            ablation_config,
            priors={
                str(model): float(value)
                for model, value in dict(ablation["decision_priors"]).items()
            },
            shrinkage=float(ablation["decision_shrinkage"]),
        )
        ablation_scale = np.maximum(
            ablation["scale_pipeline"].predict(frame),
            float(ablation["scale_floor"]),
        )
        prefix = f"ablation__{name}__"
        output[f"{prefix}predicted_selection_gain_sd"] = ablation_prediction
        output[f"{prefix}decision_selection_gain_sd"] = ablation_decision
        output[f"{prefix}predicted_error_scale"] = ablation_scale
        output[f"{prefix}lower_selection_gain_sd"] = (
            ablation_decision
            - float(ablation["conformal_quantile"]) * ablation_scale
        )
        ablation_selected = _apply_auditor_policy(
            output,
            ablation_decision,
            ablation_config,
            dict(ablation["auditor_policy"]),
        )
        output[f"{prefix}auditor_selected"] = False
        output[f"{prefix}auditor_confidence"] = np.nan
        ablation_indices = ablation_selected["_source_index"].to_numpy(dtype=int)
        output.loc[ablation_indices, f"{prefix}auditor_selected"] = True
        output.loc[ablation_indices, f"{prefix}auditor_confidence"] = (
            ablation_selected["confidence"].to_numpy(dtype=float)
        )
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
        if "auditor_selected" in working:
            selected = working.loc[working["auditor_selected"].astype(bool)].copy()
            if selected[config.task_column].nunique() != working[config.task_column].nunique():
                raise ValueError("Auditor policy must select exactly one model per task")
            selected["confidence"] = selected["auditor_confidence"]
        else:
            decision_column = (
                "decision_selection_gain_sd"
                if "decision_selection_gain_sd" in working
                else "predicted_selection_gain_sd"
            )
            index = working.groupby(config.task_column)[decision_column].idxmax()
            selected = working.loc[index].copy()
            selected["confidence"] = selected["lower_selection_gain_sd"]
    elif policy == "decision_only":
        decision_column = (
            "decision_selection_gain_sd"
            if "decision_selection_gain_sd" in working
            else "predicted_selection_gain_sd"
        )
        if "auditor_selected" in working:
            selected = working.loc[working["auditor_selected"].astype(bool)].copy()
        else:
            index = working.groupby(config.task_column)[decision_column].idxmax()
            selected = working.loc[index].copy()
        selected["confidence"] = selected[decision_column]
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
    elif policy.startswith("ablation:"):
        name = policy.removeprefix("ablation:")
        decision_column = f"ablation__{name}__decision_selection_gain_sd"
        confidence_column = f"ablation__{name}__lower_selection_gain_sd"
        if confidence_column not in working:
            raise ValueError(f"Frozen predictions do not contain ablation: {name}")
        selected_column = f"ablation__{name}__auditor_selected"
        auditor_confidence = f"ablation__{name}__auditor_confidence"
        if selected_column in working:
            selected = working.loc[working[selected_column].astype(bool)].copy()
            selected["confidence"] = selected[auditor_confidence]
        else:
            if decision_column not in working:
                decision_column = confidence_column
            index = working.groupby(config.task_column)[decision_column].idxmax()
            selected = working.loc[index].copy()
            selected["confidence"] = selected[confidence_column]
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
        }[policy]
        confidence = (
            np.zeros(len(selected))
            if confidence_column is None
            else selected[confidence_column].to_numpy(dtype=float)
        )
        selected["confidence"] = confidence
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
    oracle = frame.groupby(config.task_column)[config.target_column].max()
    selected["oracle_selection_gain_sd"] = selected[config.task_column].map(oracle)
    selected["selection_regret_sd"] = (
        selected["oracle_selection_gain_sd"] - selected[config.target_column]
    )
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(selected) * coverage)))
        retained = selected.iloc[:count]
        gains = retained[config.target_column].to_numpy(dtype=float)
        regrets = retained["selection_regret_sd"].to_numpy(dtype=float)
        tail_count = max(1, int(np.ceil(0.20 * len(gains))))
        worst_tail = np.sort(gains)[:tail_count]
        rows.append(
            {
                "policy": policy,
                "coverage": float(coverage),
                "retained_tasks": count,
                "failure_rate": float(np.mean(gains <= 0)),
                "mean_selection_gain_sd": float(np.mean(gains)),
                "median_selection_gain_sd": float(np.median(gains)),
                "mean_selection_regret_sd": float(np.mean(regrets)),
                "median_selection_regret_sd": float(np.median(regrets)),
                "worst_quintile_mean_gain_sd": float(np.mean(worst_tail)),
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
                    trapezoid(group["failure_rate"], group["coverage"])
                ),
                "regret_coverage_auc": float(
                    trapezoid(group["mean_selection_regret_sd"], group["coverage"])
                ),
                "utility_coverage_auc": float(
                    trapezoid(group["mean_selection_gain_sd"], group["coverage"])
                ),
                "failure_rate_at_50pct": float(at_half["failure_rate"]),
                "mean_selection_gain_at_50pct": float(at_half["mean_selection_gain_sd"]),
                "mean_selection_regret_at_50pct": float(
                    at_half["mean_selection_regret_sd"]
                ),
                "worst_quintile_gain_at_50pct": float(
                    at_half["worst_quintile_mean_gain_sd"]
                ),
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
    return float(trapezoid(risks, coverages))


def _regret_auc(
    selected: pd.DataFrame,
    oracle: pd.Series,
    *,
    task_column: str,
    target_column: str,
) -> float:
    ordered = selected.sort_values("confidence", ascending=False, kind="stable").copy()
    ordered["_regret"] = (
        ordered[task_column].map(oracle).to_numpy(dtype=float)
        - ordered[target_column].to_numpy(dtype=float)
    )
    coverages = np.linspace(0.1, 1.0, 10)
    regrets = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(ordered) * coverage)))
        regrets.append(float(ordered.iloc[:count]["_regret"].mean()))
    return float(trapezoid(regrets, coverages))


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
    """Nested bootstrap of comparator minus VariantShift selective-deployment utility."""
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
    oracle = frame.groupby(config.task_column)[config.target_column].max()
    rng = np.random.default_rng(config.seed + 73)
    regret_estimates = np.empty(repeats, dtype=float)
    failure_estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = _nested_task_sample(hierarchy, rng)
        variantshift_auc = _risk_auc(variantshift.loc[sampled], config.target_column)
        comparator_auc = _risk_auc(comparison.loc[sampled], config.target_column)
        failure_estimates[repeat] = comparator_auc - variantshift_auc
        variantshift_regret = _regret_auc(
            variantshift.loc[sampled],
            oracle,
            task_column=config.task_column,
            target_column=config.target_column,
        )
        comparator_regret = _regret_auc(
            comparison.loc[sampled],
            oracle,
            task_column=config.task_column,
            target_column=config.target_column,
        )
        regret_estimates[repeat] = comparator_regret - variantshift_regret
    replicates = pd.DataFrame(
        {
            "repeat": np.arange(repeats),
            "comparator": comparator,
            "regret_coverage_auc_improvement": regret_estimates,
            "risk_coverage_auc_improvement": failure_estimates,
        }
    )
    risk_point = _risk_auc(comparison, config.target_column) - _risk_auc(
        variantshift, config.target_column
    )
    regret_point = _regret_auc(
        comparison,
        oracle,
        task_column=config.task_column,
        target_column=config.target_column,
    ) - _regret_auc(
        variantshift,
        oracle,
        task_column=config.task_column,
        target_column=config.target_column,
    )
    summary = {
        "comparator": comparator,
        "primary_endpoint": "regret_coverage_auc",
        "regret_coverage_auc_improvement": regret_point,
        "regret_ci_low": float(np.quantile(regret_estimates, 0.025)),
        "regret_ci_high": float(np.quantile(regret_estimates, 0.975)),
        "regret_probability_improvement_above_zero": float(
            np.mean(regret_estimates > 0)
        ),
        "risk_coverage_auc_improvement": risk_point,
        "risk_ci_low": float(np.quantile(failure_estimates, 0.025)),
        "risk_ci_high": float(np.quantile(failure_estimates, 0.975)),
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


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a named family of p-values without an optional stats dependency."""
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _at_half(summary: pd.DataFrame, policy: str) -> pd.Series:
    rows = summary.loc[summary["policy"].eq(policy)]
    if len(rows) != 1:
        raise ValueError(f"Expected one policy summary for {policy}; found {len(rows)}")
    return rows.iloc[0]


def _relative_failure_reduction(baseline: float, candidate: float) -> float:
    if baseline > 0:
        return float((baseline - candidate) / baseline)
    return 0.0 if candidate <= 0 else -1.0


def confirmation_acceptance_gates(
    merged: pd.DataFrame,
    curves: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    config: TransportConfig,
    *,
    comparator: str,
    required_panels: tuple[str, ...],
    negative_conclusion: dict[str, object] | None = None,
) -> dict[str, object]:
    """Apply the preregistered scientific gates without adapting their thresholds."""
    summary = summarize_policy_curves(curves)
    candidate = _at_half(summary, "variantshift")
    baseline = _at_half(summary, comparator)
    primary = bootstrap_summary.loc[bootstrap_summary["comparator"].eq(comparator)]
    if len(primary) != 1:
        raise ValueError(f"Missing frozen comparator bootstrap: {comparator}")
    primary = primary.iloc[0]
    failure_reduction = _relative_failure_reduction(
        float(baseline["failure_rate_at_50pct"]),
        float(candidate["failure_rate_at_50pct"]),
    )
    marginal_coverage = float(
        np.mean(
            merged[config.target_column].to_numpy(dtype=float)
            >= merged["lower_selection_gain_sd"].to_numpy(dtype=float)
        )
    )
    selected = _choose_task_rows(
        merged,
        config,
        policy="variantshift",
        seed=config.seed,
        best_average_model=None,
    )
    selected_coverage = float(
        np.mean(
            selected[config.target_column].to_numpy(dtype=float)
            >= selected["lower_selection_gain_sd"].to_numpy(dtype=float)
        )
    )
    panels = panel_summary.set_index("panel_id", drop=False)
    missing_panels = sorted(set(required_panels).difference(panels.index))
    direction_consistent = not missing_panels and all(
        float(panels.loc[panel, "regret_coverage_auc_improvement"]) > 0
        for panel in required_panels
    )
    ablation_rows = bootstrap_summary.loc[
        bootstrap_summary["comparator"].isin(
            ["ablation:msa_only", "ablation:ensemble_only"]
        )
    ]
    expected_ablations = {"ablation:msa_only", "ablation:ensemble_only"}
    ablation_pass = set(ablation_rows["comparator"]) == expected_ablations and bool(
        (ablation_rows["regret_coverage_auc_improvement"] > 0).all()
    )
    conclusion_valid = bool(
        negative_conclusion
        and str(negative_conclusion.get("statement", "")).strip()
        and str(negative_conclusion.get("interpretation_change", "")).strip()
        and negative_conclusion.get("evidence_artifacts")
    )
    gate_rows = [
        {
            "gate": "primary_regret_coverage",
            "passed": bool(
                float(primary["regret_ci_low"]) > 0
                and float(primary["holm_adjusted_p_value"]) < 0.05
            ),
            "value": float(primary["regret_coverage_auc_improvement"]),
            "criterion": "family-bootstrap 95% CI excludes zero; Holm-adjusted p < 0.05",
        },
        {
            "gate": "selective_utility_at_50pct",
            "passed": bool(
                float(candidate["mean_selection_regret_at_50pct"])
                < float(baseline["mean_selection_regret_at_50pct"])
                and float(candidate["mean_selection_gain_at_50pct"])
                >= float(baseline["mean_selection_gain_at_50pct"])
                and float(candidate["failure_rate_at_50pct"])
                <= float(baseline["failure_rate_at_50pct"])
            ),
            "value": float(candidate["mean_selection_regret_at_50pct"]),
            "failure_reduction": failure_reduction,
            "criterion": (
                "lower regret, no lower mean gain, and no higher failure rate at 50% coverage"
            ),
        },
        {
            "gate": "nominal_90pct_coverage",
            "passed": bool(0.85 <= selected_coverage <= 0.95),
            "value": selected_coverage,
            "criterion": "selected-policy lower-bound coverage in [0.85, 0.95]",
        },
        {
            "gate": "primary_panel_direction_consistency",
            "passed": direction_consistent,
            "value": None if missing_panels else float(
                min(
                    panels.loc[panel, "regret_coverage_auc_improvement"]
                    for panel in required_panels
                )
            ),
            "criterion": "positive regret-coverage improvement in Domainome and MaveDB",
        },
        {
            "gate": "feature_ablation",
            "passed": ablation_pass,
            "value": None if len(ablation_rows) != 2 else float(
                ablation_rows["regret_coverage_auc_improvement"].min()
            ),
            "criterion": "full method beats MSA-only and ensemble-only selectors",
        },
        {
            "gate": "useful_negative_conclusion",
            "passed": conclusion_valid,
            "value": None,
            "criterion": "statement, interpretation change, and evidence artifacts documented",
        },
    ]
    all_automatic = all(
        bool(row["passed"])
        for row in gate_rows
        if row["gate"] != "useful_negative_conclusion"
    )
    return {
        "schema_version": 2,
        "status": "pass" if all(bool(row["passed"]) for row in gate_rows) else "fail",
        "all_automatic_gates_pass": all_automatic,
        "frozen_comparator": comparator,
        "required_primary_panels": list(required_panels),
        "missing_primary_panels": missing_panels,
        "marginal_lower_bound_coverage": marginal_coverage,
        "selected_policy_lower_bound_coverage": selected_coverage,
        "gates": gate_rows,
        "negative_conclusion": negative_conclusion,
    }


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
    TRANSPORT_FEATURE_SCHEMA.validate(frame)
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
        "decision_only",
        "uncalibrated",
        "elastic_net",
        "always_best",
        "random",
        "oracle",
    ]
    for optional_policy, column in (
        ("msa_depth", "msa_neff"),
        ("score_dispersion", "score_dispersion"),
        ("ensemble_agreement", "ensemble_agreement"),
    ):
        if column in crossfit:
            policies.append(optional_policy)
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
    RISK_COVERAGE_SCHEMA.validate(curves)
    summary = summarize_policy_curves(curves)
    comparator_candidates = summary.loc[
        summary["policy"].isin(
            [
                "uncalibrated",
                "decision_only",
                "elastic_net",
                "msa_depth",
                "score_dispersion",
                "ensemble_agreement",
            ]
        )
    ]
    best_comparator = str(
        comparator_candidates.sort_values("regret_coverage_auc").iloc[0]["policy"]
    )
    bundle["best_label_free_comparator"] = best_comparator
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
        "schema_version": 2,
        "name": "VariantShift Selective Transport Auditor",
        "config": asdict(config),
        "parameters": bundle["parameters"],
        "conformal_quantile": bundle["conformal_quantile"],
        "decision_rule": (
            "override the best-average model only when nested-CV shrunken expected gain "
            "exceeds the frozen margin"
        ),
        "decision_shrinkage": bundle["decision_shrinkage"],
        "shrinkage_candidates": list(DECISION_SHRINKAGE_CANDIDATES),
        "override_margin_candidates": list(OVERRIDE_MARGIN_CANDIDATES),
        "priority_weights": PRIORITY_WEIGHTS,
        "priority_inputs": ["msa_neff", "protein_length", "decision_score"],
        "auditor_policy": bundle["auditor_policy"],
        "conformal_mode": (
            "selection-aware, locally scaled hierarchical conformal calibration; "
            "equal family mass and equal within-family task mass"
        ),
        "error_scale_floor": bundle["scale_floor"],
        "best_average_model": best_model,
        "best_label_free_comparator": best_comparator,
        "frozen_feature_ablations": sorted(bundle["ablations"]),
        "confirmation_primary_panels": list(CONFIRMATION_PRIMARY_PANELS),
        "training_frame_sha256": bundle["training_frame_sha256"],
        "formal_scope": (
            "one-sided marginal coverage for a new task from a new family under "
            "two-level hierarchical exchangeability"
        ),
        "primary_failure": "selection_gain_sd <= 0",
        "primary_endpoint": "task-level selection-regret coverage AUC",
        "secondary_endpoints": [
            "failure risk-coverage AUC",
            "mean selection gain",
            "worst-quintile selection gain",
            "selected-policy lower-bound coverage",
        ],
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
    *,
    negative_conclusion_path: Path | None = None,
) -> dict[str, Path]:
    bundle = joblib.load(bundle_path)
    config = TransportConfig.from_dict(dict(bundle["config"]))
    predictions = pd.read_csv(frozen_predictions_path)
    outcomes = pd.read_csv(outcomes_path)
    TASK_METRIC_SCHEMA.validate(outcomes)
    join_columns = [
        "protocol_id",
        "panel_id",
        "dataset_id",
        config.assay_column,
        "target_id",
        config.model_column,
        config.group_column,
        config.protein_column,
    ]
    missing_identifiers = sorted(
        set(join_columns).difference(predictions.columns)
        | set(join_columns).difference(outcomes.columns)
    )
    if missing_identifiers:
        raise ValueError(
            f"Confirmation predictions/outcomes are missing identity columns: "
            f"{missing_identifiers}"
        )
    if predictions.duplicated(join_columns).any() or outcomes.duplicated(join_columns).any():
        raise ValueError("Confirmation task-model identity columns must be unique")
    outcome_columns = [
        column for column in outcomes.columns if column not in predictions.columns
    ]
    merged = predictions.merge(
        outcomes.loc[:, [*join_columns, *outcome_columns]],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Frozen predictions and confirmation outcomes have no shared tasks")
    best_model = str(bundle["best_average_model"])
    policies = [
        "variantshift",
        "decision_only",
        "uncalibrated",
        "always_best",
        "random",
        "oracle",
    ]
    if "elastic_predicted_selection_gain_sd" in merged:
        policies.append("elastic_net")
    for optional_policy, column in (
        ("msa_depth", "msa_neff"),
        ("score_dispersion", "score_dispersion"),
        ("ensemble_agreement", "ensemble_agreement"),
    ):
        if column in merged:
            policies.append(optional_policy)
    ablation_names = sorted(dict(bundle.get("ablations", {})))
    policies.extend(f"ablation:{name}" for name in ablation_names)
    comparator = str(bundle.get("best_label_free_comparator", ""))
    if not comparator:
        raise ValueError("Frozen bundle does not declare the development-selected comparator")
    if comparator not in policies:
        raise ValueError(
            f"Frozen comparator {comparator} is unavailable in confirmation predictions"
        )
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
    RISK_COVERAGE_SCHEMA.validate(curves)
    summary = summarize_policy_curves(curves)
    comparator_policies = [
        policy for policy in policies if policy not in {"variantshift", "oracle", "random"}
    ]
    bootstrap_frames = []
    bootstrap_rows = []
    for policy in comparator_policies:
        replicates, row = bootstrap_policy_difference(
            merged,
            config,
            comparator=policy,
            best_average_model=best_model,
        )
        one_sided_p = float(
            (1 + np.sum(replicates["regret_coverage_auc_improvement"] <= 0))
            / (len(replicates) + 1)
        )
        row["one_sided_p_value"] = one_sided_p
        bootstrap_frames.append(replicates)
        bootstrap_rows.append(row)
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    adjusted = _holm_adjust(
        dict(
            zip(
                bootstrap_summary["comparator"].astype(str),
                bootstrap_summary["one_sided_p_value"].astype(float),
                strict=True,
            )
        )
    )
    bootstrap_summary["holm_adjusted_p_value"] = bootstrap_summary["comparator"].map(
        adjusted
    )
    panel_rows = []
    for panel_id, panel in merged.groupby("panel_id", sort=True):
        candidate = _choose_task_rows(
            panel,
            config,
            policy="variantshift",
            seed=config.seed,
            best_average_model=best_model,
        )
        baseline = _choose_task_rows(
            panel,
            config,
            policy=comparator,
            seed=config.seed,
            best_average_model=best_model,
        )
        panel_oracle = panel.groupby(config.task_column)[config.target_column].max()
        candidate_regret = _regret_auc(
            candidate,
            panel_oracle,
            task_column=config.task_column,
            target_column=config.target_column,
        )
        baseline_regret = _regret_auc(
            baseline,
            panel_oracle,
            task_column=config.task_column,
            target_column=config.target_column,
        )
        panel_rows.append(
            {
                "panel_id": panel_id,
                "tasks": int(candidate[config.task_column].nunique()),
                "variantshift_risk_coverage_auc": _risk_auc(
                    candidate, config.target_column
                ),
                "comparator": comparator,
                "comparator_risk_coverage_auc": _risk_auc(
                    baseline, config.target_column
                ),
                "risk_coverage_auc_improvement": _risk_auc(
                    baseline, config.target_column
                )
                - _risk_auc(candidate, config.target_column),
                "variantshift_regret_coverage_auc": candidate_regret,
                "comparator_regret_coverage_auc": baseline_regret,
                "regret_coverage_auc_improvement": (
                    baseline_regret - candidate_regret
                ),
            }
        )
    panel_summary = pd.DataFrame(panel_rows)
    negative_conclusion = (
        json.loads(Path(negative_conclusion_path).read_text(encoding="utf-8"))
        if negative_conclusion_path is not None
        else None
    )
    acceptance = confirmation_acceptance_gates(
        merged,
        curves,
        bootstrap_summary,
        panel_summary,
        config,
        comparator=comparator,
        required_panels=tuple(
            map(str, bundle.get("confirmation_primary_panels", CONFIRMATION_PRIMARY_PANELS))
        ),
        negative_conclusion=negative_conclusion,
    )
    summary["observed_lower_bound_coverage"] = acceptance[
        "marginal_lower_bound_coverage"
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "merged": output_dir / "confirmation-task-predictions.csv",
        "curves": output_dir / "confirmation-risk-coverage.csv",
        "summary": output_dir / "confirmation-transport-summary.csv",
        "bootstrap": output_dir / "confirmation-policy-bootstrap.csv.gz",
        "bootstrap_summary": output_dir / "confirmation-policy-bootstrap-summary.csv",
        "panel_summary": output_dir / "confirmation-panel-summary.csv",
        "acceptance": output_dir / "confirmation-acceptance.json",
    }
    write_table(merged, outputs["merged"])
    write_table(curves, outputs["curves"])
    write_table(summary, outputs["summary"])
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        outputs["bootstrap"], index=False, compression="gzip"
    )
    write_table(bootstrap_summary, outputs["bootstrap_summary"])
    write_table(panel_summary, outputs["panel_summary"])
    outputs["acceptance"].write_text(
        json.dumps(acceptance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return outputs
