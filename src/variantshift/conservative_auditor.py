"""Conservative outcome-free reliability auditing around a fixed best model.

Version 2 deliberately removes model overrides. The best development model is always deployed;
two separately trained, outcome-free regressors rank tasks for abstention by predicted utility and
predicted model-selection regret. One interpretable ranking rule is fixed after development, while
alternative rules and safety checks are retained as an audit inside family-held-out folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .metrics import regression_metrics, top_selection_metrics
from .outcome_lock import assert_evaluation_artifacts_locked
from .provenance import sha256_file
from .schemas import OUTCOME_SCHEMA, TASK_METRIC_SCHEMA, write_table

IDENTITY_COLUMNS = [
    "protocol_id",
    "panel_id",
    "dataset_id",
    "assay_id",
    "target_id",
    "protein_id",
    "family_id",
    "model_id",
]

FORBIDDEN_CONFIRMATION_COLUMNS = {
    "effect",
    "direction",
    "selection_gain_sd",
    "best_variant_regret_sd",
    "spearman",
    "top_recall",
    "ndcg",
}
PERCENTILE_DECIMALS = 12


@dataclass(frozen=True)
class AuditorConfig:
    payload: dict[str, object]

    @property
    def baseline_model(self) -> str:
        return str(self.payload["baseline_model"])

    @property
    def model_ids(self) -> list[str]:
        return list(map(str, self.payload["model_ids"]))

    @property
    def coverage_grid(self) -> np.ndarray:
        return np.asarray(self.payload["coverage_grid"], dtype=float)

    @property
    def estimator(self) -> dict[str, object]:
        return dict(self.payload["estimator"])

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])


def load_auditor_config(path: Path) -> AuditorConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "baseline_model",
        "model_ids",
        "allowed_development_protocols",
        "allowed_development_panels",
        "outer_folds",
        "inner_folds",
        "bootstrap_repeats",
        "coverage_grid",
        "gain_numeric_features",
        "gain_categorical_features",
        "score_shape_features",
        "candidate_policies",
        "selected_policy",
        "inner_feasibility",
        "development_gates",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Conservative-auditor config is incomplete: {missing}")
    models = list(map(str, payload["model_ids"]))
    if str(payload["baseline_model"]) not in models:
        raise ValueError("The fixed baseline model must belong to the auditor model panel")
    if str(payload["selected_policy"]) not in set(map(str, payload["candidate_policies"])):
        raise ValueError("The selected policy must belong to the audited candidate set")
    coverage = np.asarray(payload["coverage_grid"], dtype=float)
    if len(coverage) < 2 or not np.all(np.diff(coverage) > 0):
        raise ValueError("Coverage grid must be strictly increasing")
    return AuditorConfig(payload)


def _task_identifier(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["protocol_id"].astype(str)
        + "::"
        + frame["panel_id"].astype(str)
        + "::"
        + frame["assay_id"].astype(str)
        + "::"
        + frame["target_id"].astype(str)
    )


def build_combined_development_frame(
    development_features_path: Path,
    pilot_features_path: Path,
    pilot_metrics_path: Path,
    config: AuditorConfig,
) -> pd.DataFrame:
    """Join the revealed pilot to ProteinGym development without touching any holdout."""
    development = pd.read_csv(development_features_path)
    pilot_features = pd.read_csv(pilot_features_path)
    pilot_metrics = pd.read_csv(pilot_metrics_path)
    allowed_protocols = set(map(str, config.payload["allowed_development_protocols"]))
    allowed_panels = set(map(str, config.payload["allowed_development_panels"]))
    for label, source in (
        ("development features", development),
        ("pilot features", pilot_features),
        ("pilot metrics", pilot_metrics),
    ):
        observed_protocols = set(source["protocol_id"].astype(str))
        observed_panels = set(source["panel_id"].astype(str))
        if not observed_protocols.issubset(allowed_protocols):
            raise ValueError(
                f"{label} contain a non-development protocol: "
                f"{sorted(observed_protocols - allowed_protocols)}"
            )
        if not observed_panels.issubset(allowed_panels):
            raise ValueError(
                f"{label} contain a non-development panel: "
                f"{sorted(observed_panels - allowed_panels)}"
            )
    pilot = pilot_features.merge(
        pilot_metrics.loc[:, [*IDENTITY_COLUMNS, "selection_gain_sd"]],
        on=IDENTITY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    combined = pd.concat([development, pilot], ignore_index=True, sort=False)
    combined["auditor_task_id"] = _task_identifier(combined)
    combined = combined.loc[combined["model_id"].astype(str).isin(config.model_ids)].copy()
    model_counts = combined.groupby("auditor_task_id")["model_id"].nunique()
    complete_tasks = set(model_counts.loc[model_counts.eq(len(config.model_ids))].index)
    combined = combined.loc[combined["auditor_task_id"].isin(complete_tasks)].copy()
    if set(combined["model_id"].astype(str)) != set(config.model_ids):
        raise ValueError("Combined development data omit a configured model")
    if combined.duplicated(["auditor_task_id", "model_id"]).any():
        raise ValueError("Combined development task-model identities are not unique")

    oracle = combined.groupby("auditor_task_id")["selection_gain_sd"].max()
    baseline = combined.loc[
        combined["model_id"].astype(str).eq(config.baseline_model)
    ].copy()
    baseline["oracle_selection_gain_sd"] = baseline["auditor_task_id"].map(oracle)
    baseline["selection_regret_sd"] = (
        baseline["oracle_selection_gain_sd"] - baseline["selection_gain_sd"]
    )
    shape_features = list(map(str, config.payload["score_shape_features"]))
    wide = combined.pivot_table(
        index="auditor_task_id",
        columns="model_id",
        values=shape_features,
        aggfunc="first",
    )
    wide.columns = [f"{feature}__{model}" for feature, model in wide.columns]
    baseline = baseline.merge(
        wide.reset_index(), on="auditor_task_id", how="left", validate="one_to_one"
    )
    baseline["assay_modality_normalized"] = (
        baseline["assay_modality"].astype(str).str.strip().str.lower()
    )
    return baseline.sort_values("auditor_task_id").reset_index(drop=True)


def build_outcome_free_task_frame(
    feature_rows: pd.DataFrame, config: AuditorConfig
) -> pd.DataFrame:
    """Collapse a task-model feature panel without accepting experimental outcomes."""
    forbidden = sorted(FORBIDDEN_CONFIRMATION_COLUMNS.intersection(feature_rows.columns))
    if forbidden:
        raise ValueError(f"Outcome-free auditor input contains forbidden columns: {forbidden}")
    required = {
        "protocol_id",
        "panel_id",
        "dataset_id",
        "assay_id",
        "target_id",
        "protein_id",
        "family_id",
        "model_id",
        *map(str, config.payload["gain_numeric_features"]),
        *map(str, config.payload["gain_categorical_features"]),
        *map(str, config.payload["score_shape_features"]),
    }
    missing = sorted(required.difference(feature_rows.columns))
    if missing:
        raise ValueError(f"Outcome-free auditor input is incomplete: {missing}")
    frame = feature_rows.loc[
        feature_rows["model_id"].astype(str).isin(config.model_ids)
    ].copy()
    frame["auditor_task_id"] = _task_identifier(frame)
    counts = frame.groupby("auditor_task_id")["model_id"].nunique()
    incomplete = counts.loc[~counts.eq(len(config.model_ids))]
    if not incomplete.empty:
        raise ValueError(
            f"Outcome-free auditor input has {len(incomplete)} incomplete model tasks"
        )
    if frame.duplicated(["auditor_task_id", "model_id"]).any():
        raise ValueError("Outcome-free task-model identities are not unique")
    shape_features = list(map(str, config.payload["score_shape_features"]))
    wide = frame.pivot_table(
        index="auditor_task_id",
        columns="model_id",
        values=shape_features,
        aggfunc="first",
    )
    wide.columns = [f"{feature}__{model}" for feature, model in wide.columns]
    baseline = frame.loc[
        frame["model_id"].astype(str).eq(config.baseline_model)
    ].merge(wide.reset_index(), on="auditor_task_id", how="left", validate="one_to_one")
    _shape_columns(baseline, config)
    return baseline.sort_values("auditor_task_id").reset_index(drop=True)


def _shape_columns(frame: pd.DataFrame, config: AuditorConfig) -> list[str]:
    prefixes = tuple(
        f"{feature}__" for feature in map(str, config.payload["score_shape_features"])
    )
    columns = sorted(column for column in frame.columns if column.startswith(prefixes))
    expected = len(config.model_ids) * len(prefixes)
    if len(columns) != expected:
        raise ValueError(f"Expected {expected} task-wide score-shape columns, found {len(columns)}")
    return columns


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric)
    ]
    if categorical:
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
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _component_specifications(
    frame: pd.DataFrame, config: AuditorConfig
) -> dict[str, tuple[str, list[str], list[str]]]:
    shapes = _shape_columns(frame, config)
    metadata = list(map(str, config.payload["gain_numeric_features"]))
    categorical = list(map(str, config.payload["gain_categorical_features"]))
    return {
        "regret": ("selection_regret_sd", shapes, []),
        "gain_meta": ("selection_gain_sd", metadata, categorical),
        "gain_all": ("selection_gain_sd", [*metadata, *shapes], categorical),
        "gain_shape": ("selection_gain_sd", shapes, []),
    }


def _panel_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    inverse = (1.0 / frame["panel_id"].value_counts()).to_dict()
    weights = frame["panel_id"].map(inverse).to_numpy(dtype=float)
    return weights / np.mean(weights)


def _fit_components(
    frame: pd.DataFrame,
    config: AuditorConfig,
    *,
    seed: int,
    trees: int,
) -> dict[str, Pipeline]:
    estimator = config.estimator
    weights = _panel_balanced_weights(frame)
    models = {}
    for name, (target, numeric, categorical) in _component_specifications(
        frame, config
    ).items():
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=int(estimator["min_samples_leaf"]),
            max_features=float(estimator["max_features"]),
            random_state=seed + sum(map(ord, name)),
            n_jobs=-1,
        )
        pipeline = Pipeline(
            [("preprocess", _preprocessor(numeric, categorical)), ("model", model)]
        )
        pipeline.fit(frame, frame[target], model__sample_weight=weights)
        models[name] = pipeline
    return models


def _component_predictions(
    models: dict[str, Pipeline], frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    return {name: model.predict(frame) for name, model in models.items()}


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(
        np.round(np.asarray(reference, dtype=float), decimals=PERCENTILE_DECIMALS)
    )
    if len(reference) == 0:
        raise ValueError("Percentile reference cannot be empty")
    values = np.round(np.asarray(values, dtype=float), decimals=PERCENTILE_DECIMALS)
    return np.searchsorted(reference, values, side="right") / len(reference)


def candidate_scores(
    predictions: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return every preregistered abstention score from outcome-free predictions."""
    safety = 1.0 - _percentile(reference["regret"], predictions["regret"])
    output = {"pure_regret": safety}
    for name in ("gain_meta", "gain_all", "gain_shape"):
        gain = _percentile(reference[name], predictions[name])
        output[f"harm_{name}"] = 2.0 * gain * safety / (gain + safety + 1e-12)
    gain_mean = float(np.mean(reference["gain_all"]))
    gain_sd = max(float(np.std(reference["gain_all"])), 1e-9)
    regret_mean = float(np.mean(reference["regret"]))
    regret_sd = max(float(np.std(reference["regret"])), 1e-9)
    standardized_gain = (predictions["gain_all"] - gain_mean) / gain_sd
    standardized_regret = (predictions["regret"] - regret_mean) / regret_sd
    for weight in (0.5, 1.0, 1.5, 2.0):
        output[f"blend_{weight}"] = standardized_gain - weight * standardized_regret
    return output


def _policy_statistics(
    frame: pd.DataFrame,
    confidence: np.ndarray,
    coverage_grid: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    regret = []
    risk = []
    utility = []
    retained = []
    for coverage in coverage_grid:
        count = max(1, int(np.ceil(len(order) * coverage)))
        selected = frame.iloc[order[:count]]
        regret.append(float(selected["selection_regret_sd"].mean()))
        risk.append(float((selected["selection_gain_sd"] <= 0).mean()))
        utility.append(float(selected["selection_gain_sd"].mean()))
        retained.append(count)
    half = int(np.argmin(np.abs(coverage_grid - 0.5)))
    return {
        "regret_coverage_auc": float(trapezoid(regret, coverage_grid)),
        "risk_coverage_auc": float(trapezoid(risk, coverage_grid)),
        "utility_coverage_auc": float(trapezoid(utility, coverage_grid)),
        "mean_regret_at_50pct": regret[half],
        "failure_rate_at_50pct": risk[half],
        "mean_gain_at_50pct": utility[half],
        "retained_at_50pct": retained[half],
    }


def _expected_fixed_baseline_statistics(
    frame: pd.DataFrame, coverage_grid: np.ndarray
) -> dict[str, float]:
    width = float(coverage_grid[-1] - coverage_grid[0])
    return {
        "regret_coverage_auc": width * float(frame["selection_regret_sd"].mean()),
        "risk_coverage_auc": width * float((frame["selection_gain_sd"] <= 0).mean()),
        "utility_coverage_auc": width * float(frame["selection_gain_sd"].mean()),
        "mean_regret_at_50pct": float(frame["selection_regret_sd"].mean()),
        "failure_rate_at_50pct": float((frame["selection_gain_sd"] <= 0).mean()),
        "mean_gain_at_50pct": float(frame["selection_gain_sd"].mean()),
        "retained_at_50pct": max(1, int(np.ceil(len(frame) * 0.5))),
    }


def _improvements(
    frame: pd.DataFrame,
    confidence: np.ndarray,
    coverage_grid: np.ndarray,
) -> dict[str, float]:
    auditor = _policy_statistics(frame, confidence, coverage_grid)
    baseline = _expected_fixed_baseline_statistics(frame, coverage_grid)
    return {
        "regret_auc_improvement": baseline["regret_coverage_auc"]
        - auditor["regret_coverage_auc"],
        "risk_auc_improvement": baseline["risk_coverage_auc"]
        - auditor["risk_coverage_auc"],
        "utility_auc_improvement": auditor["utility_coverage_auc"]
        - baseline["utility_coverage_auc"],
        "regret_improvement_at_50pct": baseline["mean_regret_at_50pct"]
        - auditor["mean_regret_at_50pct"],
        "risk_improvement_at_50pct": baseline["failure_rate_at_50pct"]
        - auditor["failure_rate_at_50pct"],
        "mean_gain_change_at_50pct": auditor["mean_gain_at_50pct"]
        - baseline["mean_gain_at_50pct"],
    }


def _inner_select_policy(
    frame: pd.DataFrame,
    config: AuditorConfig,
    *,
    seed: int,
) -> tuple[str, pd.DataFrame]:
    groups = frame["family_id"].astype(str).to_numpy()
    splits = min(int(config.payload["inner_folds"]), len(np.unique(groups)))
    if splits < 3:
        raise ValueError("Nested policy selection requires at least three family groups")
    predictions = {
        name: np.full(len(frame), np.nan, dtype=float)
        for name in _component_specifications(frame, config)
    }
    splitter = GroupKFold(n_splits=splits)
    for fold, (train, validation) in enumerate(splitter.split(frame, groups=groups)):
        models = _fit_components(
            frame.iloc[train],
            config,
            seed=seed + fold * 100,
            trees=int(config.estimator["n_estimators_inner"]),
        )
        values = _component_predictions(models, frame.iloc[validation])
        for name, heldout in predictions.items():
            heldout[validation] = values[name]
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Inner family cross-fitting did not cover every task")
    scores = candidate_scores(predictions, predictions)
    feasibility = dict(config.payload["inner_feasibility"])
    rows = []
    allowed = set(map(str, config.payload["candidate_policies"]))
    for name, confidence in scores.items():
        if name not in allowed:
            continue
        metrics = _improvements(frame, confidence, config.coverage_grid)
        is_feasible = (
            metrics["risk_auc_improvement"]
            >= float(feasibility["minimum_risk_auc_improvement"])
            and metrics["mean_gain_change_at_50pct"]
            >= float(feasibility["minimum_mean_gain_change_at_50pct"])
        )
        rows.append({"candidate_policy": name, "feasible": is_feasible, **metrics})
    selected = str(config.payload["selected_policy"])
    selected_rows = [row for row in rows if row["candidate_policy"] == selected]
    if not selected_rows:
        raise RuntimeError(f"Fixed candidate {selected} was not audited")
    return selected, pd.DataFrame(rows).sort_values("candidate_policy")


def cross_fit_conservative_auditor(
    frame: pd.DataFrame, config: AuditorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame["family_id"].astype(str).to_numpy()
    splits = min(int(config.payload["outer_folds"]), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=splits)
    output = frame.copy()
    for column in [
        "predicted_gain",
        "predicted_regret",
        "gain_percentile",
        "safety_percentile",
        "auditor_confidence",
    ]:
        output[column] = np.nan
    output["selected_candidate_policy"] = ""
    fold_rows = []
    candidate_rows = []
    for fold, (train, test) in enumerate(splitter.split(frame, groups=groups)):
        training = frame.iloc[train].reset_index(drop=True)
        testing = frame.iloc[test].reset_index(drop=True)
        selected, audit = _inner_select_policy(
            training, config, seed=config.seed + fold * 10_000
        )
        audit["outer_fold"] = fold
        candidate_rows.append(audit)
        models = _fit_components(
            training,
            config,
            seed=config.seed + 100_000 + fold,
            trees=int(config.estimator["n_estimators_final"]),
        )
        reference = _component_predictions(models, training)
        predictions = _component_predictions(models, testing)
        scores = candidate_scores(predictions, reference)
        confidence = scores[selected]
        gain_percentile = _percentile(reference["gain_meta"], predictions["gain_meta"])
        safety_percentile = 1.0 - _percentile(
            reference["regret"], predictions["regret"]
        )
        output.loc[test, "predicted_gain"] = predictions["gain_meta"]
        output.loc[test, "predicted_regret"] = predictions["regret"]
        output.loc[test, "gain_percentile"] = gain_percentile
        output.loc[test, "safety_percentile"] = safety_percentile
        output.loc[test, "auditor_confidence"] = confidence
        output.loc[test, "selected_candidate_policy"] = selected
        fold_rows.append(
            {
                "outer_fold": fold,
                "training_tasks": len(train),
                "test_tasks": len(test),
                "training_families": int(training["family_id"].nunique()),
                "test_families": int(testing["family_id"].nunique()),
                "family_overlap": bool(
                    set(training["family_id"]).intersection(testing["family_id"])
                ),
                "selected_candidate_policy": selected,
            }
        )
    if output["auditor_confidence"].isna().any():
        raise RuntimeError("Outer family cross-fitting did not cover every task")
    return output, pd.DataFrame(fold_rows), pd.concat(candidate_rows, ignore_index=True)


def _curves(
    frame: pd.DataFrame, confidence: np.ndarray, config: AuditorConfig
) -> pd.DataFrame:
    rows = []
    order = np.argsort(-np.asarray(confidence), kind="stable")
    for coverage in config.coverage_grid:
        count = max(1, int(np.ceil(len(order) * coverage)))
        selected = frame.iloc[order[:count]]
        for policy, values in (
            ("conservative_auditor_v2", selected),
            ("always_vespag_expected_random_abstention", frame),
        ):
            rows.append(
                {
                    "policy": policy,
                    "coverage": float(coverage),
                    "retained_tasks": count,
                    "failure_rate": float((values["selection_gain_sd"] <= 0).mean()),
                    "mean_selection_regret_sd": float(values["selection_regret_sd"].mean()),
                    "mean_selection_gain_sd": float(values["selection_gain_sd"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _hierarchy(frame: pd.DataFrame) -> list[list[np.ndarray]]:
    return [
        [protein.index.to_numpy() for _, protein in family.groupby("protein_id", sort=True)]
        for _, family in frame.groupby("family_id", sort=True)
    ]


def bootstrap_improvements(
    frame: pd.DataFrame, config: AuditorConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    hierarchy = _hierarchy(frame)
    repeats = int(config.payload["bootstrap_repeats"])
    rng = np.random.default_rng(config.seed)
    rows = []
    for repeat in range(repeats):
        indices: list[int] = []
        for family_index in rng.integers(0, len(hierarchy), len(hierarchy)):
            proteins = hierarchy[family_index]
            for protein_index in rng.integers(0, len(proteins), len(proteins)):
                tasks = proteins[protein_index]
                indices.extend(tasks[rng.integers(0, len(tasks), len(tasks))])
        sampled = frame.loc[indices].reset_index(drop=True)
        rows.append(
            {
                "repeat": repeat,
                **_improvements(
                    sampled,
                    sampled["auditor_confidence"].to_numpy(dtype=float),
                    config.coverage_grid,
                ),
            }
        )
    replicates = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "family, then protein, then assay",
    }
    for column in [name for name in replicates if name != "repeat"]:
        values = replicates[column].to_numpy(dtype=float)
        summary[f"{column}_point"] = float(
            _improvements(
                frame,
                frame["auditor_confidence"].to_numpy(dtype=float),
                config.coverage_grid,
            )[column]
        )
        summary[f"{column}_ci_low"] = float(np.quantile(values, 0.025))
        summary[f"{column}_ci_high"] = float(np.quantile(values, 0.975))
        summary[f"{column}_probability_above_zero"] = float(np.mean(values > 0))
    return replicates, summary


def leave_one_panel_out(
    frame: pd.DataFrame, config: AuditorConfig
) -> pd.DataFrame:
    rows = []
    for panel_index, panel_id in enumerate(sorted(frame["panel_id"].astype(str).unique())):
        training = frame.loc[~frame["panel_id"].astype(str).eq(panel_id)].reset_index(drop=True)
        testing = frame.loc[frame["panel_id"].astype(str).eq(panel_id)].reset_index(drop=True)
        selected, _audit = _inner_select_policy(
            training, config, seed=config.seed + 500_000 + panel_index * 10_000
        )
        models = _fit_components(
            training,
            config,
            seed=config.seed + 600_000 + panel_index,
            trees=int(config.estimator["n_estimators_final"]),
        )
        reference = _component_predictions(models, training)
        predictions = _component_predictions(models, testing)
        confidence = candidate_scores(predictions, reference)[selected]
        rows.append(
            {
                "held_out_panel": panel_id,
                "training_tasks": len(training),
                "test_tasks": len(testing),
                "status": "evaluated",
                "reason": "",
                "selected_candidate_policy": selected,
                **_improvements(testing, confidence, config.coverage_grid),
            }
        )
    return pd.DataFrame(rows)


def _fit_frozen_bundle(
    frame: pd.DataFrame, config: AuditorConfig
) -> tuple[dict[str, object], pd.DataFrame]:
    selected, audit = _inner_select_policy(
        frame, config, seed=config.seed + 700_000
    )
    models = _fit_components(
        frame,
        config,
        seed=config.seed + 800_000,
        trees=int(config.estimator["n_estimators_final"]),
    )
    reference = _component_predictions(models, frame)
    confidence = candidate_scores(reference, reference)[selected]
    bundle = {
        "schema_version": 1,
        "method_id": config.payload["method_id"],
        "config": config.payload,
        "selected_candidate_policy": selected,
        "baseline_model": config.baseline_model,
        "component_models": models,
        "reference_predictions": reference,
        "training_confidence_threshold_50pct": float(np.quantile(confidence, 0.5)),
        "training_task_ids": frame["auditor_task_id"].astype(str).tolist(),
    }
    return bundle, audit


def fit_conservative_auditor(
    development_features_path: Path,
    pilot_features_path: Path,
    pilot_metrics_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Fit, evaluate, and freeze the conservative auditor using development outcomes only."""
    config = load_auditor_config(config_path)
    frame = build_combined_development_frame(
        development_features_path, pilot_features_path, pilot_metrics_path, config
    )
    crossfit, folds, candidates = cross_fit_conservative_auditor(frame, config)
    curves = _curves(crossfit, crossfit["auditor_confidence"].to_numpy(), config)
    replicates, bootstrap_summary = bootstrap_improvements(crossfit, config)
    panels = []
    for panel_id, selected in crossfit.groupby("panel_id", sort=True):
        panels.append(
            {
                "panel_id": panel_id,
                "tasks": len(selected),
                **_improvements(
                    selected,
                    selected["auditor_confidence"].to_numpy(dtype=float),
                    config.coverage_grid,
                ),
            }
        )
    panel_summary = pd.DataFrame(panels)
    panel_holdout = leave_one_panel_out(frame, config)
    bundle, final_candidate_audit = _fit_frozen_bundle(frame, config)
    gates = dict(config.payload["development_gates"])
    family_gate = bool(
        bootstrap_summary["regret_auc_improvement_ci_low"] > 0
        and bootstrap_summary["risk_auc_improvement_ci_low"] >= 0
        and bootstrap_summary["mean_gain_change_at_50pct_point"] >= 0
    )
    cross_panel_gate = bool(
        panel_holdout["regret_auc_improvement"].ge(0).all()
        and panel_holdout["risk_auc_improvement"].ge(0).all()
        and panel_holdout["mean_gain_change_at_50pct"].ge(0).all()
    )
    gate_report = {
        "schema_version": 1,
        "method_id": config.payload["method_id"],
        "family_heldout_gate_passed": family_gate,
        "leave_one_panel_out_gate_passed": cross_panel_gate,
        "top_publication_evidence_ready": family_gate and cross_panel_gate,
        "safe_interpretation": (
            "Candidate is frozen for a future untouched test, but cross-panel transport remains "
            "unproven. Do not reveal confirmation outcomes unless the final protocol treats "
            "cross-panel failure as a decisive negative result."
        ),
        "configured_gates": gates,
        "bootstrap": bootstrap_summary,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "development": output_dir / "combined-development-tasks.csv",
        "crossfit": output_dir / "family-crossfit-predictions.csv",
        "folds": output_dir / "family-fold-audit.csv",
        "candidates": output_dir / "nested-candidate-audit.csv",
        "curves": output_dir / "risk-regret-coverage.csv",
        "bootstrap": output_dir / "hierarchical-bootstrap.csv.gz",
        "bootstrap_summary": output_dir / "hierarchical-bootstrap-summary.json",
        "panels": output_dir / "panel-summary.csv",
        "panel_holdout": output_dir / "leave-one-panel-out.csv",
        "final_candidates": output_dir / "final-nested-candidate-audit.csv",
        "bundle": output_dir / "conservative-auditor-v2.joblib",
        "method": output_dir / "conservative-auditor-v2.json",
        "gates": output_dir / "development-gates.json",
        "manifest": output_dir / "candidate-freeze-manifest.json",
    }
    write_table(frame, outputs["development"])
    write_table(crossfit, outputs["crossfit"])
    write_table(folds, outputs["folds"])
    write_table(candidates, outputs["candidates"])
    write_table(curves, outputs["curves"])
    replicates.to_csv(
        outputs["bootstrap"],
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    outputs["bootstrap_summary"].write_text(
        json.dumps(bootstrap_summary, indent=2, sort_keys=True) + "\n"
    )
    write_table(panel_summary, outputs["panels"])
    write_table(panel_holdout, outputs["panel_holdout"])
    write_table(final_candidate_audit, outputs["final_candidates"])
    joblib.dump(bundle, outputs["bundle"])
    method = {
        "schema_version": 1,
        "method_id": config.payload["method_id"],
        "baseline_model": config.baseline_model,
        "selected_candidate_policy": bundle["selected_candidate_policy"],
        "decision": "deploy VespaG or abstain; no model override",
        "confidence": (
            "fixed harmonic mean of family-cross-fitted predicted VespaG utility rank and "
            "predicted safety rank (one minus predicted regret percentile)"
        ),
        "training_tasks": len(frame),
        "training_families": int(frame["family_id"].nunique()),
        "training_panels": sorted(frame["panel_id"].astype(str).unique()),
        "development_only": True,
        "adaptive_analysis_disclosure": (
            "The external pilot was explicitly development data and informed the v2 redesign. "
            "Only a subsequent untouched evaluation can support confirmation claims."
        ),
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                development_features_path,
                pilot_features_path,
                pilot_metrics_path,
                config_path,
            ]
        },
    }
    outputs["method"].write_text(json.dumps(method, indent=2, sort_keys=True) + "\n")
    outputs["gates"].write_text(json.dumps(gate_report, indent=2, sort_keys=True) + "\n")
    frozen_artifacts = {
        key: sha256_file(path)
        for key, path in outputs.items()
        if key != "manifest" and path.exists()
    }
    manifest = {
        "schema_version": 1,
        "state": "development_candidate_frozen",
        "method_id": config.payload["method_id"],
        "development_only": True,
        "confirmation_outcomes_requested_by_this_run": False,
        "baseline_policy": "always deploy VespaG or abstain",
        "model_override_permitted": False,
        "implementation_sha256": sha256_file(Path(__file__)),
        "family_heldout_gate_passed": family_gate,
        "leave_one_panel_out_gate_passed": cross_panel_gate,
        "artifacts": frozen_artifacts,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def add_confirmation_coverage_decisions(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Freeze deterministic pooled and within-panel confidence ranks."""
    scored = scored.copy()
    tie_breakers = ["panel_id", "assay_id", "target_id"]
    pooled_order = scored.sort_values(
        ["auditor_confidence", *tie_breakers],
        ascending=[False, True, True, True],
        kind="stable",
    ).index
    scored["pooled_confidence_rank"] = 0
    scored.loc[pooled_order, "pooled_confidence_rank"] = np.arange(1, len(scored) + 1)
    scored["pooled_coverage_fraction"] = scored["pooled_confidence_rank"] / len(scored)
    pooled_count = max(1, int(np.ceil(len(scored) * 0.5)))
    scored["pooled_decision_at_50pct_coverage"] = np.where(
        scored["pooled_confidence_rank"] <= pooled_count, "deploy_vespag", "abstain"
    )
    scored["panel_confidence_rank"] = 0
    scored["panel_coverage_fraction"] = np.nan
    scored["panel_decision_at_50pct_coverage"] = ""
    panel_deploy_counts: dict[str, int] = {}
    for panel_id, panel in scored.groupby("panel_id", sort=True):
        panel_order = panel.sort_values(
            ["auditor_confidence", "assay_id", "target_id"],
            ascending=[False, True, True],
            kind="stable",
        ).index
        ranks = np.arange(1, len(panel_order) + 1)
        selected_count = max(1, int(np.ceil(len(panel_order) * 0.5)))
        scored.loc[panel_order, "panel_confidence_rank"] = ranks
        scored.loc[panel_order, "panel_coverage_fraction"] = ranks / len(panel_order)
        scored.loc[panel_order, "panel_decision_at_50pct_coverage"] = np.where(
            ranks <= selected_count, "deploy_vespag", "abstain"
        )
        panel_deploy_counts[str(panel_id)] = selected_count
    return scored, pooled_count, panel_deploy_counts


def score_conservative_auditor(
    features_path: Path,
    bundle_path: Path,
    output_path: Path,
) -> dict[str, Path]:
    """Apply a frozen auditor to outcome-free task-model features."""
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or "config" not in bundle:
        raise ValueError("Conservative-auditor bundle is invalid")
    config = AuditorConfig(dict(bundle["config"]))
    frame = build_outcome_free_task_frame(pd.read_csv(features_path), config)
    predictions = _component_predictions(bundle["component_models"], frame)
    reference = bundle["reference_predictions"]
    selected = str(bundle["selected_candidate_policy"])
    confidence = candidate_scores(predictions, reference)[selected]
    gain_percentile = _percentile(reference["gain_meta"], predictions["gain_meta"])
    safety_percentile = 1.0 - _percentile(reference["regret"], predictions["regret"])
    threshold = float(bundle["training_confidence_threshold_50pct"])
    identity = [
        "protocol_id",
        "panel_id",
        "dataset_id",
        "assay_id",
        "target_id",
        "protein_id",
        "family_id",
    ]
    scored = frame.loc[:, identity].copy()
    scored["baseline_model"] = config.baseline_model
    scored["selected_candidate_policy"] = selected
    scored["predicted_gain"] = predictions["gain_meta"]
    scored["predicted_regret"] = predictions["regret"]
    scored["gain_percentile"] = gain_percentile
    scored["safety_percentile"] = safety_percentile
    scored["auditor_confidence"] = confidence
    scored["development_50pct_threshold"] = threshold
    scored["decision_at_development_threshold"] = np.where(
        confidence >= threshold, "deploy_vespag", "abstain"
    )
    scored, pooled_count, panel_deploy_counts = add_confirmation_coverage_decisions(scored)
    output_path = write_table(scored, output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "schema_version": 1,
        "method_id": bundle["method_id"],
        "outcome_columns_accepted": False,
        "features_sha256": sha256_file(features_path),
        "bundle_sha256": sha256_file(bundle_path),
        "scores_sha256": sha256_file(output_path),
        "tasks": len(scored),
        "pooled_deploy_at_50pct_coverage": pooled_count,
        "panel_deploy_at_50pct_coverage": panel_deploy_counts,
        "baseline_model": config.baseline_model,
        "model_override_permitted": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"scores": output_path, "manifest": manifest_path}


CONFIRMATION_V2_PANELS = ("human-domainome-v1", "venusmuthub-v1")
CONFIRMATION_IDENTITY = [
    "protocol_id",
    "panel_id",
    "dataset_id",
    "assay_id",
    "target_id",
]


def build_conservative_confirmation_metrics(
    decisions: pd.DataFrame,
    prediction_registry: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: AuditorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align frozen predictions to revealed effects without fitting or changing decisions."""
    OUTCOME_SCHEMA.validate(outcomes)
    required_decisions = {
        *CONFIRMATION_IDENTITY,
        "protein_id",
        "family_id",
        "baseline_model",
        "auditor_confidence",
        "pooled_confidence_rank",
        "panel_confidence_rank",
    }
    missing_decisions = sorted(required_decisions.difference(decisions.columns))
    if missing_decisions:
        raise ValueError(f"Frozen decisions are incomplete: {missing_decisions}")
    if decisions.duplicated(CONFIRMATION_IDENTITY).any():
        raise ValueError("Frozen confirmation decisions are not task-unique")
    observed_panels = set(decisions["panel_id"].astype(str))
    if observed_panels != set(CONFIRMATION_V2_PANELS):
        raise ValueError(
            "Conservative Auditor v2 requires exactly Domainome and untouched Venus; "
            f"found {sorted(observed_panels)}"
        )
    if set(decisions["baseline_model"].astype(str)) != {config.baseline_model}:
        raise ValueError("Frozen decisions do not consistently deploy the configured baseline")
    if outcomes.duplicated(CONFIRMATION_IDENTITY + ["variant_id"]).any():
        raise ValueError("Confirmation outcomes contain duplicate task-variant identities")
    outcome_panels = set(outcomes["panel_id"].astype(str))
    unexpected_outcomes = sorted(outcome_panels.difference(CONFIRMATION_V2_PANELS))
    if unexpected_outcomes:
        raise ValueError(f"Confirmation outcomes contain undeclared panels: {unexpected_outcomes}")
    required_registry = {
        "panel_id",
        "model_id",
        "prediction_path",
        "prediction_sha256",
    }
    missing_registry = sorted(required_registry.difference(prediction_registry.columns))
    if missing_registry:
        raise ValueError(f"Frozen prediction registry is incomplete: {missing_registry}")
    if prediction_registry.duplicated(["panel_id", "model_id"]).any():
        raise ValueError("Frozen prediction registry is not panel-model unique")

    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    decision_columns = [
        *CONFIRMATION_IDENTITY,
        "protein_id",
        "family_id",
    ]
    for registry_row in prediction_registry.itertuples(index=False):
        panel_id = str(registry_row.panel_id)
        if panel_id not in CONFIRMATION_V2_PANELS:
            continue
        prediction_path = Path(registry_row.prediction_path)
        expected_digest = str(registry_row.prediction_sha256)
        if not prediction_path.is_file() or sha256_file(prediction_path) != expected_digest:
            raise ValueError(f"Frozen prediction artifact failed verification: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        required_prediction_columns = {"target_id", "variant_id", "score"}
        missing_predictions = sorted(required_prediction_columns.difference(predictions.columns))
        if missing_predictions:
            raise ValueError(
                f"Prediction artifact {prediction_path} is incomplete: {missing_predictions}"
            )
        if "status" in predictions:
            predictions = predictions.loc[predictions["status"].astype(str).eq("ok")]
        predictions = predictions.loc[:, ["target_id", "variant_id", "score"]].dropna()
        if predictions.duplicated(["target_id", "variant_id"]).any():
            raise ValueError(f"Prediction artifact has duplicate variants: {prediction_path}")
        panel_tasks = decisions.loc[
            decisions["panel_id"].astype(str).eq(panel_id), decision_columns
        ]
        panel_outcomes = outcomes.loc[
            outcomes["panel_id"].astype(str).eq(panel_id),
            [*CONFIRMATION_IDENTITY, "variant_id", "effect"],
        ]
        model_id = str(registry_row.model_id)
        for task in panel_tasks.itertuples(index=False):
            task_outcomes = panel_outcomes.loc[
                panel_outcomes["assay_id"].astype(str).eq(str(task.assay_id))
                & panel_outcomes["target_id"].astype(str).eq(str(task.target_id)),
                ["variant_id", "effect"],
            ]
            task_predictions = predictions.loc[
                predictions["target_id"].astype(str).eq(str(task.target_id)),
                ["variant_id", "score"],
            ]
            aligned = task_outcomes.merge(
                task_predictions,
                on="variant_id",
                how="inner",
                validate="one_to_one",
            ).dropna()
            outcome_variants = len(task_outcomes)
            aligned_fraction = (
                float(len(aligned) / outcome_variants) if outcome_variants else 0.0
            )
            if len(aligned) < 10:
                status = "too_few_aligned_variants"
            elif panel_id == "human-domainome-v1" and aligned_fraction < 0.95:
                status = "below_frozen_95pct_domainome_coverage"
            else:
                status = "included"
            identity = {column: getattr(task, column) for column in decision_columns}
            audit_rows.append(
                {
                    **identity,
                    "model_id": model_id,
                    "outcome_variants": outcome_variants,
                    "prediction_variants": len(task_predictions),
                    "aligned_variants": len(aligned),
                    "outcome_alignment_fraction": aligned_fraction,
                    "status": status,
                }
            )
            if status != "included":
                continue
            observed = aligned["effect"].to_numpy(dtype=float)
            predicted = aligned["score"].to_numpy(dtype=float)
            selection = top_selection_metrics(observed, predicted, fraction=0.1)
            metric_rows.append(
                {
                    **identity,
                    "model_id": model_id,
                    "aligned_variants": len(aligned),
                    "spearman": regression_metrics(observed, predicted).spearman,
                    **selection,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise ValueError("No confirmation task-model metrics passed the frozen alignment rules")
    TASK_METRIC_SCHEMA.validate(metrics)
    audit = pd.DataFrame(audit_rows)
    return metrics.sort_values([*CONFIRMATION_IDENTITY, "model_id"]), audit.sort_values(
        [*CONFIRMATION_IDENTITY, "model_id"]
    )


def _prepare_conservative_confirmation_tasks(
    decisions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: AuditorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auditor_metrics = metrics.loc[
        metrics["model_id"].astype(str).isin(config.model_ids)
    ].copy()
    task_keys = CONFIRMATION_IDENTITY
    oracle = auditor_metrics.groupby(task_keys)["selection_gain_sd"].max()
    baseline = auditor_metrics.loc[
        auditor_metrics["model_id"].astype(str).eq(config.baseline_model)
    ].copy()
    if baseline.duplicated(task_keys).any():
        raise ValueError("Confirmation baseline metrics are not task-unique")
    baseline = baseline.rename(
        columns={
            "selection_gain_sd": "vespag_selection_gain_sd",
            "spearman": "vespag_spearman",
            "top_recall": "vespag_top_recall",
            "ndcg": "vespag_ndcg",
            "best_variant_regret_sd": "vespag_best_variant_regret_sd",
        }
    )
    keep = [
        *task_keys,
        "vespag_selection_gain_sd",
        "vespag_spearman",
        "vespag_top_recall",
        "vespag_ndcg",
        "vespag_best_variant_regret_sd",
    ]
    evaluated = decisions.merge(
        baseline.loc[:, keep], on=task_keys, how="left", validate="one_to_one"
    )
    evaluated["oracle_selection_gain_sd"] = pd.MultiIndex.from_frame(
        evaluated.loc[:, task_keys]
    ).map(oracle)
    evaluated["evaluation_status"] = np.where(
        evaluated["vespag_selection_gain_sd"].notna()
        & evaluated["oracle_selection_gain_sd"].notna(),
        "included",
        "missing_frozen_baseline_metric",
    )
    missing = evaluated.loc[evaluated["evaluation_status"].ne("included")].copy()
    evaluated = evaluated.loc[evaluated["evaluation_status"].eq("included")].copy()
    if evaluated.empty:
        raise ValueError("No frozen Conservative Auditor tasks have evaluable VespaG outcomes")
    evaluated["selection_gain_sd"] = evaluated["vespag_selection_gain_sd"]
    evaluated["selection_regret_sd"] = (
        evaluated["oracle_selection_gain_sd"] - evaluated["selection_gain_sd"]
    )
    return evaluated.reset_index(drop=True), missing.reset_index(drop=True)


def _confirmation_policy_statistics(
    frame: pd.DataFrame,
    coverage_grid: np.ndarray,
    *,
    rank_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    order = frame.sort_values(rank_column, kind="stable").index.to_numpy()
    rows: list[dict[str, object]] = []
    candidate_regret: list[float] = []
    candidate_risk: list[float] = []
    candidate_utility: list[float] = []
    baseline_regret = float(frame["selection_regret_sd"].mean())
    baseline_risk = float((frame["selection_gain_sd"] <= 0).mean())
    baseline_utility = float(frame["selection_gain_sd"].mean())
    for coverage in coverage_grid:
        count = max(1, int(np.ceil(len(order) * coverage)))
        selected = frame.loc[order[:count]]
        regret = float(selected["selection_regret_sd"].mean())
        risk = float((selected["selection_gain_sd"] <= 0).mean())
        utility = float(selected["selection_gain_sd"].mean())
        candidate_regret.append(regret)
        candidate_risk.append(risk)
        candidate_utility.append(utility)
        for policy, values in (
            ("conservative_auditor_v2", (regret, risk, utility)),
            (
                "always_vespag_expected_random_abstention",
                (baseline_regret, baseline_risk, baseline_utility),
            ),
        ):
            rows.append(
                {
                    "policy": policy,
                    "coverage": float(coverage),
                    "retained_tasks": count,
                    "mean_selection_regret_sd": values[0],
                    "failure_rate": values[1],
                    "mean_selection_gain_sd": values[2],
                }
            )
    half = int(np.argmin(np.abs(coverage_grid - 0.5)))
    improvements = {
        "regret_auc_improvement": float(
            trapezoid(
                np.repeat(baseline_regret, len(coverage_grid)), coverage_grid
            )
            - trapezoid(candidate_regret, coverage_grid)
        ),
        "risk_auc_improvement": float(
            trapezoid(np.repeat(baseline_risk, len(coverage_grid)), coverage_grid)
            - trapezoid(candidate_risk, coverage_grid)
        ),
        "utility_auc_improvement": float(
            trapezoid(candidate_utility, coverage_grid)
            - trapezoid(
                np.repeat(baseline_utility, len(coverage_grid)), coverage_grid
            )
        ),
        "regret_improvement_at_50pct": baseline_regret - candidate_regret[half],
        "risk_improvement_at_50pct": baseline_risk - candidate_risk[half],
        "mean_gain_change_at_50pct": candidate_utility[half] - baseline_utility,
        "candidate_failure_rate_at_50pct": candidate_risk[half],
        "baseline_failure_rate_at_50pct": baseline_risk,
        "candidate_mean_gain_at_50pct": candidate_utility[half],
        "baseline_mean_gain_at_50pct": baseline_utility,
    }
    return pd.DataFrame(rows), improvements


def _bootstrap_conservative_confirmation(
    frame: pd.DataFrame,
    config: AuditorConfig,
    *,
    rank_column: str,
    seed_offset: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hierarchy = _hierarchy(frame)
    repeats = int(config.payload["bootstrap_repeats"])
    rng = np.random.default_rng(config.seed + seed_offset)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        indices: list[int] = []
        for family_index in rng.integers(0, len(hierarchy), len(hierarchy)):
            proteins = hierarchy[family_index]
            for protein_index in rng.integers(0, len(proteins), len(proteins)):
                tasks = proteins[protein_index]
                indices.extend(tasks[rng.integers(0, len(tasks), len(tasks))])
        sampled = frame.loc[indices].reset_index(drop=True)
        _, improvements = _confirmation_policy_statistics(
            sampled, config.coverage_grid, rank_column=rank_column
        )
        rows.append({"repeat": repeat, **improvements})
    replicates = pd.DataFrame(rows)
    _, point = _confirmation_policy_statistics(
        frame, config.coverage_grid, rank_column=rank_column
    )
    summary: dict[str, object] = {
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "family, then protein, then assay",
    }
    for column, value in point.items():
        values = replicates[column].to_numpy(dtype=float)
        summary[f"{column}_point"] = float(value)
        summary[f"{column}_ci_low"] = float(np.quantile(values, 0.025))
        summary[f"{column}_ci_high"] = float(np.quantile(values, 0.975))
        summary[f"{column}_one_sided_p"] = float(
            (1 + np.sum(values <= 0)) / (len(values) + 1)
        )
    return replicates, summary


def _holm_adjustment(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def evaluate_conservative_confirmation(
    config_path: Path,
    final_freeze_path: Path,
    decisions_path: Path,
    prediction_registry_path: Path,
    outcomes_path: Path,
    outcome_lock_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run the one-shot preregistered v2 confirmation without refitting any model."""
    lock = assert_evaluation_artifacts_locked(
        outcome_lock_path,
        prediction_artifact=decisions_path,
        method_artifact=final_freeze_path,
        outcome_artifact=outcomes_path,
    )
    if sha256_file(config_path) not in set(
        dict(lock.get("method_artifacts", {})).values()
    ):
        raise ValueError("Auditor config is not a frozen method artifact")
    if sha256_file(prediction_registry_path) not in set(
        dict(lock.get("prediction_artifacts", {})).values()
    ):
        raise ValueError("Prediction registry is not a frozen prediction artifact")
    config = load_auditor_config(config_path)
    final_freeze = json.loads(Path(final_freeze_path).read_text(encoding="utf-8"))
    if final_freeze.get("protocol_id") != "variantshift-confirmation-freeze-v2":
        raise ValueError("Confirmation evaluation requires the registered v2 freeze")
    if config.baseline_model != "vespag":
        raise ValueError("Registered v2 confirmation fixes VespaG as the baseline model")
    decisions = pd.read_csv(decisions_path)
    registry = pd.read_csv(prediction_registry_path)
    outcomes = pd.read_csv(outcomes_path)
    expected_protocol = str(final_freeze["protocol_id"])
    if set(decisions["protocol_id"].astype(str)) != {expected_protocol}:
        raise ValueError("Frozen decisions do not match the registered protocol")
    if set(outcomes["protocol_id"].astype(str)) != {expected_protocol}:
        raise ValueError("Revealed outcomes do not match the registered protocol")
    metrics, metric_audit = build_conservative_confirmation_metrics(
        decisions, registry, outcomes, config
    )
    evaluated, missing_tasks = _prepare_conservative_confirmation_tasks(
        decisions, metrics, config
    )
    curves, pooled_point = _confirmation_policy_statistics(
        evaluated, config.coverage_grid, rank_column="pooled_confidence_rank"
    )
    pooled_replicates, pooled_summary = _bootstrap_conservative_confirmation(
        evaluated, config, rank_column="pooled_confidence_rank"
    )
    pooled_replicates["scope"] = "pooled"
    panel_rows: list[dict[str, object]] = []
    replicate_frames = [pooled_replicates]
    panel_bootstrap: dict[str, dict[str, object]] = {}
    for panel_index, panel_id in enumerate(CONFIRMATION_V2_PANELS):
        panel = evaluated.loc[evaluated["panel_id"].astype(str).eq(panel_id)].copy()
        if panel.empty:
            raise ValueError(f"Registered primary panel is unevaluable: {panel_id}")
        panel_curves, panel_point = _confirmation_policy_statistics(
            panel, config.coverage_grid, rank_column="panel_confidence_rank"
        )
        panel_curves.insert(0, "panel_id", panel_id)
        curves = pd.concat([curves, panel_curves], ignore_index=True)
        panel_replicates, panel_summary = _bootstrap_conservative_confirmation(
            panel,
            config,
            rank_column="panel_confidence_rank",
            seed_offset=(panel_index + 1) * 100_000,
        )
        panel_replicates["scope"] = panel_id
        replicate_frames.append(panel_replicates)
        panel_bootstrap[panel_id] = panel_summary
        panel_rows.append({"panel_id": panel_id, "tasks": len(panel), **panel_point})
    panel_summary_frame = pd.DataFrame(panel_rows)

    secondary_p = {
        "pooled_failure_risk": float(
            pooled_summary["risk_auc_improvement_one_sided_p"]
        ),
        "pooled_mean_gain_50pct": float(
            pooled_summary["mean_gain_change_at_50pct_one_sided_p"]
        ),
    }
    secondary_p.update(
        {
            f"{panel_id}_regret_direction": float(
                summary["regret_auc_improvement_one_sided_p"]
            )
            for panel_id, summary in panel_bootstrap.items()
        }
    )
    holm = _holm_adjustment(secondary_p)
    direction_consistent = bool(
        panel_summary_frame["regret_auc_improvement"].gt(0).all()
    )
    gates = [
        {
            "gate": "primary_regret_coverage",
            "passed": float(pooled_summary["regret_auc_improvement_ci_low"]) > 0,
            "value": pooled_point["regret_auc_improvement"],
            "ci_low": pooled_summary["regret_auc_improvement_ci_low"],
            "ci_high": pooled_summary["regret_auc_improvement_ci_high"],
            "criterion": "family-bootstrap 95% CI lower bound strictly above zero",
        },
        {
            "gate": "failure_risk_noninferiority",
            "passed": float(pooled_summary["risk_auc_improvement_ci_low"]) >= 0,
            "value": pooled_point["risk_auc_improvement"],
            "ci_low": pooled_summary["risk_auc_improvement_ci_low"],
            "ci_high": pooled_summary["risk_auc_improvement_ci_high"],
            "holm_adjusted_p": holm["pooled_failure_risk"],
            "criterion": "family-bootstrap 95% CI lower bound at least zero",
        },
        {
            "gate": "mean_utility_noninferiority_at_50pct",
            "passed": pooled_point["mean_gain_change_at_50pct"] >= 0,
            "value": pooled_point["mean_gain_change_at_50pct"],
            "holm_adjusted_p": holm["pooled_mean_gain_50pct"],
            "criterion": "point estimate at least zero at 50% task coverage",
        },
        {
            "gate": "domainome_venus_direction_consistency",
            "passed": direction_consistent,
            "value": float(panel_summary_frame["regret_auc_improvement"].min()),
            "holm_adjusted_p": {
                panel_id: holm[f"{panel_id}_regret_direction"]
                for panel_id in CONFIRMATION_V2_PANELS
            },
            "criterion": "positive regret-coverage improvement in Domainome and untouched Venus",
        },
        {
            "gate": "no_post_reveal_refit",
            "passed": True,
            "value": "evaluation uses frozen decisions and contains no fitting step",
            "criterion": "no model, feature, threshold, rank, or exclusion refit",
        },
    ]
    acceptance = {
        "schema_version": 1,
        "protocol_id": expected_protocol,
        "status": "pass" if all(bool(gate["passed"]) for gate in gates) else "fail",
        "claim_allowed": all(bool(gate["passed"]) for gate in gates),
        "negative_result_rule": (
            "If any gate fails, do not claim improved deployment reliability; report the "
            "failure as evidence about benchmark transportability without refitting v2."
        ),
        "gates": gates,
    }
    model_summary = (
        metrics.groupby(["panel_id", "model_id"], as_index=False)
        .agg(
            tasks=("assay_id", "nunique"),
            mean_selection_gain_sd=("selection_gain_sd", "mean"),
            failure_rate=("selection_gain_sd", lambda values: float((values <= 0).mean())),
            mean_spearman=("spearman", "mean"),
            mean_top_recall=("top_recall", "mean"),
            mean_ndcg=("ndcg", "mean"),
            mean_best_variant_regret_sd=("best_variant_regret_sd", "mean"),
        )
        .sort_values(["panel_id", "mean_selection_gain_sd"], ascending=[True, False])
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics": output_dir / "confirmation-task-model-metrics.csv",
        "metric_audit": output_dir / "confirmation-task-model-metric-audit.csv",
        "evaluated_tasks": output_dir / "confirmation-auditor-tasks.csv",
        "missing_tasks": output_dir / "confirmation-auditor-missing-tasks.csv",
        "curves": output_dir / "confirmation-risk-regret-coverage.csv",
        "model_summary": output_dir / "confirmation-model-summary.csv",
        "panel_summary": output_dir / "confirmation-panel-summary.csv",
        "bootstrap": output_dir / "confirmation-hierarchical-bootstrap.csv.gz",
        "bootstrap_summary": output_dir / "confirmation-bootstrap-summary.json",
        "acceptance": output_dir / "confirmation-acceptance.json",
        "manifest": output_dir / "confirmation-evaluation-manifest.json",
    }
    write_table(metrics, outputs["metrics"])
    write_table(metric_audit, outputs["metric_audit"])
    write_table(evaluated, outputs["evaluated_tasks"])
    write_table(missing_tasks, outputs["missing_tasks"])
    write_table(curves, outputs["curves"])
    write_table(model_summary, outputs["model_summary"])
    write_table(panel_summary_frame, outputs["panel_summary"])
    pd.concat(replicate_frames, ignore_index=True).to_csv(
        outputs["bootstrap"],
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    bootstrap_payload = {
        "schema_version": 1,
        "pooled": pooled_summary,
        "panels": panel_bootstrap,
        "secondary_holm_adjusted_p_values": holm,
    }
    outputs["bootstrap_summary"].write_text(
        json.dumps(bootstrap_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["acceptance"].write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": expected_protocol,
        "classification": "one-shot preregistered confirmation evaluation",
        "method_refit": False,
        "threshold_refit": False,
        "included_tasks": len(evaluated),
        "missing_tasks": len(missing_tasks),
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                config_path,
                final_freeze_path,
                decisions_path,
                prediction_registry_path,
                outcomes_path,
                outcome_lock_path,
            ]
        },
        "artifacts": {
            str(path): sha256_file(path)
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
