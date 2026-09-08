import numpy as np
import pandas as pd
import pytest

from variantshift.transportability import (
    TransportConfig,
    confirmation_acceptance_gates,
    cross_fitted_transport_predictions,
    fit_frozen_transport_model,
    group_conformal_quantile,
    hierarchical_bootstrap_mean,
    hierarchical_conformal_quantile,
    predict_with_frozen_transport_model,
    selective_policy_curve,
    summarize_policy_curves,
)


def _transport_frame() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    for family_index in range(10):
        for model_index, model in enumerate(["good", "weak"]):
            signal = rng.normal()
            rows.append(
                {
                    "protocol_id": "development",
                    "panel_id": "proteingym",
                    "dataset_id": "pg",
                    "assay_id": f"A{family_index}",
                    "task_id": f"A{family_index}",
                    "target_id": f"T{family_index}",
                    "protein_id": f"P{family_index}",
                    "family_id": f"F{family_index}",
                    "model_id": model,
                    "protein_length": 100 + family_index,
                    "msa_neff": 5 + family_index,
                    "score_dispersion": abs(signal) + model_index,
                    "model_family": model,
                    "selection_gain_sd": signal + (1.0 if model == "good" else -0.5),
                }
            )
    return pd.DataFrame(rows)


def _config() -> TransportConfig:
    return TransportConfig(
        numeric_features=("protein_length", "msa_neff", "score_dispersion"),
        categorical_features=("model_family",),
        outer_folds=5,
        seed=9,
    )


def test_cross_fitted_transport_never_predicts_from_its_own_family() -> None:
    predictions, audit = cross_fitted_transport_predictions(_transport_frame(), _config())
    assert predictions["fold"].ge(0).all()
    assert predictions["predicted_selection_gain_sd"].notna().all()
    assert predictions.groupby("family_id")["fold"].nunique().eq(1).all()
    assert audit["test_groups"].eq(2).all()


def test_group_conformal_uses_worst_residual_per_family() -> None:
    residuals = np.asarray([0.1, 2.0, 0.2, 0.3])
    groups = np.asarray(["A", "A", "B", "B"])
    assert group_conformal_quantile(residuals, groups, coverage=0.5) == 2.0


def test_hierarchical_conformal_gives_families_equal_total_mass() -> None:
    residuals = np.asarray([0.1, 10.0, 0.2, 0.3])
    groups = np.asarray(["A", "A", "B", "B"])
    assert hierarchical_conformal_quantile(residuals, groups, coverage=0.5) == 0.3


def test_conformal_bound_does_not_change_the_frozen_model_choice() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["T", "T"],
            "model_id": ["chosen", "other"],
            "selection_gain_sd": [1.0, -1.0],
            "decision_selection_gain_sd": [0.8, 0.7],
            "lower_selection_gain_sd": [-10.0, 10.0],
            "auditor_selected": [True, False],
            "auditor_confidence": [0.5, np.nan],
        }
    )
    curve = selective_policy_curve(frame, _config(), policy="variantshift")
    assert curve.loc[curve["coverage"].eq(1.0), "mean_selection_gain_sd"].item() == 1.0


def test_selective_curve_and_hierarchical_bootstrap_are_task_level() -> None:
    frame = _transport_frame()
    predictions, _ = cross_fitted_transport_predictions(frame, _config())
    curve = selective_policy_curve(predictions, _config(), policy="variantshift")
    assert curve["coverage"].tolist()[-1] == 1.0
    assert curve["retained_tasks"].max() == 10
    bootstrap = hierarchical_bootstrap_mean(
        frame,
        "selection_gain_sd",
        family_column="family_id",
        protein_column="protein_id",
        assay_column="assay_id",
        repeats=20,
        seed=3,
    )
    assert len(bootstrap) == 20
    assert np.isfinite(bootstrap["estimate"]).all()
    summary = summarize_policy_curves(curve)
    assert summary.loc[0, "risk_coverage_auc"] >= 0


def test_transport_features_reject_outcome_leakage() -> None:
    config = TransportConfig(
        numeric_features=("observed_spearman",),
        categorical_features=(),
    )
    with pytest.raises(ValueError, match="Outcome-derived"):
        config.validate()


def test_frozen_bundle_emits_elastic_and_feature_ablation_predictions() -> None:
    frame = _transport_frame()
    bundle = fit_frozen_transport_model(frame, _config())
    confirmation = frame.drop(columns="selection_gain_sd")
    predicted = predict_with_frozen_transport_model(bundle, confirmation)
    assert predicted["elastic_predicted_selection_gain_sd"].notna().all()
    assert {
        "ablation__msa_only__lower_selection_gain_sd",
        "ablation__ensemble_only__lower_selection_gain_sd",
    }.issubset(predicted.columns)


def test_frozen_bundle_rejects_development_outcome_diagnostics() -> None:
    frame = _transport_frame()
    bundle = fit_frozen_transport_model(frame, _config())
    confirmation = frame.drop(columns="selection_gain_sd")
    confirmation["development_crossover_probability_supervised_wins"] = 0.5
    with pytest.raises(ValueError, match="outcome-derived"):
        predict_with_frozen_transport_model(bundle, confirmation)


def test_priority_treats_missing_msa_as_neutral() -> None:
    frame = _transport_frame()
    bundle = fit_frozen_transport_model(frame, _config())
    confirmation = frame.drop(columns="selection_gain_sd")
    confirmation.loc[confirmation.index[:2], "msa_neff"] = np.nan
    predicted = predict_with_frozen_transport_model(bundle, confirmation)
    selected = predicted.loc[predicted["auditor_selected"]]
    assert selected["auditor_confidence"].notna().all()


def test_confirmation_gates_are_machine_readable_and_require_both_panels() -> None:
    config = _config()
    frame = _transport_frame()
    frame["panel_id"] = np.where(
        frame["family_id"].str.removeprefix("F").astype(int) < 5,
        "human-domainome-v1",
        "mavedb-complement-v1",
    )
    predictions, _ = cross_fitted_transport_predictions(frame, config)
    predictions["elastic_predicted_selection_gain_sd"] = predictions[
        "predicted_selection_gain_sd"
    ]
    predictions["ablation__msa_only__lower_selection_gain_sd"] = 0.0
    predictions["ablation__ensemble_only__lower_selection_gain_sd"] = 0.0
    policies = [
        "variantshift",
        "elastic_net",
        "ablation:msa_only",
        "ablation:ensemble_only",
    ]
    curves = pd.concat(
        [selective_policy_curve(predictions, config, policy=policy) for policy in policies],
        ignore_index=True,
    )
    bootstrap_summary = pd.DataFrame(
        {
            "comparator": [
                "elastic_net",
                "ablation:msa_only",
                "ablation:ensemble_only",
            ],
            "risk_coverage_auc_improvement": [0.1, 0.1, 0.1],
            "regret_coverage_auc_improvement": [0.1, 0.1, 0.1],
            "regret_ci_low": [0.01, 0.01, 0.01],
            "holm_adjusted_p_value": [0.03, 0.04, 0.04],
        }
    )
    panel_summary = pd.DataFrame(
        {
            "panel_id": ["human-domainome-v1", "mavedb-complement-v1"],
            "risk_coverage_auc_improvement": [0.02, 0.01],
            "regret_coverage_auc_improvement": [0.02, 0.01],
        }
    )
    result = confirmation_acceptance_gates(
        predictions,
        curves,
        bootstrap_summary,
        panel_summary,
        config,
        comparator="elastic_net",
        required_panels=("human-domainome-v1", "mavedb-complement-v1"),
        negative_conclusion={
            "statement": "One preregistered subgroup does not transport.",
            "interpretation_change": "Do not infer cross-family utility from random splits.",
            "evidence_artifacts": ["confirmation-panel-summary.csv"],
        },
    )
    assert result["schema_version"] == 2
    assert not result["missing_primary_panels"]
    assert {row["gate"] for row in result["gates"]} == {
        "primary_regret_coverage",
        "selective_utility_at_50pct",
        "nominal_90pct_coverage",
        "primary_panel_direction_consistency",
        "feature_ablation",
        "useful_negative_conclusion",
    }
