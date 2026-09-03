import numpy as np
import pandas as pd
import pytest

from variantshift.transportability import (
    TransportConfig,
    cross_fitted_transport_predictions,
    group_conformal_quantile,
    hierarchical_bootstrap_mean,
    selective_policy_curve,
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


def test_transport_features_reject_outcome_leakage() -> None:
    config = TransportConfig(
        numeric_features=("observed_spearman",),
        categorical_features=(),
    )
    with pytest.raises(ValueError, match="Outcome-derived"):
        config.validate()
