import numpy as np
import pandas as pd

from variantshift.predictor_span_pilot import (
    evaluate_task,
    predictor_geometry,
    safe_spearman,
    selection_metrics,
)


def test_safe_spearman_handles_constant_vectors() -> None:
    assert np.isnan(safe_spearman(np.ones(4), np.arange(4)))
    assert np.isclose(safe_spearman(np.arange(4), np.arange(4)[::-1]), -1.0)


def test_predictor_geometry_detects_redundancy() -> None:
    axis = np.linspace(-1, 1, 100)
    matrix = np.column_stack([axis, axis * 2, -axis, axis + 0.001])
    result = predictor_geometry(matrix)
    assert result["pcs_90pct"] == 1
    assert result["effective_rank"] < 1.1


def test_selection_metrics_reward_correct_ranking() -> None:
    observed = np.arange(100, dtype=float)
    result = selection_metrics(observed, observed)
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["top_recall"], 1.0)
    assert result["selection_gain_sd"] > 1.0


def test_crossfit_linear_span_extracts_complementary_signal() -> None:
    positions = np.repeat(np.arange(30), 4)
    mutation = np.tile(np.arange(4), 30)
    first = np.sin(positions / 4) + mutation * 0.1
    second = np.cos(positions / 5) - mutation * 0.1
    observed = first - second
    frame = pd.DataFrame(
        {
            "mutant": [f"A{position + 1}C" for position in positions],
            "position": positions,
            "DMS_score": observed,
            "first": first,
            "second": second,
            "duplicate": first * 2,
        }
    )
    _, rows = evaluate_task(
        frame,
        ["first", "second", "duplicate"],
        minimum_predictor_coverage=0.9,
        outer_folds=5,
        alpha_grid=[0.1, 1.0, 10.0],
    )
    by_policy = {row["policy"]: row for row in rows}
    assert by_policy["crossfit_linear_span"]["spearman"] > 0.95
    assert (
        by_policy["crossfit_linear_span"]["spearman"]
        > by_policy["crossfit_best_single"]["spearman"]
    )
