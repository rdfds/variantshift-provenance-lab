import numpy as np
import pytest

from variantshift.metrics import conformal_radius, interval_coverage, regression_metrics


def test_perfect_predictions() -> None:
    values = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(values, values)
    assert metrics.spearman == pytest.approx(1.0)
    assert metrics.rmse == 0
    assert metrics.r2 == 1


def test_conformal_radius_uses_absolute_residuals() -> None:
    radius = conformal_radius(np.array([-1.0, 0.25, 0.5, 2.0]), coverage=0.5)
    assert radius == 1.0


def test_interval_coverage() -> None:
    observed = np.array([0.0, 1.0, 3.0])
    predicted = np.array([0.0, 2.0, 1.0])
    assert interval_coverage(observed, predicted, radius=1.0) == pytest.approx(2 / 3)

