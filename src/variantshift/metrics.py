"""Metrics that separate ranking, calibration, and absolute error."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    spearman: float
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted values must be aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")

    if np.std(observed) == 0 or np.std(predicted) == 0:
        spearman = 0.0
    else:
        spearman = float(spearmanr(observed, predicted).statistic)
    return RegressionMetrics(
        spearman=spearman,
        rmse=float(mean_squared_error(observed, predicted) ** 0.5),
        mae=float(mean_absolute_error(observed, predicted)),
        r2=float(r2_score(observed, predicted)),
    )


def conformal_radius(residuals: np.ndarray, coverage: float = 0.8) -> float:
    """Finite-sample split-conformal radius using the conservative higher quantile."""
    residuals = np.asarray(residuals, dtype=float)
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one")
    if residuals.size == 0:
        raise ValueError("At least one calibration residual is required")
    level = min(1.0, np.ceil((residuals.size + 1) * coverage) / residuals.size)
    return float(np.quantile(np.abs(residuals), level, method="higher"))


def interval_coverage(observed: np.ndarray, predicted: np.ndarray, radius: float) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(observed - predicted) <= radius))

