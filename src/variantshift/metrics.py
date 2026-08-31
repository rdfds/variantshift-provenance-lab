"""Metrics that separate ranking, calibration, and absolute error."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .mutations import parse_variant


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

    # Peak-to-peak avoids cancellation in ``std`` for a constant vector with a
    # large offset (observed in a small number of ProteinGym assay/model pairs).
    if np.ptp(observed) < 1e-12 or np.ptp(predicted) < 1e-12:
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
    rank = min(residuals.size, int(np.ceil((residuals.size + 1) * coverage)))
    return float(np.sort(np.abs(residuals))[rank - 1])


def interval_coverage(observed: np.ndarray, predicted: np.ndarray, radius: float) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(observed - predicted) <= radius))


def interval_metrics(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Return marginal coverage and interval width for aligned intervals."""
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if observed.shape != lower.shape or observed.shape != upper.shape:
        raise ValueError("Observed values and interval bounds must have matching shapes")
    if np.any(lower > upper):
        raise ValueError("Interval lower bounds cannot exceed upper bounds")
    widths = upper - lower
    return {
        "observed_coverage": float(np.mean((observed >= lower) & (observed <= upper))),
        "mean_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
    }


def position_conditional_coverage(
    codes: list[str],
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Summarize coverage after giving each mutated position equal weight."""
    if not (len(codes) == len(observed) == len(lower) == len(upper)):
        raise ValueError("Codes, values, and interval bounds must be aligned")
    positions = np.asarray([parse_variant(code)[0].position for code in codes], dtype=int)
    covered = (np.asarray(observed) >= np.asarray(lower)) & (
        np.asarray(observed) <= np.asarray(upper)
    )
    per_position = pd.Series(covered.astype(float)).groupby(positions).mean().to_numpy()
    return {
        "position_coverage_mean": float(np.mean(per_position)),
        "position_coverage_p10": float(np.quantile(per_position, 0.1)),
        "position_coverage_min": float(np.min(per_position)),
        "covered_positions": len(per_position),
    }


def top_selection_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    fraction: float = 0.1,
) -> dict[str, float]:
    """Measure whether predicted top variants recover experimentally strong variants."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted values must be aligned vectors")
    if not 0 < fraction <= 1:
        raise ValueError("Selection fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(len(observed) * fraction)))
    true_top = set(np.argpartition(observed, -count)[-count:])
    predicted_top = set(np.argpartition(predicted, -count)[-count:])
    overlap = len(true_top & predicted_top)
    baseline_mean = float(np.mean(observed))
    selected_mean = float(np.mean(observed[list(predicted_top)]))
    target_scale = float(np.std(observed))
    return {
        "top_fraction": fraction,
        "top_recall": overlap / count,
        "ndcg": normalized_discounted_cumulative_gain(observed, predicted),
        "selected_target_mean": selected_mean,
        "selection_gain_sd": (selected_mean - baseline_mean) / target_scale
        if target_scale > 1e-12
        else 0.0,
        "best_variant_regret_sd": (
            float(np.max(observed)) - float(np.max(observed[list(predicted_top)]))
        )
        / target_scale
        if target_scale > 1e-12
        else 0.0,
    }


def normalized_discounted_cumulative_gain(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Compute NDCG from outcome ranks so arbitrary assay scales and signs are safe."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted values must be aligned vectors")
    if not len(observed):
        raise ValueError("NDCG requires at least one observation")
    order = np.argsort(observed, kind="stable")
    relevance = np.empty(len(observed), dtype=float)
    relevance[order] = np.arange(1, len(observed) + 1, dtype=float) / len(observed)
    predicted_order = np.argsort(-predicted, kind="stable")
    ideal_order = np.argsort(-relevance, kind="stable")
    discounts = np.log2(np.arange(2, len(observed) + 2, dtype=float))
    discounted_gain = float(np.sum((np.exp2(relevance[predicted_order]) - 1) / discounts))
    ideal_gain = float(np.sum((np.exp2(relevance[ideal_order]) - 1) / discounts))
    return discounted_gain / ideal_gain if ideal_gain > 0 else 0.0


def risk_coverage_curve(
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
    *,
    retained_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> list[dict[str, float]]:
    """Compute selective absolute error after retaining least-uncertain predictions."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if observed.shape != predicted.shape or observed.shape != uncertainty.shape:
        raise ValueError("Observed values, predictions, and uncertainty must align")
    order = np.argsort(uncertainty, kind="stable")
    scale = float(np.std(observed))
    rows = []
    for fraction in retained_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Retained fractions must lie in (0, 1]")
        count = max(1, int(np.ceil(len(observed) * fraction)))
        selected = order[:count]
        mae = float(np.mean(np.abs(observed[selected] - predicted[selected])))
        rows.append(
            {
                "retained_fraction": fraction,
                "retained_rows": count,
                "mae": mae,
                "normalized_mae": mae / scale if scale > 1e-12 else np.nan,
                "uncertainty_threshold": float(uncertainty[selected].max()),
            }
        )
    return rows
