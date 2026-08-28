"""Conformal interval baselines for structured protein-variant shifts.

The adaptive methods in this module are empirical comparators, not new coverage
theorems. Every method uses calibration labels only; test labels are consumed solely
by downstream evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import conformal_radius
from .mutations import parse_variant

RESIDUE_GROUPS = {
    **{residue: "hydrophobic" for residue in "AILMFWVY"},
    **{residue: "polar" for residue in "CNQST"},
    **{residue: "charged" for residue in "DEHKR"},
    **{residue: "special" for residue in "GP"},
}


@dataclass(frozen=True)
class ConformalIntervals:
    method: str
    lower: np.ndarray
    upper: np.ndarray
    uncertainty: np.ndarray


def mutation_group(code: str) -> str:
    """Map a single substitution to a coarse reference/alternate residue class."""
    mutations = parse_variant(code)
    if len(mutations) != 1:
        return "multiple"
    mutation = mutations[0]
    reference = RESIDUE_GROUPS.get(mutation.reference, "other")
    alternate = RESIDUE_GROUPS.get(mutation.alternate, "other")
    return f"{reference}>{alternate}"


def standard_intervals(
    calibration_observed: np.ndarray,
    calibration_predicted: np.ndarray,
    test_predicted: np.ndarray,
    *,
    coverage: float = 0.8,
) -> ConformalIntervals:
    """Ordinary split-conformal intervals with one pooled residual radius."""
    radius = conformal_radius(
        np.asarray(calibration_observed) - np.asarray(calibration_predicted),
        coverage=coverage,
    )
    predicted = np.asarray(test_predicted, dtype=float)
    uncertainty = np.full(len(predicted), radius, dtype=float)
    return ConformalIntervals(
        method="standard_split",
        lower=predicted - radius,
        upper=predicted + radius,
        uncertainty=uncertainty,
    )


def mondrian_intervals(
    calibration_codes: list[str],
    calibration_observed: np.ndarray,
    calibration_predicted: np.ndarray,
    test_codes: list[str],
    test_predicted: np.ndarray,
    *,
    coverage: float = 0.8,
    min_group_size: int = 20,
) -> ConformalIntervals:
    """Use coarse substitution-class-specific radii with a pooled fallback.

    Coarse groups are deliberately shared across positions, so the method remains
    defined when every test position is absent from training and calibration.
    """
    residuals = np.abs(
        np.asarray(calibration_observed, dtype=float)
        - np.asarray(calibration_predicted, dtype=float)
    )
    pooled = conformal_radius(residuals, coverage=coverage)
    calibration_groups = np.asarray([mutation_group(code) for code in calibration_codes])
    radii: dict[str, float] = {}
    for group in np.unique(calibration_groups):
        selected = residuals[calibration_groups == group]
        if len(selected) >= min_group_size:
            radii[str(group)] = conformal_radius(selected, coverage=coverage)
    test_radius = np.asarray(
        [radii.get(mutation_group(code), pooled) for code in test_codes], dtype=float
    )
    predicted = np.asarray(test_predicted, dtype=float)
    return ConformalIntervals(
        method="mondrian_substitution",
        lower=predicted - test_radius,
        upper=predicted + test_radius,
        uncertainty=test_radius,
    )


def _positions(codes: list[str]) -> np.ndarray:
    return np.asarray([parse_variant(code)[0].position for code in codes], dtype=float)


def _nearest_position_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.unique(np.asarray(reference, dtype=float))
    if reference.size == 0:
        raise ValueError("At least one reference position is required")
    indices = np.searchsorted(reference, query)
    left = reference[np.clip(indices - 1, 0, len(reference) - 1)]
    right = reference[np.clip(indices, 0, len(reference) - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right))


def distance_scaled_intervals(
    fit_codes: list[str],
    calibration_codes: list[str],
    calibration_observed: np.ndarray,
    calibration_predicted: np.ndarray,
    test_codes: list[str],
    test_predicted: np.ndarray,
    *,
    coverage: float = 0.8,
) -> ConformalIntervals:
    """Scale residual intervals by distance to the nearest fitted residue position.

    This is a transparent shift-aware heuristic: it widens intervals where a test
    residue is farther from positions used to fit the supervised model. It should be
    evaluated empirically and is not claimed to guarantee coverage under shift.
    """
    fit_positions = _positions(fit_codes)
    unique_fit = np.unique(fit_positions)
    gaps = np.diff(unique_fit)
    position_scale = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
    calibration_distance = _nearest_position_distance(
        _positions(calibration_codes), unique_fit
    )
    test_distance = _nearest_position_distance(_positions(test_codes), unique_fit)
    # Sublinear scaling avoids allowing a long contiguous gap to create an
    # effectively unbounded interval while still increasing uncertainty away
    # from fitted positions.
    calibration_scale = np.sqrt(
        1.0 + calibration_distance / max(position_scale, 1.0)
    )
    test_scale = np.sqrt(1.0 + test_distance / max(position_scale, 1.0))
    residuals = (
        np.abs(
            np.asarray(calibration_observed, dtype=float)
            - np.asarray(calibration_predicted, dtype=float)
        )
        / calibration_scale
    )
    radius = conformal_radius(residuals, coverage=coverage)
    half_width = radius * test_scale
    predicted = np.asarray(test_predicted, dtype=float)
    return ConformalIntervals(
        method="position_distance_scaled",
        lower=predicted - half_width,
        upper=predicted + half_width,
        uncertainty=half_width,
    )


def interval_suite(
    fit_codes: list[str],
    calibration_codes: list[str],
    calibration_observed: np.ndarray,
    calibration_predicted: np.ndarray,
    test_codes: list[str],
    test_predicted: np.ndarray,
    *,
    coverage: float = 0.8,
) -> tuple[ConformalIntervals, ...]:
    """Return the fixed set of calibration methods used by the extended benchmark."""
    return (
        standard_intervals(
            calibration_observed,
            calibration_predicted,
            test_predicted,
            coverage=coverage,
        ),
        mondrian_intervals(
            calibration_codes,
            calibration_observed,
            calibration_predicted,
            test_codes,
            test_predicted,
            coverage=coverage,
        ),
        distance_scaled_intervals(
            fit_codes,
            calibration_codes,
            calibration_observed,
            calibration_predicted,
            test_codes,
            test_predicted,
            coverage=coverage,
        ),
    )
