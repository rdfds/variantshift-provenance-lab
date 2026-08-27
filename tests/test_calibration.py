import numpy as np

from variantshift.calibration import interval_suite
from variantshift.metrics import interval_metrics, risk_coverage_curve


def test_interval_suite_is_label_blind_at_test_time_and_returns_aligned_bounds():
    fit_codes = [f"A{position}C" for position in range(1, 21)]
    calibration_codes = [f"A{position}D" for position in range(1, 21)]
    test_codes = ["A25C", "A30D", "A40E"]
    observed = np.linspace(0.0, 1.0, 20)
    predicted = observed + 0.1
    test_prediction = np.asarray([0.2, 0.4, 0.8])
    intervals = interval_suite(
        fit_codes,
        calibration_codes,
        observed,
        predicted,
        test_codes,
        test_prediction,
    )
    assert {interval.method for interval in intervals} == {
        "standard_split",
        "mondrian_substitution",
        "position_distance_scaled",
    }
    for interval in intervals:
        assert interval.lower.shape == test_prediction.shape
        assert np.all(interval.lower <= interval.upper)
    standard = next(interval for interval in intervals if interval.method == "standard_split")
    shifted = next(
        interval for interval in intervals if interval.method == "position_distance_scaled"
    )
    assert shifted.uncertainty[-1] > standard.uncertainty[-1]
    summary = interval_metrics(test_prediction, standard.lower, standard.upper)
    assert summary["observed_coverage"] == 1.0


def test_risk_coverage_prefers_low_uncertainty_rows_first():
    curve = risk_coverage_curve(
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.1, 0.2, 0.8, 1.0]),
        np.asarray([0.1, 0.2, 0.8, 1.0]),
        retained_fractions=(0.5, 1.0),
    )
    assert curve[0]["mae"] < curve[1]["mae"]
