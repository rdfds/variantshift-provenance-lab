import pandas as pd

from variantshift.confirmation_features import _interval_coverage, _score_shape


def test_interval_coverage_merges_overlapping_domains() -> None:
    frame = pd.DataFrame({"domain_start": [1, 5], "domain_end": [5, 8]})
    assert _interval_coverage(frame, 10) == 0.8


def test_score_shape_records_missingness_and_peer_agreement() -> None:
    wide = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    result = _score_shape(wide, expected=4)
    assert result["a"]["missing_fraction"] == 0.25
    assert result["a"]["score_dispersion"] > 0
