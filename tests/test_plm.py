import numpy as np
import pytest

from variantshift.plm import score_variant_log_odds


def test_scores_alternate_against_reference() -> None:
    token_index = {"A": 0, "C": 1, "D": 2, "E": 3}
    log_probabilities = np.array(
        [
            [-2.0, -0.5, -1.0, -3.0],
            [-1.0, -2.0, -0.4, -0.1],
        ]
    )
    assert score_variant_log_odds(log_probabilities, "A1C", token_index) == 1.5
    assert score_variant_log_odds(log_probabilities, "A1C/D2E", token_index) == pytest.approx(
        1.8
    )


def test_rejects_stop_variants() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        score_variant_log_odds(np.zeros((1, 2)), "A1*", {"A": 0, "*": 1})

