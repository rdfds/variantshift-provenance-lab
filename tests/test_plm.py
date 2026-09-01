import numpy as np
import pytest

from variantshift.plm import (
    assign_positions_to_windows,
    long_sequence_windows,
    score_single_substitutions,
    score_variant_log_odds,
)


def test_esm_runtime_is_an_explicit_reusable_argument() -> None:
    import inspect

    from variantshift.plm import esm2_position_log_probabilities

    assert "runtime" in inspect.signature(esm2_position_log_probabilities).parameters


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


def test_long_sequence_window_assignment_is_complete_and_deterministic() -> None:
    windows = long_sequence_windows(2_000, window_size=1_022, overlap=256)
    assert windows[0] == (0, 1_022)
    assert windows[-1] == (978, 2_000)
    assignments = assign_positions_to_windows(
        2_000,
        [1, 900, 1_000, 1_500, 2_000],
        window_size=1_022,
        overlap=256,
    )
    assigned = [position for positions in assignments.values() for position in positions]
    assert sorted(assigned) == [1, 900, 1_000, 1_500, 2_000]
    assert len(assigned) == len(set(assigned))


def test_scores_single_substitutions_from_selected_positions() -> None:
    position_probabilities = {
        1: np.array([-2.0, -0.5, -1.0]),
        2: np.array([-1.0, -2.0, -0.4]),
    }
    scored = score_single_substitutions(
        position_probabilities,
        ["A1C", "C2D"],
        {"A": 0, "C": 1, "D": 2},
    )
    assert scored["prediction"].tolist() == pytest.approx([1.5, 1.6])
