import json

import pandas as pd
import pytest

from variantshift.outcome_lock import (
    assert_outcomes_accessible,
    assert_target_only,
    create_outcome_lock,
    freeze_predictions,
    read_outcome_lock,
    record_outcome_reveal,
    register_confirmation,
)


def test_confirmation_lock_is_one_way_and_requires_public_registration(tmp_path) -> None:
    target = tmp_path / "targets.csv"
    prediction = tmp_path / "predictions.csv"
    method = tmp_path / "method.json"
    outcome = tmp_path / "outcomes.csv"
    target.write_text("target_id,sequence\nT1,AC\n")
    prediction.write_text("target_id,variant_id,score\nT1,A1C,0.1\n")
    method.write_text(json.dumps({"method": "frozen"}))
    outcome.write_text("target_id,variant_id,effect\nT1,A1C,1.0\n")
    lock = tmp_path / "outcome-lock.json"
    create_outcome_lock(lock, protocol_id="confirm-v1", target_artifacts=[target])
    with pytest.raises(PermissionError):
        assert_outcomes_accessible(lock)
    freeze_predictions(
        lock,
        prediction_artifacts=[prediction],
        method_artifacts=[method],
    )
    with pytest.raises(ValueError, match="public HTTP"):
        register_confirmation(lock, registration_uri="private-record")
    register_confirmation(lock, registration_uri="https://osf.io/example")
    assert_outcomes_accessible(lock)
    record_outcome_reveal(lock, outcome_artifacts=[outcome])
    assert read_outcome_lock(lock)["state"] == "revealed"


def test_target_only_firewall_rejects_outcome_columns() -> None:
    assert_target_only(pd.DataFrame({"target_id": ["T1"], "sequence": ["AC"]}))
    with pytest.raises(ValueError, match="prohibited"):
        assert_target_only(
            pd.DataFrame({"target_id": ["T1"], "sequence": ["AC"], "effect": [1.0]})
        )
