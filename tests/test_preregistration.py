import json

import pandas as pd
import pytest

from variantshift.outcome_lock import create_outcome_lock, freeze_predictions
from variantshift.preregistration import (
    build_preregistration_bundle,
    build_preregistration_model_audit,
)


def test_qualification_audit_is_converted_without_outcomes(tmp_path) -> None:
    audit_path = tmp_path / "qualification.csv"
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "preregistration-model-audit.csv"
    pd.DataFrame(
        {
            "model_id": ["passed", "failed"],
            "family": ["sequence", "structure"],
            "qualification_status": ["passed", "parity_failed"],
        }
    ).to_csv(audit_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "shared_confirmation_targets": 413,
                "gates": {
                    "configurations": True,
                    "families": True,
                    "shared_confirmation_targets": True,
                },
            }
        )
    )

    manifest = build_preregistration_model_audit(
        audit_path, summary_path, output_path
    )

    converted = pd.read_csv(output_path)
    assert converted["primary_eligible"].tolist() == [True, False]
    assert converted["primary_shared_target_count"].tolist() == [413, 413]
    assert manifest["outcomes_accessed"] is False
    assert manifest["feasibility_gate_passed"] is True


def test_preregistration_is_built_only_from_frozen_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    target = tmp_path / "targets.csv"
    prediction = tmp_path / "predictions.csv"
    target.write_text("target_id,sequence\nT1,AC\n")
    prediction.write_text("target_id,variant_id,score\nT1,A1C,0.1\n")
    method = tmp_path / "method.json"
    method.write_text(json.dumps({"name": "VariantShift", "features": ["length"]}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "protocol_id": "p1",
                "panel_id": "confirmation",
                "inclusion": {"minimum_variants": 100},
                "exclusion": {"outcome_seen": True},
            }
        )
    )
    model_audit = tmp_path / "model-audit.csv"
    eligible_models = [f"eligible-{index}" for index in range(8)]
    pd.DataFrame(
        {
            "model_id": [*eligible_models, "failed"],
            "family": [*[f"family-{index % 4}" for index in range(8)], "failed-family"],
            "primary_eligible": [*[True] * 8, False],
            "exclusion_reason": [*[""] * 8, "parity_failed"],
            "primary_shared_target_count": [300] * 9,
            "feasibility_gate_passed": [True] * 9,
        }
    ).to_csv(model_audit, index=False)
    lock = tmp_path / "outcome-lock.json"
    create_outcome_lock(lock, protocol_id="p1", target_artifacts=[target, protocol])
    freeze_predictions(lock, prediction_artifacts=[prediction], method_artifacts=[method])
    monkeypatch.setattr("variantshift.preregistration.git_revision", lambda _: "abc123")
    outputs = build_preregistration_bundle(
        protocol, lock, model_audit, method, tmp_path / "registration"
    )
    payload = json.loads(outputs["registration"].read_text())
    assert payload["outcome_state"] == "predictions_frozen"
    assert payload["eligible_models"] == eligible_models
    assert payload["eligible_model_family_count"] == 4
    assert payload["shared_confirmation_targets"] == 300
    assert payload["primary_endpoint"] == "task-level selection-regret coverage AUC"
    assert "failure risk-coverage AUC" in payload["secondary_endpoints"]
    assert "failed" in payload["excluded_models"]
    assert outputs["checksums"].read_text().count("\n") == 6


def test_preregistration_rejects_an_underpowered_model_panel(tmp_path) -> None:
    target = tmp_path / "targets.csv"
    prediction = tmp_path / "predictions.csv"
    method = tmp_path / "method.json"
    protocol = tmp_path / "protocol.json"
    target.write_text("target_id,sequence\nT1,AC\n")
    prediction.write_text("target_id,variant_id,score\nT1,A1C,0.1\n")
    method.write_text(json.dumps({"name": "VariantShift"}))
    protocol.write_text(json.dumps({"protocol_id": "p1", "panel_id": "confirmation"}))
    audit = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "model_id": ["only-model"],
            "family": ["sequence"],
            "primary_eligible": [True],
            "exclusion_reason": [""],
            "primary_shared_target_count": [300],
            "feasibility_gate_passed": [False],
        }
    ).to_csv(audit, index=False)
    lock = tmp_path / "lock.json"
    create_outcome_lock(lock, protocol_id="p1", target_artifacts=[target])
    freeze_predictions(lock, prediction_artifacts=[prediction], method_artifacts=[method])
    with pytest.raises(ValueError, match="feasibility gate failed"):
        build_preregistration_bundle(protocol, lock, audit, method, tmp_path / "registration")
