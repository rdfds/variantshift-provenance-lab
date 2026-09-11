import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from variantshift.conservative_auditor import (
    CONFIRMATION_V2_PANELS,
    _expected_fixed_baseline_statistics,
    _percentile,
    add_confirmation_coverage_decisions,
    build_outcome_free_task_frame,
    candidate_scores,
    evaluate_conservative_confirmation,
    load_auditor_config,
)
from variantshift.outcome_lock import (
    create_outcome_lock,
    freeze_predictions,
    record_outcome_reveal,
    register_confirmation,
)
from variantshift.provenance import sha256_file


def test_config_fixes_vespag_as_the_only_deployed_model() -> None:
    config = load_auditor_config(Path("configs/conservative-auditor-v2.json"))

    assert config.baseline_model == "vespag"
    assert "vespag" in config.model_ids
    assert config.payload["selected_policy"] == "harm_gain_meta"
    assert "no model override" in str(config.payload["policy"]).lower()


def test_candidate_scores_use_only_prediction_references() -> None:
    predictions = {
        "regret": np.array([0.1, 0.9]),
        "gain_meta": np.array([0.1, 0.9]),
        "gain_all": np.array([0.1, 0.9]),
        "gain_shape": np.array([0.1, 0.9]),
    }

    scores = candidate_scores(predictions, predictions)

    assert set(scores) == {
        "pure_regret",
        "harm_gain_meta",
        "harm_gain_all",
        "harm_gain_shape",
        "blend_0.5",
        "blend_1.0",
        "blend_1.5",
        "blend_2.0",
    }
    assert scores["pure_regret"] == pytest.approx([0.5, 0.0])
    assert np.isfinite(np.concatenate(list(scores.values()))).all()


def test_percentiles_are_stable_to_submachine_refit_noise() -> None:
    reference = np.array([0.1, 0.2, 0.2, 0.3])
    perturbed = reference + np.array([1e-16, -1e-16, 1e-16, -1e-16])

    assert _percentile(reference, reference) == pytest.approx(
        _percentile(perturbed, perturbed)
    )


def test_fixed_model_comparator_is_analytical_random_abstention_expectation() -> None:
    frame = pd.DataFrame(
        {
            "selection_regret_sd": [0.0, 0.3, 0.6],
            "selection_gain_sd": [-0.2, 0.1, 0.4],
        }
    )

    statistics = _expected_fixed_baseline_statistics(frame, np.array([0.1, 1.0]))

    assert statistics["regret_coverage_auc"] == pytest.approx(0.27)
    assert statistics["risk_coverage_auc"] == pytest.approx(0.3)
    assert statistics["mean_regret_at_50pct"] == pytest.approx(0.3)
    assert statistics["failure_rate_at_50pct"] == pytest.approx(1 / 3)


def test_confirmation_scorer_rejects_outcomes_before_model_execution() -> None:
    config = load_auditor_config(Path("configs/conservative-auditor-v2.json"))
    leaked = pd.DataFrame({"selection_gain_sd": [0.5]})

    with pytest.raises(ValueError, match="forbidden columns"):
        build_outcome_free_task_frame(leaked, config)


def test_confirmation_coverage_is_ranked_within_pool_and_panel() -> None:
    frame = pd.DataFrame(
        {
            "panel_id": ["a", "a", "a", "b"],
            "assay_id": ["1", "2", "3", "1"],
            "target_id": ["1", "2", "3", "4"],
            "auditor_confidence": [0.4, 0.3, 0.2, 0.1],
        }
    )

    ranked, pooled_count, panel_counts = add_confirmation_coverage_decisions(frame)

    assert pooled_count == 2
    assert panel_counts == {"a": 2, "b": 1}
    assert (ranked["pooled_decision_at_50pct_coverage"] == "deploy_vespag").sum() == 2
    assert (
        ranked["panel_decision_at_50pct_coverage"] == "deploy_vespag"
    ).sum() == 3


def test_registered_confirmation_evaluator_is_domainome_venus_only(tmp_path) -> None:
    source_config = load_auditor_config(Path("configs/conservative-auditor-v2.json"))
    payload = dict(source_config.payload)
    payload["model_ids"] = ["vespag", "model_b"]
    payload["bootstrap_repeats"] = 25
    config_path = tmp_path / "auditor.json"
    config_path.write_text(json.dumps(payload))
    final_freeze = tmp_path / "final-freeze.json"
    final_freeze.write_text(
        json.dumps({"protocol_id": "variantshift-confirmation-freeze-v2"})
    )
    target = tmp_path / "targets.csv"
    target.write_text("target_id,sequence\nT0,ACDEFGHIKL\n")

    decision_rows = []
    outcome_rows = []
    prediction_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    pooled_rank = 0
    for panel_id in CONFIRMATION_V2_PANELS:
        for panel_rank in range(1, 5):
            pooled_rank += 1
            target_id = f"{panel_id}-T{panel_rank}"
            decision_rows.append(
                {
                    "protocol_id": "variantshift-confirmation-freeze-v2",
                    "panel_id": panel_id,
                    "dataset_id": panel_id,
                    "assay_id": target_id,
                    "target_id": target_id,
                    "protein_id": f"P{panel_rank % 2}",
                    "family_id": f"F{panel_rank % 2}",
                    "baseline_model": "vespag",
                    "auditor_confidence": float(5 - panel_rank),
                    "pooled_confidence_rank": pooled_rank,
                    "panel_confidence_rank": panel_rank,
                }
            )
            for variant in range(10):
                variant_id = f"A{variant + 1}C"
                outcome_rows.append(
                    {
                        "protocol_id": "variantshift-confirmation-freeze-v2",
                        "panel_id": panel_id,
                        "dataset_id": panel_id,
                        "assay_id": target_id,
                        "target_id": target_id,
                        "variant_id": variant_id,
                        "effect": float(variant),
                        "direction": 1,
                    }
                )
                for model_id in payload["model_ids"]:
                    score = float(variant)
                    if model_id == "vespag" and panel_rank > 2:
                        score = -score
                    prediction_rows.setdefault((panel_id, model_id), []).append(
                        {
                            "target_id": target_id,
                            "variant_id": variant_id,
                            "score": score,
                            "status": "ok",
                        }
                    )
    decisions = tmp_path / "decisions.csv"
    pd.DataFrame(decision_rows).to_csv(decisions, index=False)
    outcomes = tmp_path / "outcomes.csv"
    pd.DataFrame(outcome_rows).to_csv(outcomes, index=False)
    registry_rows = []
    prediction_paths = []
    for (panel_id, model_id), rows in prediction_rows.items():
        path = tmp_path / f"{panel_id}-{model_id}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        prediction_paths.append(path)
        registry_rows.append(
            {
                "panel_id": panel_id,
                "model_id": model_id,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
            }
        )
    registry = tmp_path / "registry.csv"
    pd.DataFrame(registry_rows).to_csv(registry, index=False)
    lock = tmp_path / "outcome-lock.json"
    create_outcome_lock(lock, protocol_id="variantshift-confirmation-freeze-v2", target_artifacts=[target])
    freeze_predictions(
        lock,
        prediction_artifacts=[decisions, registry, *prediction_paths],
        method_artifacts=[config_path, final_freeze],
    )
    register_confirmation(lock, registration_uri="https://osf.io/example")
    record_outcome_reveal(lock, outcome_artifacts=[outcomes])

    outputs = evaluate_conservative_confirmation(
        config_path,
        final_freeze,
        decisions,
        registry,
        outcomes,
        lock,
        tmp_path / "evaluation",
    )
    repeated = evaluate_conservative_confirmation(
        config_path,
        final_freeze,
        decisions,
        registry,
        outcomes,
        lock,
        tmp_path / "evaluation-repeat",
    )

    acceptance = json.loads(outputs["acceptance"].read_text())
    panels = pd.read_csv(outputs["panel_summary"])
    assert set(panels["panel_id"]) == set(CONFIRMATION_V2_PANELS)
    assert {gate["gate"] for gate in acceptance["gates"]} == {
        "primary_regret_coverage",
        "failure_risk_noninferiority",
        "mean_utility_noninferiority_at_50pct",
        "domainome_venus_direction_consistency",
        "no_post_reveal_refit",
    }
    assert acceptance["gates"][-1]["passed"] is True
    assert json.loads(outputs["manifest"].read_text())["method_refit"] is False
    assert sha256_file(outputs["acceptance"]) == sha256_file(repeated["acceptance"])
    assert sha256_file(outputs["bootstrap"]) == sha256_file(repeated["bootstrap"])


def test_registered_confirmation_evaluator_fails_closed_before_reveal(tmp_path) -> None:
    target = tmp_path / "targets.csv"
    decisions = tmp_path / "decisions.csv"
    registry = tmp_path / "registry.csv"
    config = tmp_path / "config.json"
    final_freeze = tmp_path / "final.json"
    outcomes = tmp_path / "outcomes.csv"
    for path in [target, decisions, registry, config, final_freeze, outcomes]:
        path.write_text("placeholder")
    lock = tmp_path / "outcome-lock.json"
    create_outcome_lock(lock, protocol_id="variantshift-confirmation-freeze-v2", target_artifacts=[target])
    freeze_predictions(
        lock,
        prediction_artifacts=[decisions, registry],
        method_artifacts=[config, final_freeze],
    )

    with pytest.raises(PermissionError, match="recorded one-time outcome reveal"):
        evaluate_conservative_confirmation(
            config,
            final_freeze,
            decisions,
            registry,
            outcomes,
            lock,
            tmp_path / "evaluation",
        )
