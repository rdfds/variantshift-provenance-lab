import pandas as pd
import pytest

from variantshift.external_validation import (
    ExternalSelectionCriteria,
    calibration_orientation,
    canonicalize_mavedb_scores,
    evaluate_external_cohort,
    evaluate_external_predictions,
    parse_mave_hgvs_protein_substitution,
    select_metadata_candidates,
)


def test_parse_single_mave_hgvs_protein_effects() -> None:
    assert parse_mave_hgvs_protein_substitution("p.Ala1Cys") == ("A", 1, "C")
    assert parse_mave_hgvs_protein_substitution("p.Asp3Ter") == ("D", 3, "*")
    assert parse_mave_hgvs_protein_substitution("p.Glu4=") == ("E", 4, "E")


def test_metadata_selection_normalizes_a_terminal_stop() -> None:
    row = {
        "urn": "urn:mavedb:test",
        "title": "test",
        "publishedDate": "2026-01-01",
        "modificationDate": "2026-01-02",
        "numVariants": 500,
        "license": {"shortName": "CC0"},
        "targetGenes": [
            {
                "name": "TEST",
                "targetSequence": {"sequenceType": "protein", "sequence": "A" * 40 + "*"},
            }
        ],
    }
    selected = select_metadata_candidates([row], criteria=ExternalSelectionCriteria())
    assert selected.loc[0, "sequence_length"] == 40
    assert selected.loc[0, "terminal_stop_removed"]


def test_calibration_orientation_uses_functional_class_order() -> None:
    high_is_functional = {
        "scoreCalibrations": [
            {
                "functionalClassifications": [
                    {"functionalClassification": "abnormal", "range": [None, 0.4]},
                    {"functionalClassification": "normal", "range": [0.7, None]},
                ]
            }
        ]
    }
    low_is_functional = {
        "scoreCalibrations": [
            {
                "functionalClassifications": [
                    {"functionalClassification": "normal", "range": [None, 1.0]},
                    {"functionalClassification": "abnormal", "range": [2.0, None]},
                ]
            }
        ]
    }
    assert calibration_orientation(high_is_functional) == 1
    assert calibration_orientation(low_is_functional) == -1


def test_canonicalize_collapses_duplicate_protein_consequences() -> None:
    frame = pd.DataFrame(
        {
            "hgvs_pro": ["p.Ala1Cys", "p.Ala1Cys", "p.Cys2Ala", "p.Asp3Ter", "p.Glu4="],
            "score": [0.1, 0.3, 0.8, -1.0, 1.0],
        }
    )
    canonical, audit = canonicalize_mavedb_scores(
        frame,
        sequence="ACDE",
        orientation=1,
    )
    assert canonical["mutation_codes"].tolist() == ["A1C", "C2A"]
    assert canonical.loc[canonical["mutation_codes"].eq("A1C"), "raw_score"].item() == 0.2
    assert audit["single_missense_variants"] == 2
    assert audit["duplicate_protein_consequence_rows"] == 2


def test_evaluate_external_predictions_reports_selection_metrics() -> None:
    frame = pd.DataFrame(
        {
            "DMS_score": [0.0, 1.0, 2.0, 3.0],
            "prediction": [0.1, 0.8, 2.2, 2.9],
        }
    )
    metrics = evaluate_external_predictions(frame)
    assert metrics["spearman"] == 1.0
    assert metrics["top_recall"] == 1.0


def test_external_nested_bootstrap_preserves_protein_balancing() -> None:
    cohort_rows = []
    prediction_rows = []
    for assay_index, protein in enumerate(["P1", "P1", "P2"]):
        urn = f"urn:{assay_index}"
        digest = f"digest{assay_index}"
        for position in range(1, 6):
            mutation = f"A{position}C"
            cohort_rows.append(
                {
                    "urn": urn,
                    "target_name": protein,
                    "protein_id": protein,
                    "sequence_sha256": digest,
                    "mutation_codes": mutation,
                    "position": position,
                    "DMS_score": float(position),
                }
            )
            prediction_rows.append(
                {
                    "sequence_sha256": digest,
                    "mutation_codes": mutation,
                    "masked_marginal": float(position),
                    "wild_type_marginal": float(6 - position),
                }
            )
    outputs = evaluate_external_cohort(
        pd.DataFrame(cohort_rows),
        pd.DataFrame(prediction_rows),
        bootstrap_repeats=100,
        bootstrap_seed=7,
    )
    assay_metrics, protein_metrics, point_summary, intervals, bootstraps = outputs
    assert len(assay_metrics) == 6
    assert protein_metrics["protein_id"].nunique() == 2
    assert point_summary.loc[
        point_summary["model"].eq("masked_marginal"), "mean_spearman"
    ].item() == pytest.approx(1.0)
    assert intervals.loc[
        intervals["model"].eq("masked_marginal"), "success_lower_bound_above_zero"
    ].item()
    assert len(bootstraps) == 100


def test_external_strategy_difference_bootstrap_is_paired() -> None:
    cohort = pd.DataFrame(
        {
            "urn": ["urn:1"] * 6,
            "target_name": ["P1"] * 6,
            "protein_id": ["P1"] * 6,
            "sequence_sha256": ["digest"] * 6,
            "mutation_codes": [f"A{i}C" for i in range(1, 7)],
            "position": list(range(1, 7)),
            "DMS_score": [0.0, 0.3, 0.1, 0.8, 0.5, 1.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "sequence_sha256": ["digest"] * 6,
            "mutation_codes": [f"A{i}C" for i in range(1, 7)],
            "masked_marginal": [0.2, 0.4, 0.3, 0.7, 0.6, 0.9],
            "wild_type_marginal": [0.2, 0.4, 0.3, 0.7, 0.6, 0.9],
        }
    )
    *_, bootstrap = evaluate_external_cohort(
        cohort,
        predictions,
        bootstrap_repeats=100,
        bootstrap_seed=11,
    )
    assert bootstrap["masked_minus_wild_type"].eq(0).all()
