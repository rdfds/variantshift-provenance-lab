import pandas as pd

from variantshift.external_validation import (
    ExternalSelectionCriteria,
    calibration_orientation,
    canonicalize_mavedb_scores,
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
