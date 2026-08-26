from zipfile import ZipFile

import pandas as pd

from variantshift.proteingym import (
    EligibilityCriteria,
    audit_archive,
    canonicalize_assay,
    iter_eligible_assays,
    read_reference_index,
)


def _metadata(**overrides):
    values = {
        "DMS_id": "TEST_HUMAN_Author_2024",
        "DMS_filename": "TEST_HUMAN_Author_2024.csv",
        "UniProt_ID": "TEST_HUMAN",
        "target_seq": "ACDE",
        "taxon": "Human",
        "selection_type": "Binding",
        "coarse_selection_type": "Activity",
        "ProteinGym_version": 1.3,
    }
    values.update(overrides)
    return values


def _write_fixture(tmp_path, assay: pd.DataFrame, metadata: dict | None = None):
    metadata = metadata or _metadata()
    reference_path = tmp_path / "DMS_substitutions.csv"
    pd.DataFrame([metadata]).to_csv(reference_path, index=False)
    assay_path = tmp_path / metadata["DMS_filename"]
    assay.to_csv(assay_path, index=False)
    archive_path = tmp_path / "assays.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(assay_path, f"DMS_ProteinGym_substitutions/{assay_path.name}")
    return archive_path, reference_path


def test_reference_index_rejects_duplicate_identifiers(tmp_path):
    path = tmp_path / "reference.csv"
    pd.DataFrame([_metadata(), _metadata(DMS_filename="other.csv")]).to_csv(
        path, index=False
    )
    try:
        read_reference_index(path)
    except ValueError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("Expected duplicate identifiers to be rejected")


def test_audit_and_iteration_use_predeclared_eligibility(tmp_path):
    assay = pd.DataFrame(
        {
            "mutant": ["A1C", "C2A", "A1D", "A1C:C2A"],
            "mutated_sequence": ["CCDE", "AADE", "DCDE", "CADE"],
            "DMS_score": [0.2, -0.1, 0.4, 0.8],
        }
    )
    archive_path, reference_path = _write_fixture(tmp_path, assay)
    criteria = EligibilityCriteria(
        min_single_variants=3,
        min_positions=2,
        min_unique_scores=3,
    )
    audit = audit_archive(archive_path, reference_path, criteria=criteria)

    assert audit.loc[0, "eligible"]
    assert audit.loc[0, "single_variants"] == 3
    assert audit.loc[0, "multiple_variants"] == 1
    assert audit.loc[0, "reference_mismatches"] == 0
    assert audit.loc[0, "mutated_sequence_mismatches"] == 0

    loaded = list(iter_eligible_assays(archive_path, reference_path, audit))
    assert len(loaded) == 1
    frame, metadata = loaded[0]
    assert frame["mutation_codes"].tolist() == ["A1C", "C2A", "A1D"]
    assert frame["goi_amino_mutations"].eq(1).all()
    assert metadata["DMS_id"] == "TEST_HUMAN_Author_2024"


def test_audit_records_sequence_validation_failure(tmp_path):
    assay = pd.DataFrame(
        {
            "mutant": ["C1A", "C2A"],
            "mutated_sequence": ["ACDE", "AADE"],
            "DMS_score": [0.2, -0.1],
        }
    )
    archive_path, reference_path = _write_fixture(tmp_path, assay)
    audit = audit_archive(
        archive_path,
        reference_path,
        criteria=EligibilityCriteria(
            min_single_variants=1,
            min_positions=1,
            min_unique_scores=1,
        ),
    )

    assert not audit.loc[0, "eligible"]
    assert audit.loc[0, "reference_mismatches"] == 1
    assert "reference_sequence_mismatch" in audit.loc[0, "exclusion_reasons"]


def test_canonicalize_assay_keeps_only_finite_single_substitutions():
    assay = pd.DataFrame(
        {
            "mutant": ["A1C", "A1C:C2A", "C2D"],
            "mutated_sequence": ["CCDE", "CADE", "ADDE"],
            "DMS_score": [0.2, 0.8, float("nan")],
        }
    )
    canonical = canonicalize_assay(assay, pd.Series(_metadata()))
    assert canonical["mutation_codes"].tolist() == ["A1C"]
    assert canonical["assay_id"].unique().tolist() == ["TEST_HUMAN_Author_2024"]
