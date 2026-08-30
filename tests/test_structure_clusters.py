import pandas as pd

from variantshift.structure_clusters import (
    FOLDSEEK_COLUMNS,
    _combine_families,
    reciprocal_structure_pairs,
)


def _directed_structure_matrix() -> pd.DataFrame:
    proteins = ["P1", "P2", "P3", "P4"]
    rows = []
    for query in proteins:
        for target in proteins:
            identity = query == target
            values = {
                "query_uniprot_id": query,
                "target_uniprot_id": target,
                "query_tm_score": 1.0 if identity else 0.2,
                "target_tm_score": 1.0 if identity else 0.2,
                "alignment_tm_score": 1.0 if identity else 0.2,
                "query_coverage": 1.0 if identity else 0.4,
                "target_coverage": 1.0 if identity else 0.4,
                "lddt": 1.0 if identity else 0.3,
                "homology_probability": 1.0 if identity else 0.1,
                "e_value": 0.0 if identity else 1.0,
                "bit_score": 100.0 if identity else 5.0,
                "alignment_length": 100 if identity else 40,
                "sequence_identity": 1.0 if identity else 0.1,
            }
            if {query, target} == {"P1", "P2"}:
                values.update(
                    {
                        "query_tm_score": 0.82,
                        "target_tm_score": 0.80,
                        "query_coverage": 0.92,
                        "target_coverage": 0.90,
                        "lddt": 0.75,
                        "homology_probability": 0.98,
                    }
                )
            if {query, target} == {"P2", "P3"}:
                values.update(
                    {
                        "query_tm_score": 0.85,
                        "target_tm_score": 0.84,
                        "query_coverage": 0.95,
                        "target_coverage": 0.94,
                        "lddt": 0.80,
                        "homology_probability": 0.97 if query == "P2" else 0.91,
                    }
                )
            rows.append(values)
    return pd.DataFrame(rows, columns=FOLDSEEK_COLUMNS)


def test_combined_families_require_reciprocal_structure_evidence():
    proteins = ["P1", "P2", "P3", "P4"]
    pairs = reciprocal_structure_pairs(_directed_structure_matrix(), proteins)
    sequence = pd.DataFrame(
        {
            "uniprot_id": proteins,
            "family_id": ["S1", "S2", "S34", "S34"],
            "family_size": [1, 1, 2, 2],
            "family_members": ["P1", "P2", "P3;P4", "P3;P4"],
            "assay_count": [1, 1, 1, 1],
            "assayed_sequence_count": [1, 1, 1, 1],
        }
    )

    assignments, audited = _combine_families(
        sequence,
        pairs,
        minimum_tm_score=0.50,
        minimum_coverage=0.80,
        minimum_homology_probability=0.95,
    )

    families = assignments.set_index("uniprot_id")["family_id"]
    assert families["P1"] == families["P2"]
    assert families["P2"] != families["P3"]
    assert families["P3"] == families["P4"]
    assert assignments["family_id"].nunique() == 2
    p2_p3 = audited.loc[audited["protein_a"].eq("P2") & audited["protein_b"].eq("P3")].iloc[0]
    assert p2_p3["reciprocal_minimum_homology_probability"] == 0.91
    assert not p2_p3["qualifies_structure_edge"]
    assert not (
        audited["qualifies_structure_edge"]
        & audited["protein_a_family_id"].ne(audited["protein_b_family_id"])
    ).any()


def test_reciprocal_pairs_reject_an_incomplete_matrix():
    incomplete = _directed_structure_matrix().iloc[:-1]
    try:
        reciprocal_structure_pairs(incomplete, ["P1", "P2", "P3", "P4"])
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("An incomplete exhaustive structure search was accepted")
