import pandas as pd

from variantshift.confirmation_annotations import (
    _reciprocal_foldseek_edges,
    curated_family_id,
)


def test_curated_family_id_prefers_clan_and_has_pfam_fallback() -> None:
    assert curated_family_id("PF00001", "CL0001") == "clan:CL0001"
    assert curated_family_id("PF00002", pd.NA) == "pfam:PF00002"


def test_reciprocal_foldseek_edges_requires_two_passing_directions() -> None:
    rows = []
    for query, target, tm_score in (("a", "b", 0.7), ("b", "a", 0.6), ("a", "c", 0.8)):
        rows.append(
            {
                "query_uniprot_id": query,
                "target_uniprot_id": target,
                "query_tm_score": tm_score,
                "target_tm_score": tm_score,
                "query_coverage": 0.9,
                "target_coverage": 0.9,
                "homology_probability": 0.99,
                "sequence_identity": 0.2,
                "e_value": 1e-6,
            }
        )
    edges = _reciprocal_foldseek_edges(
        pd.DataFrame(rows),
        minimum_tm_score=0.5,
        minimum_coverage=0.8,
        minimum_homology_probability=0.95,
    )
    qualified = edges.set_index(["structure_a", "structure_b"])[
        "qualifies_structure_edge"
    ]
    assert bool(qualified.loc[("a", "b")])
    assert not bool(qualified.loc[("a", "c")])
