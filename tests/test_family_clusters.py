import pandas as pd

from variantshift.family_clusters import cluster_from_alignments


def test_connected_homology_edges_form_deterministic_families():
    segments = pd.DataFrame(
        {
            "assay_id": ["A1", "A1b", "A2", "A3", "A4"],
            "uniprot_id": ["P1", "P1", "P2", "P3", "P4"],
            "sequence_sha256": ["a", "b", "c", "d", "e"],
        }
    )
    alignments = pd.DataFrame(
        [
            {
                "query_uniprot_id": "P1",
                "target_uniprot_id": "P2",
                "sequence_identity": 0.65,
                "query_coverage": 0.95,
                "target_coverage": 0.91,
            },
            {
                "query_uniprot_id": "P2",
                "target_uniprot_id": "P3",
                "sequence_identity": 0.32,
                "query_coverage": 0.85,
                "target_coverage": 0.88,
            },
            {
                "query_uniprot_id": "P3",
                "target_uniprot_id": "P4",
                "sequence_identity": 0.29,
                "query_coverage": 0.99,
                "target_coverage": 0.99,
            },
        ]
    )

    assignments, audited = cluster_from_alignments(
        segments, alignments, identity_threshold=0.30, coverage_threshold=0.80
    )

    family = assignments.set_index("uniprot_id")["family_id"]
    assert family["P1"] == family["P2"] == family["P3"]
    assert family["P4"] != family["P3"]
    assert assignments.set_index("uniprot_id").loc["P1", "family_size"] == 3
    assert (
        audited.loc[audited["qualifies_family_edge"], "query_family_id"].to_numpy()
        == audited.loc[
            audited["qualifies_family_edge"], "target_family_id"
        ].to_numpy()
    ).all()


def test_bidirectional_coverage_is_required_for_family_edge():
    segments = pd.DataFrame(
        {
            "assay_id": ["A1", "A2"],
            "uniprot_id": ["P1", "P2"],
            "sequence_sha256": ["a", "b"],
        }
    )
    alignments = pd.DataFrame(
        [
            {
                "query_uniprot_id": "P1",
                "target_uniprot_id": "P2",
                "sequence_identity": 0.70,
                "query_coverage": 0.95,
                "target_coverage": 0.50,
            }
        ]
    )

    assignments, audited = cluster_from_alignments(
        segments, alignments, identity_threshold=0.30, coverage_threshold=0.80
    )

    assert assignments["family_id"].nunique() == 2
    assert not audited["qualifies_family_edge"].any()
