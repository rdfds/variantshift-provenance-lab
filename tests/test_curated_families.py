import pandas as pd

from variantshift.curated_families import (
    cluster_from_curated_domains,
    map_assayed_region,
)


def test_assayed_region_maps_through_construct_offset_and_substitution():
    target = "XXACDEYGHIYY"
    canonical = "ZZZZACDEFGHIQQ"
    mapping = map_assayed_region(target, canonical, 3, 10)

    assert mapping["assayed_residues"] == 8
    assert mapping["mapped_residues"] == 7
    assert mapping["mapping_coverage"] == 0.875
    assert mapping["canonical_start"] == 5
    assert mapping["canonical_end"] == 12


def test_shared_curated_clan_is_a_sensitivity_not_primary_family_union():
    base = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3", "P4"],
            "family_id": ["B1", "B2", "B34", "B34"],
            "family_size": [1, 1, 2, 2],
            "family_members": ["P1", "P2", "P3;P4", "P3;P4"],
        }
    )
    overlaps = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3"],
            "pfam_accession": ["PF1", "PF2", "PF3"],
            "clan_accession": ["CL1", "CL1", None],
            "qualifies_curated_domain": [True, True, False],
        }
    )

    primary, primary_edges = cluster_from_curated_domains(base, overlaps)
    assignments, edges = cluster_from_curated_domains(base, overlaps, grouping="pfam_clan")

    primary_families = primary.set_index("uniprot_id")["family_id"]
    assert primary_families["P1"] != primary_families["P2"]
    assert primary_edges.empty
    families = assignments.set_index("uniprot_id")["family_id"]
    assert families["P1"] == families["P2"]
    assert families["P2"] != families["P3"]
    assert families["P3"] == families["P4"]
    assert len(edges) == 1
    assert not edges["protein_a_family_id"].ne(edges["protein_b_family_id"]).any()
