import json
from pathlib import Path

import pandas as pd

from variantshift.venus_panel import (
    freeze_venusmuthub_targets,
    normalize_doi,
    parse_venus_source_identifier,
)


def test_venus_identifier_and_doi_normalization() -> None:
    assert parse_venus_source_identifier("vb1432_1AAR_ddG") == ("pdb", "1AAR")
    assert parse_venus_source_identifier("P00813_activity") == ("uniprot", "P00813")
    assert parse_venus_source_identifier("BindingDB_P00519_13216_Kd") == (
        "uniprot",
        "P00519",
    )
    assert parse_venus_source_identifier("PPB_Affinity_1A22_A_kd") == (
        "pdb_chain",
        "1A22:A",
    )
    assert normalize_doi("http://dx.doi.org/10.1/ABC") == "10.1/abc"


def test_venus_freeze_uses_only_tree_doi_and_external_sequences(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"target_seq": ["M" * 20], "jo": ["10.1/old"]}).to_csv(
        reference, index=False
    )
    mavedb = tmp_path / "mavedb.json"
    mavedb.write_text(json.dumps([]))
    tree = [
        {"path": "single_mutant/activity/P00813_activity.csv"},
        {"path": "single_mutant/stability/1AAR_ddG.csv"},
    ]
    doi = pd.DataFrame(
        {
            "mutant_file_id": ["P00813_activity", "1AAR_ddG"],
            "doi": ["10.1/new", "10.1/old"],
        }
    )
    resolutions = pd.DataFrame(
        [
            {
                "source_type": "uniprot",
                "source_identifier": "P00813",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "sequence_status": "resolved",
                "protein_polymer_sequences": 1,
                "reference_accessions": "P00813",
            },
            {
                "source_type": "pdb",
                "source_identifier": "1AAR",
                "sequence": "C" * 20,
                "sequence_status": "resolved",
                "protein_polymer_sequences": 1,
                "reference_accessions": "",
            },
        ]
    )
    outputs = freeze_venusmuthub_targets(
        reference,
        mavedb,
        tmp_path / "out",
        tree_rows=tree,
        doi_frame=doi,
        resolutions=resolutions,
    )
    targets = pd.read_csv(outputs["targets"])
    audit = pd.read_csv(outputs["assay_audit"]).fillna("")
    assert len(targets) == 1
    assert audit["selected"].sum() == 1
    protocol = json.loads(outputs["protocol"].read_text())
    assert protocol["mutation_file_requests"] == 0
    assert protocol["outcomes_accessed"] is False
