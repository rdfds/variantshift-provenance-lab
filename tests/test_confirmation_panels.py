import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from variantshift.confirmation_panels import (
    DomainomeTargetSource,
    freeze_domainome_targets,
    freeze_mavedb_complement_targets,
)


def _row(urn: str, sequence: str = "ACDEFGHIKLMNPQRSTVWY") -> dict[str, object]:
    return {
        "urn": urn,
        "title": urn,
        "publishedDate": "2026-01-01",
        "modificationDate": "2026-01-02",
        "numVariants": 400,
        "targetGenes": [
            {
                "name": "GENE",
                "targetSequence": {"sequenceType": "protein", "sequence": sequence},
            }
        ],
    }


def test_mavedb_complement_excludes_prior_meta_and_ambiguous_without_scores(
    tmp_path: Path, monkeypatch
) -> None:
    reference = pd.DataFrame(
        {
            "raw_DMS_filename": ["urn_mavedb_00000001-a-1_scores.csv", "other.csv"],
            "jo": ["10.1/overlap", "10.1/other"],
        }
    )
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps({"panel": {"selected_urns": ["urn:mavedb:00000002-a-1"]}})
    )
    registry = [
        _row("urn:mavedb:00000001-a-1"),
        _row("urn:mavedb:00000002-a-1"),
        _row("urn:mavedb:00000003-a-1"),
        _row("urn:mavedb:00000004-a-1"),
        _row("urn:mavedb:00000005-a-1"),
    ]
    details = {
        "urn:mavedb:00000003-a-1": {
            **_row("urn:mavedb:00000003-a-1"),
            "metaAnalyzesScoreSetUrns": [],
            "scoreCalibrations": [
                {
                    "functionalClassifications": [
                        {"functionalClassification": "normal", "range": [1, 2]},
                        {"functionalClassification": "abnormal", "range": [-2, -1]},
                    ]
                }
            ],
        },
        "urn:mavedb:00000004-a-1": {
            **_row("urn:mavedb:00000004-a-1"),
            "metaAnalyzesScoreSetUrns": ["urn:mavedb:source"],
            "scoreCalibrations": [],
        },
        "urn:mavedb:00000005-a-1": {
            **_row("urn:mavedb:00000005-a-1"),
            "metaAnalyzesScoreSetUrns": [],
            "scoreCalibrations": [],
        },
    }
    monkeypatch.setattr(
        "variantshift.confirmation_panels.fetch_score_set_metadata",
        lambda urns: [details[urn] for urn in urns],
    )
    outputs = freeze_mavedb_complement_targets(
        reference_path,
        prior_path,
        tmp_path / "out",
        registry=registry,
    )
    targets = pd.read_csv(outputs["targets"])
    audit = pd.read_csv(outputs["audit"]).fillna("")
    assert len(targets) == 1
    assert audit["selected"].sum() == 1
    assert "proteingym_score_set" in audit.loc[
        audit["urn"].eq("urn:mavedb:00000001-a-1"), "exclusion_reasons"
    ].iat[0]
    assert "meta_analysis_score_set" in audit.loc[
        audit["urn"].eq("urn:mavedb:00000004-a-1"), "exclusion_reasons"
    ].iat[0]
    protocol = json.loads(outputs["protocol"].read_text())
    assert protocol["outcomes_accessed"] is False
    assert protocol["outcome_endpoint_requests"] == 0


def test_domainome_freeze_decodes_only_allowlisted_target_fields(tmp_path: Path) -> None:
    source_bytes = (
        b"dom_ID\tPFAM_ID\twt_seq\tposition\twt_aa\tCDD_description\tresidual\n"
        b"P1_PF00001_1\tPF00001\tACDE\t1\tA\tDO_NOT_DECODE_\xff\t999\n"
        b"P1_PF00001_1\tPF00001\tACDE\t2\tC\tDO_NOT_DECODE_\xfe\t-999\n"
        b"P2_PF00002_1\tPF00002\tFGHI\t1\tF\tDO_NOT_DECODE_\xfd\t0\n"
    )
    outputs = freeze_domainome_targets(
        tmp_path,
        source=DomainomeTargetSource(expected_md5="fixture"),
        stream=BytesIO(source_bytes),
    )
    targets = pd.read_csv(outputs["targets"])
    receipt = json.loads(outputs["receipt"].read_text())
    assert targets["target_id"].tolist() == ["P1_PF00001_1", "P2_PF00002_1"]
    assert targets["pfam_id"].tolist() == ["PF00001", "PF00002"]
    assert set(targets.columns).isdisjoint({"position", "residual", "fitness"})
    assert receipt["decoded_columns"] == ["dom_ID", "PFAM_ID", "wt_seq"]
    assert receipt["outcomes_accessed"] is False
    assert receipt["source_rows"] == 3
    assert receipt["target_count"] == 2
    assert "999" not in outputs["receipt"].read_text()
