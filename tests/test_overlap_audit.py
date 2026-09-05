import json

import pandas as pd

from variantshift.overlap_audit import audit_confirmation_overlap
from variantshift.schemas import sequence_sha256


def test_overlap_audit_never_promotes_missing_annotations_to_clean(
    tmp_path, monkeypatch
) -> None:
    reference = tmp_path / "reference.csv"
    eligibility = tmp_path / "eligibility.csv"
    reference.write_text("fixture\n")
    eligibility.write_text("fixture\n")
    targets = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "panel_id": ["confirmation"],
            "target_id": ["C1"],
            "protein_id": ["CP1"],
            "sequence": ["ACDE"],
            "sequence_sha256": [sequence_sha256("ACDE")],
            "sequence_length": [4],
        }
    ).to_csv(targets, index=False)
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "m1",
                        "model_version": "1",
                        "family": "sequence",
                        "modalities": ["sequence"],
                        "adapter": "command",
                        "source_url": "https://example.org",
                        "license_name": "MIT",
                        "license_status": "permitted",
                    }
                ]
            }
        )
    )
    development = pd.DataFrame(
        {
            "panel_id": ["development"],
            "target_id": ["D1"],
            "protein_id": ["DP1"],
            "sequence": ["ACDF"],
            "sequence_sha256": [sequence_sha256("ACDF")],
            "sequence_length": [4],
        }
    )
    alignments = pd.DataFrame(
        {
            "query_panel_id": ["confirmation"],
            "query_target_id": ["C1"],
            "target_panel_id": ["confirmation"],
            "target_target_id": ["C1"],
            "target_source": ["confirmation"],
            "qualifies_family_edge": [True],
        }
    )
    monkeypatch.setattr(
        "variantshift.overlap_audit._development_targets",
        lambda *_: development,
    )
    monkeypatch.setattr(
        "variantshift.overlap_audit._mmseqs_confirmation_search",
        lambda *_, **__: (alignments, "fixture"),
    )
    outputs = audit_confirmation_overlap(
        reference,
        eligibility,
        [targets],
        model_config,
        tmp_path / "output",
    )
    audit = pd.read_csv(outputs["audit"])
    exposure = pd.read_csv(outputs["exposure"])
    assert bool(audit.loc[0, "exact_sequence_unseen"])
    assert bool(audit.loc[0, "mmseqs_family_unseen"])
    assert audit.loc[0, "pfam_clan_status"] == "undocumented"
    assert audit.loc[0, "foldseek_structure_family_status"] == "undocumented"
    assert exposure.loc[0, "exposure_category"] == "undocumented"
    assert json.loads(outputs["summary"].read_text())["outcomes_accessed"] is False
