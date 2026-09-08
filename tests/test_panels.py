import json

import pandas as pd

from variantshift.outcome_lock import read_outcome_lock
from variantshift.panels import freeze_panel


def test_freeze_panel_never_requires_outcomes(tmp_path) -> None:
    targets = tmp_path / "targets.csv"
    pd.DataFrame({"target_id": ["T1"], "protein_id": ["P1"], "sequence": ["ACD"]}).to_csv(
        targets, index=False
    )
    config = tmp_path / "panel.json"
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"outcomes_accessed": false}\n')
    config.write_text(
        json.dumps(
            {
                "protocol_id": "confirmation-v1",
                "panel_id": "test-panel",
                "source": "public target fixture",
                "source_version": "1",
                "adapter": "target_table",
                "target_input": "targets.csv",
                "source_artifacts": ["receipt.json"],
            }
        )
    )
    outputs = freeze_panel(config, tmp_path / "protocol")
    assert len(pd.read_csv(outputs["variants"])) == 57
    assert read_outcome_lock(outputs["outcome_lock"])["state"] == "targets_frozen"
    protocol = json.loads(outputs["protocol"].read_text())
    assert protocol["outcome_status"] == "not_accessed"
    assert str(receipt.resolve()) in protocol["source_artifact_sha256"]
    assert str(receipt.resolve()) in read_outcome_lock(outputs["outcome_lock"])[
        "target_artifacts"
    ]
