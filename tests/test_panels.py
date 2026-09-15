import json
import shutil
from pathlib import Path

import pandas as pd

from variantshift.outcome_lock import read_outcome_lock
from variantshift.panels import freeze_panel
from variantshift.provenance import sha256_file


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


def test_project_local_panel_freeze_survives_relocation(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr("variantshift.panels.git_revision", lambda _: "test-source")
    Path("targets.csv").write_text("target_id,sequence\nT1,ACD\n")
    Path("receipt.json").write_text('{"outcomes_accessed": false}\n')
    Path("panel.json").write_text(json.dumps({
        "protocol_id": "confirmation-v1",
        "panel_id": "test-panel",
        "source": "public target fixture",
        "source_version": "1",
        "target_input": "targets.csv",
        "source_artifacts": [str(root / "receipt.json")],
    }))
    freeze_panel(Path("panel.json"), Path("protocol"))

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    monkeypatch.chdir(relocated)
    shutil.rmtree(root)
    protocol = json.loads(Path("protocol/protocol.json").read_text())
    assert protocol["source_artifact_sha256"] == {
        "receipt.json": sha256_file(Path("receipt.json"))
    }
    lock = read_outcome_lock(Path("protocol/outcome-lock.json"))
    for name, digest in lock["target_artifacts"].items():
        assert not Path(name).is_absolute()
        assert sha256_file(Path(name)) == digest
