import json
from pathlib import Path

import pandas as pd

from variantshift.final_panel import _verified_receipt
from variantshift.provenance import sha256_file


def test_verified_receipt_rejects_artifact_drift(tmp_path: Path) -> None:
    audit = tmp_path / "model-qualification-audit.csv"
    shared = tmp_path / "qualified-shared-targets.csv"
    pd.DataFrame({"model_id": ["m"], "qualification_status": ["passed"]}).to_csv(
        audit, index=False
    )
    pd.DataFrame({"target_id": ["t"]}).to_csv(shared, index=False)
    summary = {
        "qualification_status": "passed",
        "artifacts": {audit.name: sha256_file(audit), shared.name: sha256_file(shared)},
    }
    (tmp_path / "qualification-summary.json").write_text(json.dumps(summary))
    _verified_receipt(tmp_path)
    audit.write_text("drift")
    try:
        _verified_receipt(tmp_path)
    except ValueError as exception:
        assert "drifted" in str(exception)
    else:
        raise AssertionError("Artifact drift must fail closed")
