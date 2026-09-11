import json
from pathlib import Path

import pandas as pd
import pytest

from variantshift.prediction_freeze import _verified_execution
from variantshift.provenance import sha256_file


def test_verified_execution_rejects_outcome_access(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.csv"
    audit = tmp_path / "prediction-audit.csv"
    provenance = tmp_path / "model-provenance.json"
    pd.DataFrame(
        {
            "protocol_id": ["p"],
            "panel_id": ["panel"],
            "target_id": ["t"],
            "variant_id": ["A1C"],
            "model_id": ["m"],
            "model_version": ["1"],
            "score": [0.0],
            "status": ["ok"],
        }
    ).to_csv(prediction, index=False)
    pd.DataFrame({"target_id": ["t"]}).to_csv(audit, index=False)
    provenance.write_text(json.dumps({"m": {}}))
    manifest = {
        "confirmation_outcomes_accessed": True,
        "artifacts": {
            prediction.name: sha256_file(prediction),
            audit.name: sha256_file(audit),
            provenance.name: sha256_file(provenance),
        },
    }
    (tmp_path / "execution-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="accessed confirmation outcomes"):
        _verified_execution(tmp_path, "m")
