import hashlib
import json

import pandas as pd

from variantshift.official_predictions import audit_official_prediction_panel
from variantshift.provenance import sha256_file


def test_official_panel_gate_is_distinct_from_execution(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv.gz"
    model_columns = [f"score_{index}" for index in range(8)]
    rows = []
    for target_index in range(300):
        for variant_index in range(2):
            rows.append(
                {
                    "target_id": f"T{target_index}",
                    "variant_id": f"A{variant_index + 1}C",
                    **{column: 0.1 for column in model_columns},
                }
            )
    pd.DataFrame(rows).to_csv(predictions, index=False)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "outcomes_accessed": False,
                "prediction_sha256": sha256_file(predictions),
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": f"m{index}",
                        "source_column": column,
                        "family": f"f{index % 4}",
                        "modalities": ["sequence"],
                        "source_url": "https://example.org",
                        "license_name": "CC-BY-4.0",
                        "license_status": "permitted",
                    }
                    for index, column in enumerate(model_columns)
                ]
            }
        )
    )
    output = tmp_path / "audit.csv"
    audit = audit_official_prediction_panel(config, predictions, receipt, output)
    assert audit["official_score_gate_passed"].all()
    assert not audit["executable_model_gate_passed"].any()
    assert audit["official_shared_target_count"].iat[0] == 300
    digest = hashlib.sha256(
        "\n".join(sorted(f"T{i}" for i in range(300))).encode()
    ).hexdigest()
    assert audit["official_shared_targets_sha256"].iat[0] == digest
