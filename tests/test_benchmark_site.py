import json

import pandas as pd

from variantshift.benchmark_site import build_benchmark_site


def test_static_site_copies_and_catalogs_result_tables(tmp_path) -> None:
    table = tmp_path / "risk.csv"
    pd.DataFrame(
        {
            "policy": ["variantshift", "always_best"],
            "coverage": [0.5, 0.5],
            "failure_rate": [0.1, 0.3],
        }
    ).to_csv(table, index=False)
    config = tmp_path / "site.json"
    config.write_text(
        json.dumps(
            {
                "title": "VariantShift",
                "tables": [
                    {
                        "id": "risk",
                        "title": "Risk coverage",
                        "description": "Task-level selective risk.",
                        "kind": "risk_coverage",
                        "path": "risk.csv",
                    }
                ],
            }
        )
    )
    outputs = build_benchmark_site(config, tmp_path / "site")
    catalog = json.loads(outputs["catalog"].read_text())
    assert catalog[0]["rows"] == 2
    assert len(catalog[0]["sha256"]) == 64
    assert "VariantShift" in outputs["index"].read_text()
    assert "failure_rate" in outputs["script"].read_text()
    assert (tmp_path / "site" / catalog[0]["download"]).is_file()
