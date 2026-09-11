import json

import pandas as pd

from variantshift.confirmation_registry import (
    freeze_untouched_confirmation_registry,
    venus_direction,
)


def test_venus_direction_is_metadata_only_and_conservative() -> None:
    assert venus_direction("P12345_km", "activity")[0] == -1
    assert venus_direction("P12345_kcatkm", "activity")[0] == 1
    assert venus_direction("1ABC_ee", "selectivity")[0] is None


def test_untouched_registry_excludes_external_development_pilot(tmp_path) -> None:
    columns = {
        "assay_modality": "activity",
        "direction": 1,
        "direction_source": "metadata",
        "included": True,
        "exclusion_reason": "",
        "publication_ids": "",
        "outcomes_accessed": False,
    }
    full = pd.DataFrame(
        [
            {
                "panel_id": "human-domainome-v1",
                "assay_id": "D1",
                "target_id": "D1",
                **columns,
            },
            {
                "panel_id": "venusmuthub-v1",
                "assay_id": "pilot",
                "target_id": "V0",
                **columns,
            },
        ]
    )
    untouched = pd.DataFrame(
        [
            {
                "panel_id": "mavedb-complement-v1",
                "assay_id": "M1",
                "target_id": "M1",
                **columns,
            },
            {
                "panel_id": "venusmuthub-v1",
                "assay_id": "V1",
                "target_id": "V1",
                **columns,
            },
        ]
    )
    full_path = tmp_path / "full.csv"
    untouched_path = tmp_path / "untouched.csv"
    full.to_csv(full_path, index=False)
    untouched.to_csv(untouched_path, index=False)

    outputs = freeze_untouched_confirmation_registry(
        full_path, untouched_path, tmp_path / "output"
    )

    registry = pd.read_csv(outputs["registry"])
    manifest = json.loads(outputs["manifest"].read_text())
    assert set(registry["assay_id"]) == {"D1", "M1", "V1"}
    assert manifest["outcomes_accessed"] is False
    assert manifest["tasks_by_panel"]["venusmuthub-v1"] == 1
