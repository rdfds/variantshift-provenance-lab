import json

import pandas as pd

from variantshift.execution_audit import audit_executed_panel, audit_prediction_cache
from variantshift.model_adapters import load_model_specifications, prediction_cache_key
from variantshift.schemas import sequence_sha256


def test_execution_audit_counts_shared_targets_without_qualifying(tmp_path) -> None:
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    targets = pd.DataFrame(
        [
            {
                "panel_id": "panel",
                "target_id": target_id,
                "protein_id": target_id,
                "sequence": "AC",
                "sequence_sha256": sequence_sha256("AC"),
                "sequence_length": 2,
            }
            for target_id in ("T1", "T2")
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "panel_id": "panel",
                "target_id": target_id,
                "variant_id": variant_id,
                "position": position,
                "reference": reference,
                "alternate": alternate,
            }
            for target_id in ("T1", "T2")
            for variant_id, position, reference, alternate in (
                ("A1C", 1, "A", "C"),
                ("C2A", 2, "C", "A"),
            )
        ]
    )
    targets_path = repository / "targets.csv"
    variants_path = repository / "variants.csv"
    targets.to_csv(targets_path, index=False)
    variants.to_csv(variants_path, index=False)
    model_config = {
        "models": [
            {
                "model_id": model_id,
                "model_version": "1",
                "family": family,
                "modalities": ["sequence"],
                "adapter": "command",
                "source_url": "https://example.org",
                "license_name": "MIT",
                "license_status": "permitted",
            }
            for model_id, family in (("m1", "f1"), ("m2", "f2"))
        ]
    }
    (repository / "configs/model-panel-v1.json").write_text(json.dumps(model_config))
    entries = []
    for model_id in ("m1", "m2"):
        artifact_dir = repository / "artifacts" / model_id
        artifact_dir.mkdir(parents=True)
        predictions = variants.loc[:, ["panel_id", "target_id", "variant_id"]].copy()
        predictions.insert(0, "protocol_id", "protocol")
        predictions["model_id"] = model_id
        predictions["model_version"] = "1"
        predictions["score"] = 0.5
        predictions["status"] = "ok"
        predictions.to_csv(artifact_dir / "predictions.csv.gz", index=False)
        (artifact_dir / "model-provenance.json").write_text(
            json.dumps({model_id: {"specification_sha256": "fixture"}})
        )
        entries.append({"model_id": model_id, "artifact_dir": f"artifacts/{model_id}"})
    execution_config = {
        "schema_version": 1,
        "scope": "execution-only",
        "protocol_id": "protocol",
        "model_config": "configs/model-panel-v1.json",
        "minimum_configurations": 2,
        "minimum_families": 2,
        "minimum_shared_targets": 2,
        "minimum_target_coverage": 0.95,
        "qualification_status": "not_started",
        "models": entries,
    }
    execution_config_path = repository / "configs/executable-panel-v1.json"
    execution_config_path.write_text(json.dumps(execution_config))

    outputs = audit_executed_panel(
        execution_config_path,
        targets_path,
        variants_path,
        repository / "results",
    )

    summary = json.loads(outputs["summary"].read_text())
    assert summary["execution_status"] == "complete"
    assert summary["executed_configurations"] == 2
    assert summary["executed_families"] == 2
    assert summary["shared_targets_at_minimum_coverage"] == 2
    assert summary["outcomes_accessed"] is False
    assert summary["qualification_status"] == "not_started"


def test_prediction_cache_audit_freezes_remaining_targets(tmp_path) -> None:
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "m1",
                        "model_version": "1",
                        "family": "f1",
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
    targets = pd.DataFrame(
        [
            {
                "panel_id": "panel",
                "target_id": target_id,
                "protein_id": target_id,
                "sequence": "AC",
                "sequence_sha256": sequence_sha256("AC"),
                "sequence_length": 2,
            }
            for target_id in ("T1", "T2")
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "panel_id": "panel",
                "target_id": target_id,
                "variant_id": "A1C",
                "position": 1,
                "reference": "A",
                "alternate": "C",
            }
            for target_id in ("T1", "T2")
        ]
    )
    targets_path = tmp_path / "targets.csv"
    variants_path = tmp_path / "variants.csv"
    targets.to_csv(targets_path, index=False)
    variants.to_csv(variants_path, index=False)
    specification = load_model_specifications(model_config)[0]
    first_variants = variants.loc[variants["target_id"].eq("T1")]
    key = prediction_cache_key(specification, targets.iloc[0], first_variants)
    cache = tmp_path / "cache" / "m1"
    cache.mkdir(parents=True)
    pd.DataFrame({"variant_id": ["A1C"], "score": [0.5]}).to_csv(
        cache / f"{key}.csv", index=False
    )

    outputs = audit_prediction_cache(
        model_config,
        "m1",
        targets_path,
        variants_path,
        tmp_path / "cache",
        tmp_path / "results",
    )

    summary = json.loads(outputs["summary"].read_text())
    assert summary["valid_cached_targets"] == 1
    assert summary["remaining_targets"] == 1
    assert pd.read_csv(outputs["remaining"])["target_id"].tolist() == ["T2"]
