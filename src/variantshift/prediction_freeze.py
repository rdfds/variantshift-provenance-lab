"""Validate and select exactly one outcome-blind execution per model and panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model_adapters import load_model_specifications
from .provenance import sha256_file
from .schemas import PREDICTION_SCHEMA, validate_targets, write_table


def _artifact_path(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}.csv.gz", directory / f"{stem}.csv"]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _verified_execution(directory: Path, model_id: str) -> dict[str, object]:
    manifest_path = directory / "execution-manifest.json"
    provenance_path = directory / "model-provenance.json"
    prediction_path = _artifact_path(directory, "predictions")
    audit_path = _artifact_path(directory, "prediction-audit")
    for path in (manifest_path, provenance_path, prediction_path, audit_path):
        if not path.is_file():
            raise ValueError(f"Selected execution artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("confirmation_outcomes_accessed")):
        raise ValueError(f"Selected execution accessed confirmation outcomes: {directory}")
    for name, expected in dict(manifest.get("artifacts", {})).items():
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Selected execution artifact drifted: {artifact}")
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if model_id not in provenance_payload:
        raise ValueError(f"Selected execution has no provenance for {model_id}")
    predictions = pd.read_csv(prediction_path)
    PREDICTION_SCHEMA.validate(predictions)
    if not predictions["model_id"].astype(str).eq(model_id).all():
        raise ValueError(f"Selected execution contains another model: {directory}")
    audit = pd.read_csv(audit_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "provenance": provenance_payload[model_id],
        "provenance_path": provenance_path,
        "predictions": predictions,
        "prediction_path": prediction_path,
        "audit": audit,
        "audit_path": audit_path,
    }


def freeze_selected_predictions(
    selection_config_path: Path,
    model_config_path: Path,
    final_qualification_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Verify the complete panel-by-model execution matrix and freeze intersections."""
    selection = json.loads(selection_config_path.read_text(encoding="utf-8"))
    specifications = {
        item.model_id: item for item in load_model_specifications(model_config_path)
    }
    qualification_summary_path = final_qualification_dir / "qualification-summary.json"
    qualification_summary = json.loads(
        qualification_summary_path.read_text(encoding="utf-8")
    )
    if qualification_summary.get("qualification_status") != "passed":
        raise ValueError("The selected panel lacks a passed final qualification receipt")
    qualification_audit = pd.read_csv(
        final_qualification_dir / "model-qualification-audit.csv"
    ).set_index("model_id")
    panels = selection["panels"]
    default_model_ids = list(map(str, selection.get("model_ids", [])))
    panel_model_ids = {
        str(panel_id): list(map(str, panel.get("model_ids", default_model_ids)))
        for panel_id, panel in panels.items()
    }
    if any(not model_ids for model_ids in panel_model_ids.values()):
        raise ValueError("Every selected panel must define at least one model")
    model_ids = sorted({model for values in panel_model_ids.values() for model in values})
    unknown_models = sorted(set(model_ids).difference(specifications))
    if unknown_models:
        raise ValueError(
            "Selection contains models absent from the frozen final model config: "
            f"{unknown_models}"
        )

    panel_inputs = {}
    for panel_id, panel in panels.items():
        targets_path = Path(panel["targets"])
        variants_path = Path(panel["variants"])
        targets = validate_targets(pd.read_csv(targets_path))
        variants = pd.read_csv(variants_path)
        panel_inputs[str(panel_id)] = {
            "targets_path": targets_path,
            "variants_path": variants_path,
            "targets": targets,
            "variants": variants,
        }

    registry_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    selected_artifacts: list[Path] = []
    for panel_id, panel in panels.items():
        for model_id in panel_model_ids[str(panel_id)]:
            specification = specifications[model_id]
            if model_id not in qualification_audit.index:
                raise ValueError(f"Final qualification audit omits {model_id}")
            qualified = qualification_audit.loc[model_id]
            if str(qualified.qualification_status) != "passed":
                raise ValueError(f"Selected model did not qualify: {model_id}")
            directory = Path(panel["executions"][model_id])
            execution = _verified_execution(directory, model_id)
            provenance = execution["provenance"]
            assert isinstance(provenance, dict)
            if provenance.get("checkpoint_sha256") != specification.checkpoint_sha256:
                raise ValueError(f"Checkpoint mismatch for {panel_id}/{model_id}")
            if provenance.get("container_sha256") != specification.container_digest:
                raise ValueError(f"Container mismatch for {panel_id}/{model_id}")
            if panel_id == "human-domainome-v1":
                expected_hash = str(qualified.prediction_sha256_a)
                if sha256_file(execution["prediction_path"]) != expected_hash:
                    raise ValueError(
                        f"Domainome selection is not qualification run A for {model_id}"
                    )
            predictions = execution["predictions"]
            assert isinstance(predictions, pd.DataFrame)
            inputs = panel_inputs[str(panel_id)]
            variants = inputs["variants"]
            assert isinstance(variants, pd.DataFrame)
            expected = variants.groupby("target_id")["variant_id"].nunique()
            observed = predictions.groupby("target_id")["variant_id"].nunique()
            targets = inputs["targets"]
            assert isinstance(targets, pd.DataFrame)
            for target in targets.sort_values("target_id").itertuples(index=False):
                target_id = str(target.target_id)
                expected_count = int(expected.get(target_id, 0))
                observed_count = int(observed.get(target_id, 0))
                coverage_rows.append(
                    {
                        "panel_id": panel_id,
                        "target_id": target_id,
                        "model_id": model_id,
                        "expected_substitutions": expected_count,
                        "scored_substitutions": observed_count,
                        "coverage": observed_count / max(expected_count, 1),
                        "usable_at_95pct": observed_count / max(expected_count, 1) >= 0.95,
                    }
                )
            manifest = execution["manifest"]
            assert isinstance(manifest, dict)
            registry_rows.append(
                {
                    "panel_id": panel_id,
                    "model_id": model_id,
                    "family": specification.family,
                    "execution_directory": str(directory),
                    "prediction_path": str(execution["prediction_path"]),
                    "prediction_sha256": sha256_file(execution["prediction_path"]),
                    "audit_path": str(execution["audit_path"]),
                    "audit_sha256": sha256_file(execution["audit_path"]),
                    "provenance_path": str(execution["provenance_path"]),
                    "provenance_sha256": sha256_file(execution["provenance_path"]),
                    "execution_manifest_path": str(execution["manifest_path"]),
                    "execution_manifest_sha256": sha256_file(
                        execution["manifest_path"]
                    ),
                    "elapsed_seconds": manifest.get("elapsed_seconds"),
                    "hardware": json.dumps(manifest.get("hardware"), sort_keys=True),
                    "failure_count": len(manifest.get("failures", [])),
                    "outcomes_accessed": False,
                }
            )
            selected_artifacts.extend(
                [
                    execution["prediction_path"],
                    execution["audit_path"],
                    execution["provenance_path"],
                    execution["manifest_path"],
                ]
            )
    registry = pd.DataFrame(registry_rows)
    coverage = pd.DataFrame(coverage_rows)
    expected_pairs = sum(len(values) for values in panel_model_ids.values())
    if len(registry) != expected_pairs:
        raise RuntimeError("The selected execution matrix is incomplete")
    shared_rows = []
    for (panel_id, target_id), group in coverage.groupby(["panel_id", "target_id"]):
        expected_models = len(panel_model_ids[str(panel_id)])
        if len(group) != expected_models:
            raise RuntimeError(f"Coverage matrix is incomplete for {panel_id}/{target_id}")
        shared_rows.append(
            {
                "panel_id": panel_id,
                "target_id": target_id,
                "models_usable": int(group["usable_at_95pct"].sum()),
                "shared_all_models": bool(group["usable_at_95pct"].all()),
                "minimum_model_coverage": float(group["coverage"].min()),
            }
        )
    shared = pd.DataFrame(shared_rows)
    domainome_shared = int(
        shared.loc[
            shared["panel_id"].eq("human-domainome-v1"), "shared_all_models"
        ].sum()
    )
    if domainome_shared < int(selection.get("minimum_domainome_shared_targets", 300)):
        raise RuntimeError("The selected final panel has fewer than 300 shared Domainome targets")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "registry": output_dir / "selected-prediction-registry.csv",
        "coverage": output_dir / "model-target-coverage.csv",
        "shared": output_dir / "shared-model-intersection.csv",
        "manifest": output_dir / "selected-prediction-manifest.json",
    }
    write_table(registry, outputs["registry"])
    write_table(coverage, outputs["coverage"])
    write_table(shared, outputs["shared"])
    manifest = {
        "schema_version": 1,
        "freeze_id": selection["freeze_id"],
        "outcomes_accessed": False,
        "models": model_ids,
        "models_by_panel": panel_model_ids,
        "panels": list(panels),
        "selected_execution_count": len(registry),
        "shared_targets_by_panel": {
            str(panel): int(group["shared_all_models"].sum())
            for panel, group in shared.groupby("panel_id")
        },
        "inputs": {
            "selection_config": sha256_file(selection_config_path),
            "model_config": sha256_file(model_config_path),
            "qualification_summary": sha256_file(qualification_summary_path),
            "panels": {
                panel: {
                    "targets": sha256_file(values["targets_path"]),
                    "variants": sha256_file(values["variants_path"]),
                }
                for panel, values in panel_inputs.items()
            },
        },
        "selected_artifacts": {
            str(path): sha256_file(path) for path in selected_artifacts
        },
        "artifacts": {
            path.name: sha256_file(path)
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
