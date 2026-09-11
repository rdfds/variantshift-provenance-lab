"""Outcome-blind audit of completed model-panel execution artifacts.

This module deliberately does not perform parity, repeatability, or container qualification. It
only establishes that named model configurations produced schema-valid predictions with sufficient
shared target coverage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_adapters import load_model_specifications, prediction_cache_key
from .schemas import PREDICTION_SCHEMA, VARIANT_SCHEMA, validate_targets, write_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_path(directory: Path) -> Path:
    candidates = [directory / "predictions.csv.gz", directory / "predictions.csv"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one predictions.csv[.gz] in {directory}, found {matches}"
        )
    return matches[0]


def audit_executed_panel(
    execution_config_path: Path,
    targets_path: Path,
    variants_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Verify execution coverage and write an explicit non-qualification receipt."""
    execution_config_path = Path(execution_config_path)
    repository = execution_config_path.resolve().parents[1]
    config = json.loads(execution_config_path.read_text(encoding="utf-8"))
    if config.get("scope") != "execution-only":
        raise ValueError("Execution audit config must declare scope=execution-only")
    if config.get("qualification_status") != "not_started":
        raise ValueError("Execution audit cannot be used for a qualification-stage config")

    targets = pd.read_csv(targets_path)
    variants = pd.read_csv(variants_path)
    validate_targets(targets)
    VARIANT_SCHEMA.validate(variants)
    expected_counts = variants.groupby("target_id")["variant_id"].nunique()
    expected_variant_keys = variants.loc[:, ["panel_id", "target_id", "variant_id"]]
    target_universe = set(targets["target_id"].astype(str))

    model_config_path = repository / config["model_config"]
    specifications = {
        item.model_id: item for item in load_model_specifications(model_config_path)
    }
    model_rows: list[dict[str, object]] = []
    passing_targets: dict[str, set[str]] = {}
    forbidden_outcome_columns = {"effect", "direction", "score_set_id", "assay_id"}

    for entry in config["models"]:
        model_id = str(entry["model_id"])
        specification = specifications.get(model_id)
        if specification is None:
            raise ValueError(f"Execution model {model_id} is absent from {model_config_path}")
        artifact_dir = repository / entry["artifact_dir"]
        prediction_path = _prediction_path(artifact_dir)
        provenance_path = artifact_dir / "model-provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Missing model provenance for {model_id}: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if model_id not in provenance:
            raise ValueError(f"Model provenance {provenance_path} has no {model_id} record")
        executed_specification_digest = provenance[model_id].get("specification_sha256")
        predictions = pd.read_csv(prediction_path)
        PREDICTION_SCHEMA.validate(predictions)
        forbidden = sorted(forbidden_outcome_columns.intersection(predictions.columns))
        if forbidden:
            raise ValueError(f"Prediction artifact {prediction_path} contains outcomes: {forbidden}")
        observed_models = set(predictions["model_id"].astype(str))
        if observed_models != {model_id}:
            raise ValueError(
                f"Prediction artifact for {model_id} contains model ids {sorted(observed_models)}"
            )
        observed_protocols = set(predictions["protocol_id"].astype(str))
        if observed_protocols != {str(config["protocol_id"])}:
            raise ValueError(
                f"Prediction artifact for {model_id} contains protocol ids "
                f"{sorted(observed_protocols)}"
            )
        observed_targets = set(predictions["target_id"].astype(str))
        unexpected_targets = observed_targets.difference(target_universe)
        if unexpected_targets:
            raise ValueError(
                f"Prediction artifact for {model_id} has unexpected targets: "
                f"{sorted(unexpected_targets)[:3]}"
            )
        unexpected_variants = (
            predictions.loc[:, ["panel_id", "target_id", "variant_id"]]
            .merge(
                expected_variant_keys,
                on=["panel_id", "target_id", "variant_id"],
                how="left",
                indicator=True,
            )["_merge"]
            .eq("left_only")
        )
        if unexpected_variants.any():
            raise ValueError(
                f"Prediction artifact for {model_id} contains "
                f"{int(unexpected_variants.sum())} unexpected variant keys"
            )
        ok = predictions.loc[
            predictions["status"].eq("ok") & predictions["score"].notna()
        ]
        ok_counts = ok.groupby("target_id")["variant_id"].nunique()
        coverage = ok_counts.div(expected_counts).fillna(0.0)
        minimum_coverage = float(config["minimum_target_coverage"])
        passing = set(coverage.loc[coverage.ge(minimum_coverage)].index.astype(str))
        passing_targets[model_id] = passing
        execution_complete = len(passing) >= int(config["minimum_shared_targets"])
        model_rows.append(
            {
                "model_id": model_id,
                "family": specification.family,
                "prediction_path": str(prediction_path.relative_to(repository)),
                "prediction_sha256": _sha256(prediction_path),
                "provenance_sha256": _sha256(provenance_path),
                "prediction_rows": len(predictions),
                "targets_at_minimum_coverage": len(passing),
                "minimum_target_coverage": minimum_coverage,
                "execution_status": "complete" if execution_complete else "incomplete",
                "executed_specification_sha256": executed_specification_digest,
                "current_specification_sha256": specification.digest(),
                "specification_digest_matches": (
                    executed_specification_digest == specification.digest()
                ),
                "qualification_status": "not_started",
            }
        )

    model_audit = pd.DataFrame(model_rows).sort_values("model_id").reset_index(drop=True)
    complete_ids = model_audit.loc[
        model_audit["execution_status"].eq("complete"), "model_id"
    ].tolist()
    shared = (
        set.intersection(*(passing_targets[model_id] for model_id in complete_ids))
        if complete_ids
        else set()
    )
    shared_targets = pd.DataFrame({"target_id": sorted(shared)})
    family_count = int(
        model_audit.loc[model_audit["execution_status"].eq("complete"), "family"].nunique()
    )
    configuration_count = len(complete_ids)
    gates = {
        "configurations": configuration_count >= int(config["minimum_configurations"]),
        "families": family_count >= int(config["minimum_families"]),
        "shared_targets": len(shared) >= int(config["minimum_shared_targets"]),
    }
    summary = {
        "schema_version": 1,
        "scope": "execution-only",
        "protocol_id": config["protocol_id"],
        "execution_status": "complete" if all(gates.values()) else "incomplete",
        "executed_configurations": configuration_count,
        "executed_families": family_count,
        "shared_targets_at_minimum_coverage": len(shared),
        "minimums": {
            "configurations": int(config["minimum_configurations"]),
            "families": int(config["minimum_families"]),
            "shared_targets": int(config["minimum_shared_targets"]),
            "target_coverage": float(config["minimum_target_coverage"]),
        },
        "gates": gates,
        "outcomes_accessed": False,
        "qualification_status": "not_started",
        "qualification_note": (
            "Parity, repeated-run determinism, pinned-container reproducibility, and checkpoint "
            "qualification have not been evaluated by this command."
        ),
        "inputs": {
            "execution_config_sha256": _sha256(execution_config_path),
            "model_config_sha256": _sha256(model_config_path),
            "targets_sha256": _sha256(Path(targets_path)),
            "variants_sha256": _sha256(Path(variants_path)),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = write_table(model_audit, output_dir / "execution-model-audit.csv")
    shared_path = write_table(shared_targets, output_dir / "shared-targets.csv")
    summary_path = output_dir / "execution-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_audit": model_path,
        "shared_targets": shared_path,
        "summary": summary_path,
    }


def audit_prediction_cache(
    model_config_path: Path,
    model_id: str,
    targets_path: Path,
    variants_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Map valid content-addressed target caches and freeze the remaining worklist."""
    specifications = {
        item.model_id: item for item in load_model_specifications(model_config_path)
    }
    if model_id not in specifications:
        raise ValueError(f"Unknown model id: {model_id}")
    specification = specifications[model_id]
    targets = pd.read_csv(targets_path)
    variants = pd.read_csv(variants_path)
    validate_targets(targets)
    VARIANT_SCHEMA.validate(variants)
    model_cache = Path(cache_dir) / model_id
    audit_rows: list[dict[str, object]] = []
    for _, target in targets.sort_values("target_id").iterrows():
        target_variants = variants.loc[
            variants["target_id"].astype(str).eq(str(target["target_id"]))
        ].copy()
        key = prediction_cache_key(specification, target, target_variants)
        path = model_cache / f"{key}.csv"
        status = "missing"
        error = ""
        cached_rows = 0
        if path.is_file():
            try:
                cached = pd.read_csv(path)
                required = {"variant_id", "score"}
                missing = sorted(required.difference(cached.columns))
                if missing:
                    raise ValueError(f"missing columns {missing}")
                if cached["variant_id"].duplicated().any():
                    raise ValueError("duplicate variant ids")
                expected_ids = set(target_variants["variant_id"].astype(str))
                observed_ids = set(cached["variant_id"].astype(str))
                if observed_ids != expected_ids:
                    raise ValueError(
                        f"variant set mismatch: expected {len(expected_ids)}, "
                        f"found {len(observed_ids)}"
                    )
                numeric_scores = pd.to_numeric(cached["score"], errors="coerce")
                if not np.isfinite(numeric_scores).all():
                    raise ValueError("non-finite scores")
                cached_rows = len(cached)
                status = "valid"
            except Exception as exception:  # noqa: BLE001 - record corrupt external cache
                status = "invalid"
                error = f"{type(exception).__name__}: {exception}"
        audit_rows.append(
            {
                "model_id": model_id,
                "target_id": str(target["target_id"]),
                "sequence_sha256": str(target["sequence_sha256"]),
                "cache_key": key,
                "cache_path": str(path),
                "expected_rows": len(target_variants),
                "cached_rows": cached_rows,
                "status": status,
                "error": error,
            }
        )
    audit = pd.DataFrame(audit_rows)
    remaining = audit.loc[audit["status"].ne("valid"), ["target_id", "cache_key", "status"]]
    summary = {
        "schema_version": 1,
        "scope": "outcome-blind-prediction-cache",
        "model_id": model_id,
        "model_specification_sha256": specification.digest(),
        "target_count": len(audit),
        "valid_cached_targets": int(audit["status"].eq("valid").sum()),
        "invalid_cached_targets": int(audit["status"].eq("invalid").sum()),
        "remaining_targets": len(remaining),
        "outcomes_accessed": False,
        "qualification_status": "not_started",
        "inputs": {
            "model_config_sha256": _sha256(Path(model_config_path)),
            "targets_sha256": _sha256(Path(targets_path)),
            "variants_sha256": _sha256(Path(variants_path)),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = write_table(audit, output_dir / "prediction-cache-audit.csv")
    remaining_path = write_table(remaining, output_dir / "remaining-targets.csv")
    summary_path = output_dir / "prediction-cache-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"audit": audit_path, "remaining": remaining_path, "summary": summary_path}
