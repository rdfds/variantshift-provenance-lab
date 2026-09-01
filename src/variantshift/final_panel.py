"""Combine independent qualification receipts into the frozen primary panel."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .model_adapters import load_model_specifications
from .provenance import sha256_file
from .schemas import write_table


def _verified_receipt(directory: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary_path = directory / "qualification-summary.json"
    audit_path = directory / "model-qualification-audit.csv"
    shared_path = directory / "qualified-shared-targets.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("qualification_status") != "passed":
        raise ValueError(f"Qualification receipt did not pass: {directory}")
    for name, expected in dict(summary.get("artifacts", {})).items():
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Qualification receipt artifact drifted: {artifact}")
    audit = pd.read_csv(audit_path)
    if not audit["qualification_status"].astype(str).eq("passed").all():
        raise ValueError(f"Receipt contains an unqualified model: {directory}")
    shared = pd.read_csv(shared_path)
    return summary, audit, shared


def freeze_final_model_panel(
    qualification_directories: list[Path],
    model_config_paths: list[Path],
    output_dir: Path,
    final_config_path: Path,
    *,
    minimum_configurations: int = 10,
    minimum_families: int = 6,
    minimum_shared_targets: int = 300,
) -> dict[str, Path]:
    """Freeze only configurations supported by complete qualification receipts."""
    receipt_records = [_verified_receipt(Path(path)) for path in qualification_directories]
    audits = pd.concat([record[1] for record in receipt_records], ignore_index=True)
    if audits["model_id"].duplicated().any():
        raise ValueError("Qualification receipts contain duplicate model identifiers")
    qualified_ids = set(audits["model_id"].astype(str))
    specifications = []
    for path in model_config_paths:
        specifications.extend(load_model_specifications(path))
    specifications_by_id = {
        model_id: [item for item in specifications if item.model_id == model_id]
        for model_id in qualified_ids
    }
    missing = sorted(
        model_id for model_id, candidates in specifications_by_id.items() if not candidates
    )
    if missing:
        raise ValueError(f"Qualified model specifications are missing: {missing}")
    selected = []
    audit_by_id = audits.set_index("model_id")
    for model_id in sorted(qualified_ids):
        audit = audit_by_id.loc[model_id]
        candidates = [
            item
            for item in specifications_by_id[model_id]
            if item.checkpoint_sha256 == str(audit.checkpoint_sha256)
            and item.container_digest == str(audit.container_sha256)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Qualified model configuration is ambiguous for {model_id}: "
                f"{len(candidates)} provenance-matched candidates"
            )
        selected.append(candidates[0])
    shared_sets = [set(record[2]["target_id"].astype(str)) for record in receipt_records]
    shared = sorted(set.intersection(*shared_sets))
    families = sorted({item.family for item in selected})
    gates = {
        "configurations": len(selected) >= minimum_configurations,
        "families": len(families) >= minimum_families,
        "shared_confirmation_targets": len(shared) >= minimum_shared_targets,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Final model-panel gates failed: {gates}")
    final_config_path.parent.mkdir(parents=True, exist_ok=True)
    final_config = {
        "schema_version": 1,
        "policy": (
            "Only independently rerun models that passed frozen coverage, ProteinGym parity, "
            "checkpoint, container, provenance, runtime, and failure-reporting gates."
        ),
        "models": [asdict(item) for item in selected],
    }
    final_config_path.write_text(
        json.dumps(final_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audit": output_dir / "model-qualification-audit.csv",
        "shared": output_dir / "qualified-shared-targets.csv",
        "summary": output_dir / "qualification-summary.json",
        "config": final_config_path,
    }
    write_table(audits.sort_values("model_id"), outputs["audit"])
    write_table(pd.DataFrame({"target_id": shared}), outputs["shared"])
    summary = {
        "schema_version": 1,
        "qualification_status": "passed",
        "qualified_configurations": len(selected),
        "qualified_families": len(families),
        "families": families,
        "shared_confirmation_targets": len(shared),
        "gates": gates,
        "thresholds": {
            "minimum_configurations": minimum_configurations,
            "minimum_families": minimum_families,
            "minimum_shared_confirmation_targets": minimum_shared_targets,
        },
        "confirmation_outcomes_accessed": False,
        "source_receipts": {
            str(path): sha256_file(Path(path) / "qualification-summary.json")
            for path in qualification_directories
        },
        "source_model_configs": {
            str(path): sha256_file(path) for path in model_config_paths
        },
        "artifacts": {
            outputs["audit"].name: sha256_file(outputs["audit"]),
            outputs["shared"].name: sha256_file(outputs["shared"]),
            str(final_config_path): sha256_file(final_config_path),
        },
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
