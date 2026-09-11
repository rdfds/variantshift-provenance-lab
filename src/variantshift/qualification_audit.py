"""Fail-closed audit for independent model qualification executions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .model_adapters import load_model_specifications
from .provenance import sha256_file
from .schemas import PREDICTION_SCHEMA, write_table


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _container_sha256(lock: dict[str, Any], name: str) -> str:
    return _canonical_sha256(
        {"base_image": lock["base_image"], "image": name, **lock["images"][name]}
    )


def _artifact_path(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}.csv.gz", directory / f"{stem}.csv"]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _load_execution(directory: Path, model_id: str) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / "execution-manifest.json"
    provenance_path = directory / "model-provenance.json"
    prediction_path = _artifact_path(directory, "predictions")
    audit_path = _artifact_path(directory, "prediction-audit")
    for path in (manifest_path, provenance_path, prediction_path, audit_path):
        if not path.is_file():
            raise ValueError(f"Qualification artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if model_id not in provenance_payload:
        raise ValueError(f"Provenance does not contain {model_id}: {provenance_path}")
    provenance = provenance_payload[model_id]
    predictions = pd.read_csv(prediction_path)
    audit = pd.read_csv(audit_path)
    PREDICTION_SCHEMA.validate(predictions)
    if not predictions["model_id"].astype(str).eq(model_id).all():
        raise ValueError(f"Prediction artifact contains the wrong model: {prediction_path}")
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Qualification artifact hash mismatch: {path}")
    if int(manifest.get("cache_hit_count", -1)) != 0:
        raise ValueError(f"Qualification manifest reports cache hits: {manifest_path}")
    if not bool(manifest.get("cache_namespace_fresh")):
        raise ValueError(f"Qualification cache was not fresh: {manifest_path}")
    if audit["cache_hit"].astype(bool).any():
        raise ValueError(f"Qualification audit contains cache hits: {audit_path}")
    if bool(manifest.get("confirmation_outcomes_accessed")):
        raise ValueError(f"Qualification execution accessed confirmation outcomes: {manifest_path}")
    return {
        "directory": directory,
        "manifest": manifest,
        "provenance": provenance,
        "predictions": predictions,
        "audit": audit,
        "prediction_path": prediction_path,
        "prediction_sha256": sha256_file(prediction_path),
        "manifest_path": manifest_path,
    }


def _repeatability(first: pd.DataFrame, second: pd.DataFrame) -> tuple[float, int, int]:
    key = ["panel_id", "target_id", "variant_id", "model_id"]
    shared = first.loc[:, [*key, "score"]].merge(
        second.loc[:, [*key, "score"]], on=key, how="inner", suffixes=("_a", "_b")
    )
    finite = np.isfinite(shared["score_a"]) & np.isfinite(shared["score_b"])
    shared = shared.loc[finite]
    correlation = (
        float(spearmanr(shared["score_a"], shared["score_b"]).statistic)
        if len(shared) >= 2
        else float("nan")
    )
    exact_matches = int((shared["score_a"] == shared["score_b"]).sum())
    return correlation, len(shared), exact_matches


def _parity(
    predictions: pd.DataFrame,
    official: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    reference = official.loc[
        official["model_id"].astype(str).eq(model_id),
        ["dms_id", "target_id", "variant_id", "official_score"],
    ]
    shared = reference.merge(
        predictions.loc[:, ["target_id", "variant_id", "score"]],
        on=["target_id", "variant_id"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for (dms_id, target_id), group in shared.groupby(["dms_id", "target_id"], sort=True):
        finite = group.loc[
            np.isfinite(group["official_score"]) & np.isfinite(group["score"])
        ]
        correlation = (
            float(spearmanr(finite["score"], finite["official_score"]).statistic)
            if len(finite) >= 2
            else float("nan")
        )
        rows.append(
            {
                "model_id": model_id,
                "dms_id": str(dms_id),
                "target_id": str(target_id),
                "shared_variants": len(finite),
                "reference_variants": len(group),
                "parity_spearman": correlation,
            }
        )
    return pd.DataFrame(rows)


def audit_model_qualification(
    qualification_config: Path,
    model_config: Path,
    container_lock: Path,
    confirmation_targets: Path,
    confirmation_variants: Path,
    outcome_lock: Path,
    parity_targets: Path,
    parity_variants: Path,
    official_scores: Path,
    run_a_root: Path,
    run_b_root: Path,
    parity_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Verify every execution and emit an all-or-nothing qualification receipt."""
    paths = [
        qualification_config,
        model_config,
        container_lock,
        confirmation_targets,
        confirmation_variants,
        outcome_lock,
        parity_targets,
        parity_variants,
        official_scores,
    ]
    for path in map(Path, paths):
        if not path.is_file():
            raise ValueError(f"Qualification input is missing: {path}")
    qualification = json.loads(Path(qualification_config).read_text(encoding="utf-8"))
    lock = json.loads(Path(container_lock).read_text(encoding="utf-8"))
    outcome = json.loads(Path(outcome_lock).read_text(encoding="utf-8"))
    required_lock = str(qualification["confirmation_lock_required_state"])
    if outcome.get("state") != required_lock:
        raise ValueError(f"Qualification requires outcome lock {required_lock}")
    if outcome.get("registration") is not None or outcome.get("reveal") is not None:
        raise ValueError("Qualification must precede registration and outcome reveal")
    thresholds = qualification["thresholds"]
    specifications = {
        item.model_id: item for item in load_model_specifications(Path(model_config))
    }
    expected_model_ids = [str(item["model_id"]) for item in qualification["models"]]
    if len(expected_model_ids) != len(set(expected_model_ids)):
        raise ValueError("Qualification model identifiers must be unique")
    missing_specifications = sorted(set(expected_model_ids).difference(specifications))
    if missing_specifications:
        raise ValueError(f"Model specifications are missing: {missing_specifications}")
    confirmation_variant_count = len(pd.read_csv(confirmation_variants))
    parity_official = pd.read_csv(official_scores)
    parity_target_count = len(pd.read_csv(parity_targets))
    common_input_hashes = {
        "model_config_sha256": sha256_file(Path(model_config)),
        "qualification_config_sha256": sha256_file(Path(qualification_config)),
        "container_lock_sha256": sha256_file(Path(container_lock)),
    }
    confirmation_input_hashes = {
        **common_input_hashes,
        "targets_sha256": sha256_file(Path(confirmation_targets)),
        "variants_sha256": sha256_file(Path(confirmation_variants)),
        "outcome_lock_sha256": sha256_file(Path(outcome_lock)),
    }
    parity_input_hashes = {
        **common_input_hashes,
        "targets_sha256": sha256_file(Path(parity_targets)),
        "variants_sha256": sha256_file(Path(parity_variants)),
        "outcome_lock_sha256": None,
    }
    if parity_target_count < int(thresholds["minimum_parity_targets"]):
        raise ValueError("Frozen parity panel has too few targets")

    model_rows: list[dict[str, object]] = []
    parity_frames: list[pd.DataFrame] = []
    passing_targets: list[set[str]] = []
    source_tree_hashes: set[object] = set()
    runner_hashes: set[object] = set()
    run_ids = list(map(str, qualification["independent_runs"]))
    parity_run_id = str(qualification.get("parity_run_id", "parity-reference"))
    for model in qualification["models"]:
        model_id = str(model["model_id"])
        specification = specifications[model_id]
        first = _load_execution(Path(run_a_root) / model_id, model_id)
        second = _load_execution(Path(run_b_root) / model_id, model_id)
        parity = _load_execution(Path(parity_root) / model_id, model_id)
        if first["manifest"].get("run_id") != run_ids[0]:
            raise ValueError(f"Unexpected first run id for {model_id}")
        if second["manifest"].get("run_id") != run_ids[1]:
            raise ValueError(f"Unexpected second run id for {model_id}")
        if parity["manifest"].get("run_id") != parity_run_id:
            raise ValueError(f"Unexpected parity run id for {model_id}")
        if first["manifest"].get("dataset") != "domainome" or second["manifest"].get(
            "dataset"
        ) != "domainome":
            raise ValueError(f"Independent runs use the wrong dataset for {model_id}")
        if parity["manifest"].get("dataset") != "parity":
            raise ValueError(f"Parity execution uses the wrong dataset for {model_id}")

        expected_container = _container_sha256(lock, str(model["container"]))
        provenance_records = [
            first["provenance"],
            second["provenance"],
            parity["provenance"],
        ]
        checkpoint_hashes = {item.get("checkpoint_sha256") for item in provenance_records}
        container_hashes = {item.get("container_sha256") for item in provenance_records}
        checkpoint_pass = bool(
            len(checkpoint_hashes) == 1
            and None not in checkpoint_hashes
            and specification.checkpoint_sha256 in checkpoint_hashes
        )
        container_pass = bool(container_hashes == {expected_container})
        specification_pass = all(
            item["manifest"]["inputs"].get("base_specification_sha256")
            == specification.digest()
            for item in (first, second, parity)
        )
        input_hash_pass = all(
            all(
                item["manifest"]["inputs"].get(key) == expected
                for key, expected in expected_hashes.items()
            )
            and bool(item["manifest"]["inputs"].get("source_tree_sha256"))
            and bool(item["manifest"]["inputs"].get("runner_sha256"))
            for item, expected_hashes in (
                (first, confirmation_input_hashes),
                (second, confirmation_input_hashes),
                (parity, parity_input_hashes),
            )
        )
        model_source_hashes = {
            item["manifest"]["inputs"].get("source_tree_sha256")
            for item in (first, second, parity)
        }
        model_runner_hashes = {
            item["manifest"]["inputs"].get("runner_sha256")
            for item in (first, second, parity)
        }
        code_hash_pass = bool(
            len(model_source_hashes) == 1
            and None not in model_source_hashes
            and len(model_runner_hashes) == 1
            and None not in model_runner_hashes
        )
        source_tree_hashes.update(model_source_hashes)
        runner_hashes.update(model_runner_hashes)
        if "structure" in specification.modalities:
            domainome_structure_manifest = Path(str(specification.input_manifest))
            parity_structure_manifest = Path(parity_targets).parent / "structure-manifest.json"
            input_hash_pass = bool(
                input_hash_pass
                and domainome_structure_manifest.is_file()
                and parity_structure_manifest.is_file()
                and first["provenance"].get("input_manifest_sha256")
                == sha256_file(domainome_structure_manifest)
                and second["provenance"].get("input_manifest_sha256")
                == sha256_file(domainome_structure_manifest)
                and parity["provenance"].get("input_manifest_sha256")
                == sha256_file(parity_structure_manifest)
            )
        hardware_pass = all(
            bool(item["manifest"].get("hardware"))
            and float(
                item["manifest"].get(
                    "elapsed_gpu_seconds_sum", item["manifest"].get("elapsed_seconds", 0)
                )
            )
            > 0
            for item in (first, second, parity)
        )
        coverage_a = len(first["predictions"]) / max(confirmation_variant_count, 1)
        coverage_b = len(second["predictions"]) / max(confirmation_variant_count, 1)
        coverage_pass = min(coverage_a, coverage_b) >= float(
            thresholds["minimum_substitution_coverage"]
        )
        repeat_spearman, repeat_rows, exact_repeat_rows = _repeatability(
            first["predictions"], second["predictions"]
        )
        repeat_pass = bool(
            np.isfinite(repeat_spearman)
            and repeat_spearman >= float(thresholds["minimum_repeat_spearman"])
        )
        parity_frame = _parity(parity["predictions"], parity_official, model_id)
        parity_frame["minimum_variants_pass"] = parity_frame["shared_variants"].ge(
            int(thresholds["minimum_parity_variants_per_target"])
        )
        parity_frame["correlation_pass"] = parity_frame["parity_spearman"].ge(
            float(thresholds["minimum_parity_spearman"])
        )
        parity_frame["parity_pass"] = (
            parity_frame["minimum_variants_pass"] & parity_frame["correlation_pass"]
        )
        parity_frames.append(parity_frame)
        parity_pass = bool(
            len(parity_frame) >= int(thresholds["minimum_parity_targets"])
            and parity_frame["parity_pass"].all()
        )
        failures_recorded = all("failures" in item["manifest"] for item in (first, second, parity))
        prediction_hash_pass = all(bool(item["prediction_sha256"]) for item in (first, second, parity))
        qualified = all(
            (
                coverage_pass,
                parity_pass,
                repeat_pass,
                checkpoint_pass,
                container_pass,
                specification_pass,
                input_hash_pass,
                code_hash_pass,
                prediction_hash_pass,
                hardware_pass,
                failures_recorded,
            )
        )
        passing_a = set(
            first["audit"].loc[
                first["audit"]["status"].eq("ok")
                & first["audit"]["coverage"].ge(
                    float(thresholds["minimum_substitution_coverage"])
                ),
                "target_id",
            ].astype(str)
        )
        passing_b = set(
            second["audit"].loc[
                second["audit"]["status"].eq("ok")
                & second["audit"]["coverage"].ge(
                    float(thresholds["minimum_substitution_coverage"])
                ),
                "target_id",
            ].astype(str)
        )
        if qualified:
            passing_targets.append(passing_a & passing_b)
        model_rows.append(
            {
                "model_id": model_id,
                "family": specification.family,
                "coverage_a": coverage_a,
                "coverage_b": coverage_b,
                "coverage_pass": coverage_pass,
                "parity_targets": len(parity_frame),
                "minimum_parity_spearman": parity_frame["parity_spearman"].min(),
                "median_parity_spearman": parity_frame["parity_spearman"].median(),
                "parity_pass": parity_pass,
                "repeat_rows": repeat_rows,
                "exact_repeat_rows": exact_repeat_rows,
                "repeat_spearman": repeat_spearman,
                "repeat_pass": repeat_pass,
                "checkpoint_sha256": next(iter(checkpoint_hashes))
                if len(checkpoint_hashes) == 1
                else "",
                "checkpoint_pass": checkpoint_pass,
                "container_sha256": expected_container,
                "container_pass": container_pass,
                "specification_pass": specification_pass,
                "input_hash_pass": input_hash_pass,
                "code_hash_pass": code_hash_pass,
                "prediction_hash_pass": prediction_hash_pass,
                "hardware_runtime_pass": hardware_pass,
                "failures_recorded": failures_recorded,
                "failures_a": len(first["manifest"]["failures"]),
                "failures_b": len(second["manifest"]["failures"]),
                "parity_failures": len(parity["manifest"]["failures"]),
                "prediction_sha256_a": first["prediction_sha256"],
                "prediction_sha256_b": second["prediction_sha256"],
                "prediction_sha256_parity": parity["prediction_sha256"],
                "qualification_status": "passed" if qualified else "failed",
            }
        )
    model_audit = pd.DataFrame(model_rows)
    parity_audit = pd.concat(parity_frames, ignore_index=True)
    qualified = model_audit.loc[
        model_audit["qualification_status"].eq("passed"), "model_id"
    ]
    qualified_families = int(
        model_audit.loc[
            model_audit["qualification_status"].eq("passed"), "family"
        ].nunique()
    )
    shared = set.intersection(*passing_targets) if passing_targets else set()
    gates = {
        "configurations": len(qualified) >= int(thresholds["minimum_configurations"]),
        "families": qualified_families >= int(thresholds["minimum_families"]),
        "shared_confirmation_targets": len(shared)
        >= int(thresholds["minimum_shared_confirmation_targets"]),
        "uniform_source_and_runner": len(source_tree_hashes) == 1
        and None not in source_tree_hashes
        and len(runner_hashes) == 1
        and None not in runner_hashes,
    }
    all_passed = all(gates.values())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_audit": output_dir / "model-qualification-audit.csv",
        "parity_audit": output_dir / "protein-gym-parity-audit.csv",
        "shared_targets": output_dir / "qualified-shared-targets.csv",
        "summary": output_dir / "qualification-summary.json",
    }
    write_table(model_audit, outputs["model_audit"])
    write_table(parity_audit, outputs["parity_audit"])
    write_table(pd.DataFrame({"target_id": sorted(shared)}), outputs["shared_targets"])
    summary = {
        "schema_version": 1,
        "protocol_id": str(qualification["protocol_id"]),
        "qualification_status": "passed" if all_passed else "failed",
        "qualified_configurations": len(qualified),
        "qualified_families": qualified_families,
        "shared_confirmation_targets": len(shared),
        "gates": gates,
        "thresholds": thresholds,
        "outcome_lock_state": outcome.get("state"),
        "confirmation_outcomes_accessed": False,
        "inputs": {
            "qualification_config_sha256": sha256_file(Path(qualification_config)),
            "model_config_sha256": sha256_file(Path(model_config)),
            "container_lock_sha256": sha256_file(Path(container_lock)),
            "confirmation_targets_sha256": sha256_file(Path(confirmation_targets)),
            "confirmation_variants_sha256": sha256_file(Path(confirmation_variants)),
            "outcome_lock_sha256": sha256_file(Path(outcome_lock)),
            "parity_targets_sha256": sha256_file(Path(parity_targets)),
            "parity_variants_sha256": sha256_file(Path(parity_variants)),
            "official_scores_sha256": sha256_file(Path(official_scores)),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                outputs["model_audit"],
                outputs["parity_audit"],
                outputs["shared_targets"],
            )
        },
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
