"""Create the final immutable, outcome-blind confirmation freeze."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import joblib
import pandas as pd

from .outcome_lock import create_outcome_lock, freeze_predictions, read_outcome_lock
from .provenance import git_revision, sha256_file
from .schemas import stable_frame_sha256, write_table
from .transportability import predict_with_frozen_transport_model


def freeze_transport_decisions(
    bundle_path: Path,
    training_features_path: Path,
    confirmation_features_path: Path,
    runtime_lock_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run the fitted selector twice under its exact runtime and freeze identical output."""
    runtime = json.loads(runtime_lock_path.read_text(encoding="utf-8"))
    observed_versions = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in runtime["packages"]
        },
    }
    expected_versions = {"python": runtime["python"], **runtime["packages"]}
    if observed_versions != expected_versions:
        raise RuntimeError(
            f"Transport runtime differs from its lock: expected {expected_versions}, "
            f"found {observed_versions}"
        )
    bundle = joblib.load(bundle_path)
    training = pd.read_csv(training_features_path)
    if stable_frame_sha256(training) != str(bundle["training_frame_sha256"]):
        raise ValueError("Transport bundle and development feature table do not match")
    confirmation = pd.read_csv(confirmation_features_path)
    first = predict_with_frozen_transport_model(bundle, confirmation)
    second = predict_with_frozen_transport_model(bundle, confirmation)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    write_table(first, output_path)
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "runtime": observed_versions,
        "deterministic_repeat": True,
        "rows": len(first),
        "tasks": int(first["task_id"].nunique()),
        "models": int(first["model_id"].nunique()),
        "trusted_rows": int(first["trusted"].sum()),
        "inputs": {
            str(bundle_path): sha256_file(bundle_path),
            str(training_features_path): sha256_file(training_features_path),
            str(confirmation_features_path): sha256_file(confirmation_features_path),
            str(runtime_lock_path): sha256_file(runtime_lock_path),
        },
        "artifact": {str(output_path): sha256_file(output_path)},
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _code_snapshot(repository_root: Path, destination: Path) -> dict[str, str]:
    roots = ["src", "workflow", "configs", "containers"]
    allowed_suffixes = {".py", ".json", ".toml", ".yaml", ".yml", ".def"}
    paths = sorted(
        path
        for root in roots
        for path in (repository_root / root).rglob("*")
        if path.is_file() and path.suffix in allowed_suffixes
    )
    hashes = {
        path.relative_to(repository_root).as_posix(): sha256_file(path) for path in paths
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = path.relative_to(repository_root).as_posix()
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return hashes


def freeze_confirmation_bundle(
    repository_root: Path,
    final_lock_path: Path,
    target_artifacts: list[Path],
    selected_prediction_manifest_path: Path,
    prediction_artifacts: list[Path],
    method_artifacts: list[Path],
    output_dir: Path,
    *,
    protocol_id: str = "variantshift-confirmation-freeze-v1",
) -> dict[str, Path]:
    """Hash the complete code, predictions, and method bundle and advance one new lock."""
    if final_lock_path.exists():
        raise FileExistsError(f"Final confirmation lock already exists: {final_lock_path}")
    for path in target_artifacts + prediction_artifacts + method_artifacts:
        if not Path(path).is_file():
            raise ValueError(f"Final freeze artifact is missing: {path}")
    selected = json.loads(selected_prediction_manifest_path.read_text(encoding="utf-8"))
    if bool(selected.get("outcomes_accessed")):
        raise ValueError("Selected-prediction manifest accessed confirmation outcomes")
    selected_artifacts = {
        Path(name): digest for name, digest in selected["selected_artifacts"].items()
    }
    for path, expected in selected_artifacts.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Selected prediction artifact drifted: {path}")
    for target in target_artifacts:
        if target.name == "outcome-lock.json":
            source_lock = read_outcome_lock(target)
            if source_lock["state"] != "targets_frozen":
                raise ValueError(f"Source outcome lock is not targets_frozen: {target}")
            if source_lock.get("registration") is not None or source_lock.get("reveal") is not None:
                raise ValueError(f"Source outcome lock has already been opened: {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "code-snapshot.zip"
    code_hashes = _code_snapshot(repository_root, snapshot_path)
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_manifest_path = output_dir / "source-environment-manifest.json"
    source_manifest = {
        "schema_version": 1,
        "git_revision": git_revision(repository_root),
        "working_tree_dirty_at_freeze": bool(git_status),
        "working_tree_entry_count": len(git_status),
        "code_snapshot": {
            "path": str(snapshot_path),
            "sha256": sha256_file(snapshot_path),
            "files": code_hashes,
        },
        "freezer_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "outcomes_accessed": False,
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_files = [*selected_artifacts, *map(Path, prediction_artifacts)]
    method_files = [
        *map(Path, method_artifacts),
        selected_prediction_manifest_path,
        snapshot_path,
        source_manifest_path,
    ]
    bundle_manifest_path = output_dir / "final-freeze-manifest.json"
    bundle_manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "state": "ready_for_preregistration",
        "outcomes_accessed": False,
        "targets": {str(path): sha256_file(path) for path in target_artifacts},
        "predictions": {str(path): sha256_file(path) for path in prediction_files},
        "methods": {str(path): sha256_file(path) for path in method_files},
    }
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    method_files.append(bundle_manifest_path)
    create_outcome_lock(
        final_lock_path,
        protocol_id=protocol_id,
        target_artifacts=target_artifacts,
    )
    freeze_predictions(
        final_lock_path,
        prediction_artifacts=prediction_files,
        method_artifacts=method_files,
    )
    return {
        "outcome_lock": final_lock_path,
        "manifest": bundle_manifest_path,
        "code_snapshot": snapshot_path,
        "source_environment": source_manifest_path,
    }
