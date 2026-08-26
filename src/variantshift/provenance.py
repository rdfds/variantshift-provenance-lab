"""Run manifests and artifact-integrity verification."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repository_root: Path) -> str:
    """Return the exact source revision used for an analysis."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "scipy": version("scipy"),
        "variantshift": version("variantshift"),
    }


def build_run_manifest(
    *,
    repository_root: Path,
    dataset_path: Path,
    dataset_name: str,
    dataset_source: str,
    dataset_version: str,
    rows_after_filtering: int,
    filters: dict[str, Any],
    run: dict[str, Any],
    artifact_paths: list[Path],
    source_revision: str | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve()
    dataset_path = dataset_path.resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    for artifact_path in sorted(map(Path, artifact_paths), key=lambda path: path.as_posix()):
        resolved = artifact_path.resolve()
        relative_path = _relative(resolved, root)
        artifacts[relative_path] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    return {
        "schema_version": 1,
        "source": {
            "git_commit": source_revision or git_revision(root),
        },
        "dataset": {
            "name": dataset_name,
            "path": _relative(dataset_path, root),
            "sha256": sha256_file(dataset_path),
            "source": dataset_source,
            "version": dataset_version,
        },
        "filters": {
            **filters,
            "rows_after_filtering": rows_after_filtering,
        },
        "run": run,
        "environment": environment_versions(),
        "artifacts": artifacts,
    }


def build_collection_manifest(
    *,
    repository_root: Path,
    inputs: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    artifact_paths: list[Path],
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Build a manifest for a study that combines multiple public input files."""
    root = repository_root.resolve()
    input_records: dict[str, dict[str, Any]] = {}
    for name, details in sorted(inputs.items()):
        path = Path(details["path"]).resolve()
        input_records[name] = {
            "path": _relative(path, root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source": details["source"],
            "version": details["version"],
        }

    artifacts: dict[str, dict[str, Any]] = {}
    for artifact_path in sorted(map(Path, artifact_paths), key=lambda path: path.as_posix()):
        resolved = artifact_path.resolve()
        artifacts[_relative(resolved, root)] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return {
        "schema_version": 2,
        "source": {"git_commit": source_revision or git_revision(root)},
        "inputs": input_records,
        "protocol": protocol,
        "environment": environment_versions(),
        "artifacts": artifacts,
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def verify_manifest_artifacts(
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if repository_root is not None:
        root = repository_root.resolve()
    else:
        root = next(
            (
                parent
                for parent in (manifest_path.parent, *manifest_path.parents)
                if (parent / ".git").exists()
            ),
            manifest_path.parent.parent,
        ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Manifest does not contain any artifact records")

    verified: list[str] = []
    for relative_path, expected in sorted(artifacts.items()):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Artifact is missing: {relative_path}")
        actual_size = path.stat().st_size
        if actual_size != expected.get("bytes"):
            raise ValueError(
                f"Artifact size mismatch for {relative_path}: "
                f"expected {expected.get('bytes')}, found {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"Artifact hash mismatch for {relative_path}")
        verified.append(relative_path)

    dataset = manifest.get("dataset", {})
    dataset_path = root / str(dataset.get("path", ""))
    dataset_status = "not_present"
    if dataset and dataset_path.is_file():
        if sha256_file(dataset_path) != dataset.get("sha256"):
            raise ValueError(f"Dataset hash mismatch for {dataset.get('path')}")
        dataset_status = "verified"

    input_status: dict[str, str] = {}
    for name, record in sorted(manifest.get("inputs", {}).items()):
        input_path = root / str(record.get("path", ""))
        status = "not_present"
        if input_path.is_file():
            if input_path.stat().st_size != record.get("bytes"):
                raise ValueError(f"Input size mismatch for {record.get('path')}")
            if sha256_file(input_path) != record.get("sha256"):
                raise ValueError(f"Input hash mismatch for {record.get('path')}")
            status = "verified"
        input_status[name] = status

    return {
        "manifest": _relative(manifest_path, root),
        "source_commit": manifest.get("source", {}).get("git_commit"),
        "verified_artifacts": verified,
        "dataset_status": dataset_status,
        "input_status": input_status,
        "python": sys.version.split()[0],
    }
