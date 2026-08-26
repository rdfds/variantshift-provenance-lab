import json
from pathlib import Path

import pytest

from variantshift.provenance import (
    build_collection_manifest,
    build_run_manifest,
    sha256_file,
    verify_manifest_artifacts,
    write_manifest,
)


def test_collection_manifest_hashes_multiple_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.csv"
    artifact = tmp_path / "results.csv"
    first.write_bytes(b"archive")
    second.write_text("index")
    artifact.write_text("result")
    manifest = build_collection_manifest(
        repository_root=tmp_path,
        inputs={
            "assays": {
                "path": first,
                "source": "https://example.org/assays.zip",
                "version": "1.0",
            },
            "index": {
                "path": second,
                "source": "https://example.org/index.csv",
                "version": "1.0",
            },
        },
        protocol={"seeds": [42, 43]},
        artifact_paths=[artifact],
        source_revision="abc123",
    )
    path = write_manifest(manifest, tmp_path / "manifest.json")
    verification = verify_manifest_artifacts(path, repository_root=tmp_path)
    assert manifest["schema_version"] == 2
    assert verification["input_status"] == {"assays": "verified", "index": "verified"}


def test_nested_manifest_discovers_repository_root(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    artifact = tmp_path / "docs" / "figure.svg"
    artifact.parent.mkdir()
    artifact.write_text("<svg/>")
    manifest = build_collection_manifest(
        repository_root=tmp_path,
        inputs={},
        protocol={},
        artifact_paths=[artifact],
        source_revision="abc123",
    )
    manifest_path = write_manifest(
        manifest, tmp_path / "results" / "study" / "run-manifest.json"
    )
    verification = verify_manifest_artifacts(manifest_path)
    assert verification["verified_artifacts"] == ["docs/figure.svg"]


def test_manifest_hashes_dataset_and_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("variant,value\nA1C,1\n")
    artifact = tmp_path / "results" / "benchmark.csv"
    artifact.parent.mkdir()
    artifact.write_text("metric,value\nspearman,0.5\n")

    manifest = build_run_manifest(
        repository_root=tmp_path,
        dataset_path=dataset,
        dataset_name=dataset.name,
        dataset_source="https://example.org/dataset",
        dataset_version="1.0",
        rows_after_filtering=1,
        filters={"min_total_counts": 1000},
        run={"seed": 42},
        artifact_paths=[artifact],
        source_revision="abc123",
    )
    manifest_path = write_manifest(manifest, tmp_path / "results" / "run-manifest.json")

    assert manifest["dataset"]["sha256"] == sha256_file(dataset)
    verification = verify_manifest_artifacts(manifest_path, repository_root=tmp_path)
    assert verification["dataset_status"] == "verified"
    assert verification["verified_artifacts"] == ["results/benchmark.csv"]


def test_verification_detects_modified_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("original")
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {"git_commit": "abc123"},
                "dataset": {"path": "missing.csv", "sha256": "unused"},
                "artifacts": {
                    "artifact.csv": {
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                },
            }
        )
    )
    artifact.write_text("modified")

    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest_artifacts(manifest_path, repository_root=tmp_path)
