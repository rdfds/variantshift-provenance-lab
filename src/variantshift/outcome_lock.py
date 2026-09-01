"""One-way confirmation lock that keeps prediction code outcome blind."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .provenance import sha256_file

LOCK_STATES = ("targets_frozen", "predictions_frozen", "registered", "revealed")
OUTCOME_COLUMN_MARKERS = {
    "dms_score",
    "effect",
    "experimental_score",
    "fitness",
    "label",
    "outcome",
    "raw_score",
    "score",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def assert_target_only(frame: pd.DataFrame) -> None:
    blocked = sorted(OUTCOME_COLUMN_MARKERS.intersection(map(str.lower, frame.columns)))
    if blocked:
        raise ValueError(f"Target-only input contains prohibited outcome columns: {blocked}")


def create_outcome_lock(
    path: Path,
    *,
    protocol_id: str,
    target_artifacts: list[Path],
) -> Path:
    if not target_artifacts:
        raise ValueError("At least one target artifact is required")
    artifacts = {
        str(Path(artifact)): sha256_file(Path(artifact)) for artifact in target_artifacts
    }
    return _atomic_json(
        Path(path),
        {
            "schema_version": 1,
            "protocol_id": protocol_id,
            "state": "targets_frozen",
            "created_at": _now(),
            "target_artifacts": artifacts,
            "prediction_artifacts": {},
            "method_artifacts": {},
            "registration": None,
            "reveal": None,
        },
    )


def read_outcome_lock(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("state") not in LOCK_STATES:
        raise ValueError(f"Invalid confirmation lock state: {payload.get('state')}")
    return payload


def freeze_predictions(
    path: Path,
    *,
    prediction_artifacts: list[Path],
    method_artifacts: list[Path],
) -> Path:
    payload = read_outcome_lock(path)
    if payload["state"] != "targets_frozen":
        raise ValueError("Predictions can only be frozen from targets_frozen state")
    if not prediction_artifacts or not method_artifacts:
        raise ValueError("Prediction and method artifacts are both required before registration")
    payload["prediction_artifacts"] = {
        str(Path(item)): sha256_file(Path(item)) for item in prediction_artifacts
    }
    payload["method_artifacts"] = {
        str(Path(item)): sha256_file(Path(item)) for item in method_artifacts
    }
    payload["state"] = "predictions_frozen"
    payload["predictions_frozen_at"] = _now()
    return _atomic_json(Path(path), payload)


def register_confirmation(
    path: Path,
    *,
    registration_uri: str,
    registration_artifacts: list[Path] | None = None,
) -> Path:
    payload = read_outcome_lock(path)
    if payload["state"] != "predictions_frozen":
        raise ValueError("Confirmation can only be registered after predictions are frozen")
    parsed = urlparse(registration_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Registration URI must be a public HTTP(S) URL")
    artifacts = {
        str(Path(item)): sha256_file(Path(item)) for item in (registration_artifacts or [])
    }
    payload["registration"] = {
        "uri": registration_uri,
        "recorded_at": _now(),
        "artifacts": artifacts,
    }
    payload["state"] = "registered"
    return _atomic_json(Path(path), payload)


def assert_outcomes_accessible(path: Path) -> dict[str, object]:
    payload = read_outcome_lock(path)
    if payload["state"] not in {"registered", "revealed"}:
        raise PermissionError(
            "Confirmation outcomes remain locked until predictions, the method, and a public "
            "registration are frozen"
        )
    return payload


def assert_evaluation_artifacts_locked(
    path: Path,
    *,
    prediction_artifact: Path,
    method_artifact: Path,
    outcome_artifact: Path,
) -> dict[str, object]:
    """Require reveal and verify the exact artifacts against the immutable lock."""
    payload = read_outcome_lock(path)
    if payload["state"] != "revealed":
        raise PermissionError(
            "Confirmation evaluation requires a recorded one-time outcome reveal"
        )
    requested = {
        "prediction_artifacts": Path(prediction_artifact),
        "method_artifacts": Path(method_artifact),
        "outcome_artifacts": Path(outcome_artifact),
    }
    records = {
        "prediction_artifacts": payload.get("prediction_artifacts", {}),
        "method_artifacts": payload.get("method_artifacts", {}),
        "outcome_artifacts": dict(payload.get("reveal", {})).get("artifacts", {}),
    }
    for section, artifact in requested.items():
        if not artifact.is_file():
            raise ValueError(f"Evaluation artifact is unavailable: {artifact}")
        digest = sha256_file(artifact)
        locked_hashes = set(dict(records[section]).values())
        if digest not in locked_hashes:
            raise ValueError(f"{artifact} is not the frozen {section} artifact")
    for section, artifacts in records.items():
        for name, expected in dict(artifacts).items():
            artifact = Path(name)
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"Locked {section} artifact changed or disappeared: {artifact}")
    return payload


def record_outcome_reveal(path: Path, *, outcome_artifacts: list[Path]) -> Path:
    payload = assert_outcomes_accessible(path)
    if payload["state"] == "revealed":
        raise ValueError("Confirmation outcomes have already been revealed")
    if not outcome_artifacts:
        raise ValueError("At least one outcome artifact is required")
    payload["reveal"] = {
        "recorded_at": _now(),
        "artifacts": {
            str(Path(item)): sha256_file(Path(item)) for item in outcome_artifacts
        },
    }
    payload["state"] = "revealed"
    return _atomic_json(Path(path), payload)
