"""Stable tabular contracts for preregistered VariantShift studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class FrameSchema:
    """A versioned, minimal contract for one public pipeline table."""

    name: str
    version: int
    required: tuple[str, ...]
    unique: tuple[str, ...] = ()

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.required).difference(frame.columns))
        if missing:
            raise ValueError(f"{self.name} is missing required columns: {missing}")
        if self.unique and frame.duplicated(list(self.unique)).any():
            duplicates = int(frame.duplicated(list(self.unique)).sum())
            raise ValueError(
                f"{self.name} contains {duplicates} duplicate rows for key {self.unique}"
            )
        return frame


TARGET_SCHEMA = FrameSchema(
    "targets-v1",
    1,
    (
        "panel_id",
        "target_id",
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
    ),
    ("panel_id", "target_id"),
)
VARIANT_SCHEMA = FrameSchema(
    "variants-v1",
    1,
    ("panel_id", "target_id", "variant_id", "position", "reference", "alternate"),
    ("panel_id", "target_id", "variant_id"),
)
PREDICTION_SCHEMA = FrameSchema(
    "predictions-v1",
    1,
    (
        "protocol_id",
        "panel_id",
        "target_id",
        "variant_id",
        "model_id",
        "model_version",
        "score",
        "status",
    ),
    ("protocol_id", "panel_id", "target_id", "variant_id", "model_id"),
)
OUTCOME_SCHEMA = FrameSchema(
    "outcomes-v1",
    1,
    (
        "protocol_id",
        "panel_id",
        "dataset_id",
        "assay_id",
        "target_id",
        "variant_id",
        "effect",
        "direction",
    ),
    ("protocol_id", "panel_id", "dataset_id", "assay_id", "target_id", "variant_id"),
)
TASK_METRIC_SCHEMA = FrameSchema(
    "task-metrics-v1",
    1,
    (
        "protocol_id",
        "panel_id",
        "dataset_id",
        "assay_id",
        "target_id",
        "protein_id",
        "family_id",
        "model_id",
        "selection_gain_sd",
    ),
    ("protocol_id", "panel_id", "dataset_id", "assay_id", "target_id", "model_id"),
)

SCHEMAS = {
    schema.name: schema
    for schema in (TARGET_SCHEMA, VARIANT_SCHEMA, PREDICTION_SCHEMA, OUTCOME_SCHEMA, TASK_METRIC_SCHEMA)
}


def sequence_sha256(sequence: str) -> str:
    normalized = sequence.removesuffix("*").strip().upper()
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def validate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical target sequences and their committed digests."""
    TARGET_SCHEMA.validate(frame)
    for row in frame.itertuples(index=False):
        sequence = str(row.sequence).removesuffix("*").upper()
        invalid = sorted(set(sequence).difference(AMINO_ACIDS))
        if not sequence or invalid:
            raise ValueError(f"Target {row.target_id} has invalid amino acids: {invalid}")
        if int(row.sequence_length) != len(sequence):
            raise ValueError(f"Target {row.target_id} has an incorrect sequence_length")
        if str(row.sequence_sha256) != sequence_sha256(sequence):
            raise ValueError(f"Target {row.target_id} has an incorrect sequence_sha256")
    return frame


def all_single_substitutions(targets: pd.DataFrame) -> pd.DataFrame:
    """Enumerate the complete 19L missense landscape without consulting outcomes."""
    validate_targets(targets)
    rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        sequence = str(target.sequence).removesuffix("*").upper()
        for position, reference in enumerate(sequence, start=1):
            for alternate in AMINO_ACIDS:
                if alternate == reference:
                    continue
                rows.append(
                    {
                        "panel_id": target.panel_id,
                        "target_id": target.target_id,
                        "variant_id": f"{reference}{position}{alternate}",
                        "position": position,
                        "reference": reference,
                        "alternate": alternate,
                    }
                )
    frame = pd.DataFrame(rows, columns=VARIANT_SCHEMA.required)
    return VARIANT_SCHEMA.validate(frame)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise RuntimeError("Parquet I/O requires pyarrow or fastparquet") from error
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        compression = "gzip" if path.name.endswith(".csv.gz") else None
        frame.to_csv(path, index=False, lineterminator="\n", compression=compression)
    elif path.suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True)
    elif path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
        except ImportError as error:
            raise RuntimeError("Parquet I/O requires pyarrow or fastparquet") from error
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    return path


def stable_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash a table independently of its incoming row and column order."""
    columns = sorted(map(str, frame.columns))
    normalized = frame.loc[:, columns].copy()
    if columns:
        normalized = normalized.sort_values(columns, kind="stable", na_position="last")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def schema_catalog() -> dict[str, object]:
    return {
        "schema_catalog_version": 1,
        "schemas": {
            name: {
                "version": schema.version,
                "required": list(schema.required),
                "unique": list(schema.unique),
            }
            for name, schema in sorted(SCHEMAS.items())
        },
    }


def write_schema_catalog(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_catalog(), indent=2, sort_keys=True) + "\n")
    return path
