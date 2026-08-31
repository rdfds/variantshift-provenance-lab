"""Outcome-blind panel construction from target-only public inputs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .outcome_lock import assert_target_only, create_outcome_lock
from .provenance import git_revision, sha256_file
from .schemas import (
    TARGET_SCHEMA,
    all_single_substitutions,
    sequence_sha256,
    validate_targets,
    write_schema_catalog,
    write_table,
)


@dataclass(frozen=True)
class PanelContext:
    panel_id: str
    source: str
    source_version: str
    exposure_policy: str = "undocumented"


class PanelAdapter(ABC):
    """Only target enumeration belongs in this interface; outcomes are intentionally absent."""

    def __init__(self, context: PanelContext):
        self.context = context

    @abstractmethod
    def targets(self) -> pd.DataFrame:
        """Return target sequences and metadata without experimental measurements."""


class TargetTablePanel(PanelAdapter):
    def __init__(self, context: PanelContext, path: Path):
        super().__init__(context)
        self.path = Path(path)

    def targets(self) -> pd.DataFrame:
        frame = pd.read_csv(self.path)
        assert_target_only(frame)
        required = {"target_id", "sequence"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Target table is missing columns: {missing}")
        result = frame.copy()
        result["panel_id"] = self.context.panel_id
        result["sequence"] = result["sequence"].astype(str).str.removesuffix("*").str.upper()
        if "protein_id" not in result:
            result["protein_id"] = result["target_id"]
        result["sequence_sha256"] = result["sequence"].map(sequence_sha256)
        result["sequence_length"] = result["sequence"].str.len()
        columns = list(TARGET_SCHEMA.required) + sorted(
            set(result.columns).difference(TARGET_SCHEMA.required)
        )
        return validate_targets(result.loc[:, columns])


class FastaPanel(PanelAdapter):
    def __init__(self, context: PanelContext, path: Path):
        super().__init__(context)
        self.path = Path(path)

    def targets(self) -> pd.DataFrame:
        records: list[tuple[str, str]] = []
        identifier: str | None = None
        chunks: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(chunks)))
                identifier = line[1:].split()[0]
                chunks = []
            elif line.strip():
                if identifier is None:
                    raise ValueError("FASTA sequence appears before its header")
                chunks.append(line.strip())
        if identifier is not None:
            records.append((identifier, "".join(chunks)))
        if not records:
            raise ValueError("FASTA input contains no targets")
        frame = pd.DataFrame(records, columns=["target_id", "sequence"])
        frame["panel_id"] = self.context.panel_id
        frame["protein_id"] = frame["target_id"]
        frame["sequence"] = frame["sequence"].str.removesuffix("*").str.upper()
        frame["sequence_sha256"] = frame["sequence"].map(sequence_sha256)
        frame["sequence_length"] = frame["sequence"].str.len()
        return validate_targets(frame.loc[:, TARGET_SCHEMA.required])


def panel_adapter_from_config(config: dict[str, object], *, config_dir: Path) -> PanelAdapter:
    context = PanelContext(
        panel_id=str(config["panel_id"]),
        source=str(config["source"]),
        source_version=str(config["source_version"]),
        exposure_policy=str(config.get("exposure_policy", "undocumented")),
    )
    input_path = Path(str(config["target_input"]))
    if not input_path.is_absolute():
        input_path = config_dir / input_path
    adapter = str(config.get("adapter", "target_table"))
    if adapter == "target_table":
        return TargetTablePanel(context, input_path)
    if adapter == "fasta":
        return FastaPanel(context, input_path)
    raise ValueError(f"Unsupported panel adapter: {adapter}")


def freeze_panel(config_path: Path, output_dir: Path) -> dict[str, Path]:
    """Freeze target sequences and the complete mutation universe without outcomes."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = panel_adapter_from_config(config, config_dir=config_path.parent)
    targets = adapter.targets()
    variants = all_single_substitutions(targets)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": output_dir / "targets.csv",
        "variants": output_dir / "variants.csv",
        "protocol": output_dir / "protocol.json",
        "schemas": output_dir / "schemas.json",
        "outcome_lock": output_dir / "outcome-lock.json",
    }
    write_table(targets, outputs["targets"])
    write_table(variants, outputs["variants"])
    write_schema_catalog(outputs["schemas"])
    protocol = {
        "schema_version": 1,
        "protocol_id": str(config["protocol_id"]),
        "panel_id": adapter.context.panel_id,
        "source": adapter.context.source,
        "source_version": adapter.context.source_version,
        "exposure_policy": adapter.context.exposure_policy,
        "outcome_status": "not_accessed",
        "target_count": len(targets),
        "variant_count": len(variants),
        "target_sha256": sha256_file(outputs["targets"]),
        "variant_sha256": sha256_file(outputs["variants"]),
        "source_git_commit": git_revision(Path.cwd()),
        "inclusion": config.get("inclusion", {}),
        "exclusion": config.get("exclusion", {}),
    }
    outputs["protocol"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    create_outcome_lock(
        outputs["outcome_lock"],
        protocol_id=str(config["protocol_id"]),
        target_artifacts=[
            outputs["targets"],
            outputs["variants"],
            outputs["protocol"],
            outputs["schemas"],
        ],
    )
    return outputs
