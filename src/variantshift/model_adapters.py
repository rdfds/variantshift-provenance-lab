"""Reproducible model adapters, preflight gates, and prediction caching."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .provenance import environment_versions, sha256_file
from .schemas import (
    PREDICTION_SCHEMA,
    VARIANT_SCHEMA,
    stable_frame_sha256,
    validate_targets,
    write_table,
)


@dataclass(frozen=True)
class ModelSpecification:
    model_id: str
    model_version: str
    family: str
    modalities: tuple[str, ...]
    adapter: str
    source_url: str
    license_name: str
    license_status: str
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    strategy: str | None = None
    command: tuple[str, ...] = ()
    parity_required: bool = False
    training_cutoff: str = "undocumented"
    exposure_status: str = "undocumented"
    container_image: str | None = None
    container_digest: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpecification:
        return cls(
            model_id=str(payload["model_id"]),
            model_version=str(payload["model_version"]),
            family=str(payload["family"]),
            modalities=tuple(map(str, payload["modalities"])),
            adapter=str(payload["adapter"]),
            source_url=str(payload["source_url"]),
            license_name=str(payload["license_name"]),
            license_status=str(payload["license_status"]),
            checkpoint=(str(payload["checkpoint"]) if payload.get("checkpoint") else None),
            checkpoint_sha256=(
                str(payload["checkpoint_sha256"])
                if payload.get("checkpoint_sha256")
                else None
            ),
            strategy=(str(payload["strategy"]) if payload.get("strategy") else None),
            command=tuple(map(str, payload.get("command", []))),
            parity_required=bool(payload.get("parity_required", False)),
            training_cutoff=str(payload.get("training_cutoff", "undocumented")),
            exposure_status=str(payload.get("exposure_status", "undocumented")),
            container_image=(
                str(payload["container_image"]) if payload.get("container_image") else None
            ),
            container_digest=(
                str(payload["container_digest"]) if payload.get("container_digest") else None
            ),
            notes=str(payload.get("notes", "")),
        )

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelAdapter(ABC):
    """Uniform inference interface; adapters never receive experimental outcomes."""

    def __init__(self, specification: ModelSpecification):
        self.specification = specification

    @abstractmethod
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        """Return ``variant_id`` and ``score`` for one target."""

    def provenance(self) -> dict[str, object]:
        checkpoint_hash = self.specification.checkpoint_sha256
        checkpoint = self.specification.checkpoint
        if checkpoint and Path(checkpoint).is_file():
            checkpoint_hash = sha256_file(Path(checkpoint))
        elif checkpoint:
            checkpoint_names = [item.strip() for item in checkpoint.split(";") if item.strip()]
            checkpoint_paths = [
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / f"{name}.pt"
                for name in checkpoint_names
            ]
            if checkpoint_paths and all(path.is_file() for path in checkpoint_paths):
                member_hashes = [sha256_file(path) for path in checkpoint_paths]
                checkpoint_hash = hashlib.sha256(
                    json.dumps(member_hashes, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
        container_hash = self.specification.container_digest
        if self.specification.container_image:
            container_path = Path(self.specification.container_image)
            if container_path.is_file():
                container_hash = sha256_file(container_path)
        return {
            "specification": asdict(self.specification),
            "specification_sha256": self.specification.digest(),
            "checkpoint_sha256": checkpoint_hash,
            "container_sha256": container_hash,
            "environment": environment_versions(),
        }


class FairESMAdapter(ModelAdapter):
    """Local fair-esm marginal scoring for ESM-1v and ESM-2 checkpoints."""

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        from .plm import esm2_position_log_probabilities, score_single_substitutions

        loader = self.specification.checkpoint
        strategy = self.specification.strategy or "masked-marginal"
        if not loader:
            raise ValueError("fair-esm adapters require a checkpoint loader name")
        positions = sorted(variants["position"].astype(int).unique())
        probabilities, token_index = esm2_position_log_probabilities(
            str(target["sequence"]),
            positions,
            model_name=loader,
            strategy=strategy,
        )
        scores = score_single_substitutions(
            probabilities,
            variants["variant_id"].astype(str),
            token_index,
        ).rename(columns={"mutation_codes": "variant_id", "prediction": "score"})
        return scores


class FairESMEnsembleAdapter(ModelAdapter):
    """Average marginal scores from an explicitly ordered fair-esm ensemble."""

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        checkpoints = [
            item.strip()
            for item in str(self.specification.checkpoint or "").split(";")
            if item.strip()
        ]
        if len(checkpoints) < 2:
            raise ValueError("fair-esm ensemble requires at least two checkpoint loaders")
        members = []
        for checkpoint in checkpoints:
            member_specification = ModelSpecification(
                **{
                    **asdict(self.specification),
                    "adapter": "fair_esm",
                    "checkpoint": checkpoint,
                }
            )
            scores = FairESMAdapter(member_specification).score_target(target, variants)
            members.append(scores.set_index("variant_id")["score"].rename(checkpoint))
        combined = pd.concat(members, axis=1, join="inner")
        return combined.mean(axis=1).rename("score").reset_index()


class CommandModelAdapter(ModelAdapter):
    """Run a pinned external scorer through a narrow CSV contract without a shell."""

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        if not self.specification.command:
            raise ValueError("Command adapter requires a non-empty command array")
        executable = self.specification.command[0]
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise RuntimeError(f"Model executable is unavailable: {executable}")
        with tempfile.TemporaryDirectory(prefix="variantshift-model-") as temporary:
            root = Path(temporary)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            input_frame = variants.copy()
            input_frame["sequence"] = str(target["sequence"])
            input_frame.to_csv(input_path, index=False)
            replacements = {
                "{input}": str(input_path),
                "{output}": str(output_path),
                "{target_id}": str(target["target_id"]),
            }
            command = [replacements.get(token, token) for token in self.specification.command]
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not output_path.is_file():
                raise RuntimeError("Model command did not create its declared output CSV")
            scores = pd.read_csv(output_path)
        missing = {"variant_id", "score"}.difference(scores.columns)
        if missing:
            raise ValueError(f"Model output is missing columns: {sorted(missing)}")
        return scores.loc[:, ["variant_id", "score"]]


class PrecomputedModelAdapter(ModelAdapter):
    def __init__(self, specification: ModelSpecification, predictions: Path):
        super().__init__(specification)
        self.predictions = pd.read_csv(predictions)

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        rows = self.predictions.loc[
            self.predictions["target_id"].astype(str).eq(str(target["target_id"]))
        ]
        if rows.empty:
            raise ValueError(f"No precomputed predictions for target {target['target_id']}")
        return rows.loc[:, ["variant_id", "score"]].copy()


def load_model_specifications(path: Path) -> list[ModelSpecification]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specifications = [ModelSpecification.from_dict(item) for item in payload["models"]]
    identifiers = [item.model_id for item in specifications]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Model identifiers must be unique")
    return specifications


def adapter_from_specification(
    specification: ModelSpecification,
    *,
    precomputed_path: Path | None = None,
) -> ModelAdapter:
    if specification.adapter == "fair_esm":
        return FairESMAdapter(specification)
    if specification.adapter == "fair_esm_ensemble":
        return FairESMEnsembleAdapter(specification)
    if specification.adapter == "command":
        return CommandModelAdapter(specification)
    if specification.adapter == "precomputed":
        if precomputed_path is None:
            raise ValueError("Precomputed adapter requires a prediction path")
        return PrecomputedModelAdapter(specification, precomputed_path)
    raise ValueError(f"Unsupported model adapter: {specification.adapter}")


def prediction_cache_key(
    specification: ModelSpecification,
    target: pd.Series,
    variants: pd.DataFrame,
) -> str:
    payload = {
        "model": specification.digest(),
        "target": str(target["sequence_sha256"]),
        "variants": stable_frame_sha256(variants.loc[:, VARIANT_SCHEMA.required]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_panel(
    adapter: ModelAdapter,
    targets: pd.DataFrame,
    variants: pd.DataFrame,
    *,
    protocol_id: str,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a target panel with content-addressed, resumable target jobs."""
    validate_targets(targets)
    VARIANT_SCHEMA.validate(variants)
    cache_dir = Path(cache_dir) / adapter.specification.model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for _, target in targets.sort_values("target_id").iterrows():
        target_variants = variants.loc[
            variants["target_id"].astype(str).eq(str(target["target_id"]))
        ].copy()
        key = prediction_cache_key(adapter.specification, target, target_variants)
        cached = cache_dir / f"{key}.csv"
        cache_hit = cached.is_file()
        started = time.perf_counter()
        try:
            if cache_hit:
                scores = pd.read_csv(cached)
            else:
                scores = adapter.score_target(target, target_variants)
                scores.to_csv(cached, index=False, lineterminator="\n")
            aligned = target_variants.loc[:, ["panel_id", "target_id", "variant_id"]].merge(
                scores.loc[:, ["variant_id", "score"]],
                on="variant_id",
                how="left",
                validate="one_to_one",
            )
            aligned["protocol_id"] = protocol_id
            aligned["model_id"] = adapter.specification.model_id
            aligned["model_version"] = adapter.specification.model_version
            aligned["status"] = np.where(aligned["score"].notna(), "ok", "missing")
            prediction_rows.append(aligned.loc[:, PREDICTION_SCHEMA.required])
            coverage = float(aligned["score"].notna().mean())
            status = "ok"
            error = ""
        except Exception as exception:  # noqa: BLE001 - isolate one external model failure
            coverage = 0.0
            status = "failed"
            error = f"{type(exception).__name__}: {exception}"
        elapsed_seconds = time.perf_counter() - started
        audit_rows.append(
            {
                "model_id": adapter.specification.model_id,
                "target_id": target["target_id"],
                "sequence_sha256": target["sequence_sha256"],
                "cache_key": key,
                "cache_hit": cache_hit,
                "coverage": coverage,
                "elapsed_seconds": elapsed_seconds,
                "variants_per_second": (
                    len(target_variants) / elapsed_seconds if elapsed_seconds > 0 else np.nan
                ),
                "status": status,
                "error": error,
            }
        )
    if not prediction_rows:
        raise RuntimeError(f"Model {adapter.specification.model_id} failed on every target")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    PREDICTION_SCHEMA.validate(predictions)
    return predictions, pd.DataFrame(audit_rows)


def _parity_correlation(predictions: pd.DataFrame, reference: pd.DataFrame) -> float:
    shared = predictions.merge(
        reference.loc[:, ["target_id", "variant_id", "score"]].rename(
            columns={"score": "reference_score"}
        ),
        on=["target_id", "variant_id"],
        how="inner",
    )
    if len(shared) < 2:
        return float("nan")
    return float(spearmanr(shared["score"], shared["reference_score"]).statistic)


def preflight_models(
    specifications: list[ModelSpecification],
    *,
    targets: pd.DataFrame | None = None,
    variants: pd.DataFrame | None = None,
    protocol_id: str = "preflight",
    cache_dir: Path = Path("artifacts/model-cache"),
    parity_dir: Path | None = None,
    execute: bool = False,
) -> pd.DataFrame:
    """Apply deterministic metadata, coverage, parity, and repeatability gates."""
    rows: list[dict[str, object]] = []
    passing_targets_by_model: dict[str, set[str]] = {}
    for specification in specifications:
        row: dict[str, object] = {
            "model_id": specification.model_id,
            "model_version": specification.model_version,
            "family": specification.family,
            "modalities": ";".join(specification.modalities),
            "adapter": specification.adapter,
            "source_url": specification.source_url,
            "license_name": specification.license_name,
            "license_status": specification.license_status,
            "training_cutoff": specification.training_cutoff,
            "exposure_status": specification.exposure_status,
            "container_image": specification.container_image,
            "container_digest": specification.container_digest,
            "checkpoint_sha256": specification.checkpoint_sha256,
            "provenance_complete": False,
            "metadata_complete": bool(
                specification.source_url
                and specification.license_name
                and specification.license_status
                in {"permitted", "restricted", "prohibited"}
            ),
            "execution_status": "not_run",
            "coverage": np.nan,
            "parity_spearman": np.nan,
            "repeat_spearman": np.nan,
            "elapsed_seconds": np.nan,
            "variants_per_second": np.nan,
            "target_count": 0 if targets is None else int(targets["target_id"].nunique()),
            "targets_at_95pct_coverage": 0,
            "primary_eligible": False,
            "exclusion_reason": "execution_not_requested",
        }
        if not execute:
            rows.append(row)
            continue
        if targets is None or variants is None:
            raise ValueError("Executable preflight requires target and variant tables")
        try:
            adapter = adapter_from_specification(specification)
            first, first_audit = score_panel(
                adapter,
                targets,
                variants,
                protocol_id=protocol_id,
                cache_dir=cache_dir / "first",
            )
            second, _ = score_panel(
                adapter,
                targets,
                variants,
                protocol_id=protocol_id,
                cache_dir=cache_dir / "second",
            )
            merged = first.merge(
                second,
                on=["target_id", "variant_id", "model_id"],
                suffixes=("_first", "_second"),
            )
            row["coverage"] = float(first["score"].notna().mean())
            passing_targets = set(
                first_audit.loc[
                    first_audit["status"].eq("ok")
                    & first_audit["coverage"].ge(0.95),
                    "target_id",
                ].astype(str)
            )
            passing_targets_by_model[specification.model_id] = passing_targets
            row["targets_at_95pct_coverage"] = len(passing_targets)
            row["elapsed_seconds"] = float(first_audit["elapsed_seconds"].sum())
            row["variants_per_second"] = float(
                len(first) / max(float(row["elapsed_seconds"]), 1e-12)
            )
            provenance = adapter.provenance()
            row["checkpoint_sha256"] = provenance["checkpoint_sha256"]
            row["container_digest"] = provenance["container_sha256"]
            checkpoint_complete = (
                not specification.checkpoint or bool(row["checkpoint_sha256"])
            )
            row["provenance_complete"] = bool(
                checkpoint_complete and row["container_digest"]
            )
            row["repeat_spearman"] = float(
                spearmanr(merged["score_first"], merged["score_second"]).statistic
            )
            if parity_dir is not None:
                parity_path = Path(parity_dir) / f"{specification.model_id}.csv"
                if parity_path.is_file():
                    row["parity_spearman"] = _parity_correlation(
                        first, pd.read_csv(parity_path)
                    )
            parity_passes = not specification.parity_required or (
                np.isfinite(row["parity_spearman"])
                and float(row["parity_spearman"]) >= 0.99
            )
            eligible = (
                specification.license_status == "permitted"
                and float(row["coverage"]) >= 0.95
                and float(row["repeat_spearman"]) >= 0.999
                and parity_passes
                and bool(row["provenance_complete"])
            )
            row["execution_status"] = "passed"
            row["primary_eligible"] = eligible
            row["exclusion_reason"] = "" if eligible else "one_or_more_preflight_gates_failed"
        except Exception as exception:  # noqa: BLE001 - preflight must audit, not abort
            row["execution_status"] = "failed"
            row["exclusion_reason"] = f"{type(exception).__name__}: {exception}"
        rows.append(row)
    audit = pd.DataFrame(rows)
    eligible_ids = audit.loc[audit["primary_eligible"].astype(bool), "model_id"].astype(str)
    eligible_sets = [passing_targets_by_model.get(model_id, set()) for model_id in eligible_ids]
    shared_targets = set.intersection(*eligible_sets) if eligible_sets else set()
    family_count = int(
        audit.loc[audit["primary_eligible"].astype(bool), "family"].nunique()
    )
    feasibility_passed = bool(
        len(eligible_ids) >= 8 and family_count >= 4 and len(shared_targets) >= 300
    )
    audit["primary_eligible_model_count"] = len(eligible_ids)
    audit["primary_family_count"] = family_count
    audit["primary_shared_target_count"] = len(shared_targets)
    audit["primary_shared_targets_sha256"] = (
        hashlib.sha256("\n".join(sorted(shared_targets)).encode("utf-8")).hexdigest()
        if shared_targets
        else ""
    )
    audit["feasibility_gate_passed"] = feasibility_passed
    return audit


def write_panel_predictions(
    model_config: Path,
    targets_path: Path,
    variants_path: Path,
    output_dir: Path,
    *,
    protocol_id: str,
    cache_dir: Path,
    model_ids: set[str] | None = None,
) -> dict[str, Path]:
    targets = pd.read_csv(targets_path)
    variants = pd.read_csv(variants_path)
    specifications = load_model_specifications(model_config)
    if model_ids is not None:
        specifications = [item for item in specifications if item.model_id in model_ids]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    provenance: dict[str, object] = {}
    for specification in specifications:
        adapter = adapter_from_specification(specification)
        model_predictions, audit = score_panel(
            adapter,
            targets,
            variants,
            protocol_id=protocol_id,
            cache_dir=cache_dir,
        )
        predictions.append(model_predictions)
        audits.append(audit)
        provenance[specification.model_id] = adapter.provenance()
    outputs = {
        "predictions": output_dir / "predictions.csv",
        "audit": output_dir / "prediction-audit.csv",
        "provenance": output_dir / "model-provenance.json",
    }
    write_table(pd.concat(predictions, ignore_index=True), outputs["predictions"])
    write_table(pd.concat(audits, ignore_index=True), outputs["audit"])
    outputs["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
