"""One-time retrieval of the two registered confirmation outcome panels."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from .confirmation_panels import (
    _DOMAINOME_MUTATION,
    _DOMAINOME_PREDICTOR_HEADER,
    DomainomePredictorSource,
)
from .outcome_lock import read_outcome_lock
from .pilot import VENUS_SOURCE_BASE, _canonicalize_venus
from .provenance import sha256_file
from .schemas import OUTCOME_SCHEMA

CONFIRMATION_PROTOCOL_ID = "variantshift-confirmation-freeze-v2"
CONFIRMATION_PANELS = ("human-domainome-v1", "venusmuthub-v1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "VariantShift/1.0 confirmation client"})
    with urlopen(request, timeout=180) as response:
        return response.read()


def _write_receipt(path: Path, *, url: str, payload_path: Path) -> dict[str, object]:
    receipt = {
        "source_locator": url,
        "accessed_at_utc": _now(),
        "raw_path": str(payload_path),
        "bytes": payload_path.stat().st_size,
        "sha256": sha256_file(payload_path),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _acquire_once(
    url: str,
    raw_path: Path,
    *,
    fetch_bytes: Callable[[str], bytes],
) -> tuple[bytes, dict[str, object]]:
    receipt_path = raw_path.with_suffix(raw_path.suffix + ".receipt.json")
    if raw_path.exists() != receipt_path.exists():
        raise RuntimeError(f"Incomplete source receipt pair: {raw_path}")
    if raw_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("source_locator") != url or receipt.get("sha256") != sha256_file(raw_path):
            raise RuntimeError(f"Cached source no longer matches its receipt: {raw_path}")
        return raw_path.read_bytes(), receipt
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_bytes(url)
    raw_path.write_bytes(payload)
    receipt = _write_receipt(receipt_path, url=url, payload_path=raw_path)
    return payload, receipt


def _verify_registered_inputs(lock: dict[str, object], artifacts: list[Path]) -> None:
    if lock.get("protocol_id") != CONFIRMATION_PROTOCOL_ID or lock.get("state") != "registered":
        raise PermissionError("Outcome retrieval requires the registered aggregate v2 lock")
    frozen = dict(lock.get("target_artifacts", {}))
    for artifact in artifacts:
        name = str(Path(artifact))
        expected = frozen.get(name)
        if expected is None:
            raise ValueError(f"Input is not recorded in the aggregate target freeze: {name}")
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Frozen confirmation input changed or disappeared: {name}")


def _domainome_outcomes(
    archive_payload: bytes,
    tasks: pd.DataFrame,
    targets_path: Path,
    variants_path: Path,
    source: DomainomePredictorSource,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hashlib.md5(archive_payload).hexdigest() != source.expected_md5:
        raise ValueError("Domainome outcome archive checksum differs from its registered pin")
    targets = pd.read_csv(targets_path).set_index("target_id")
    selected = set(tasks["target_id"].astype(str))
    target_sequences = targets.loc[sorted(selected), "sequence"].astype(str).to_dict()
    variants = pd.read_csv(variants_path, usecols=["target_id", "variant_id"])
    variants = variants.loc[variants["target_id"].astype(str).isin(selected)]
    expected = set(variants["target_id"].astype(str) + "\t" + variants["variant_id"].astype(str))
    values: dict[str, list[float]] = {}
    source_rows = 0
    with ZipFile(BytesIO(archive_payload)) as archive:
        if source.member not in archive.namelist():
            raise ValueError("Registered Domainome outcome member is absent")
        with archive.open(source.member) as handle:
            header = tuple(
                field.decode("ascii", errors="strict")
                for field in handle.readline().rstrip(b"\r\n").split(b"\t")
            )
            if header != _DOMAINOME_PREDICTOR_HEADER:
                raise ValueError("Domainome outcome columns differ from the registered schema")
            for line_number, line in enumerate(handle, start=2):
                fields = line.rstrip(b"\r\n").split(b"\t")
                if len(fields) != len(header):
                    raise ValueError(f"Domainome outcome row {line_number} has wrong field count")
                target_id = fields[0].decode("ascii", errors="strict")
                if target_id not in target_sequences:
                    continue
                source_rows += 1
                protein_id = fields[1].decode("ascii", errors="strict")
                mutation_text = fields[2].decode("ascii", errors="strict")
                mutant_sequence = fields[3].decode("ascii", errors="strict")
                if mutation_text == f"{protein_id}_NANANA":
                    continue
                match = _DOMAINOME_MUTATION.search(mutation_text)
                if match is None or not mutation_text.startswith(f"{protein_id}_"):
                    raise ValueError(f"Unrecognized Domainome mutation on row {line_number}")
                reference = match.group("reference")
                alternate = match.group("alternate")
                absolute_position = int(match.group("position"))
                domain_start = int(target_id.rsplit("_", 1)[1])
                local_position = absolute_position - domain_start + 1
                wild_type = target_sequences[target_id]
                if len(mutant_sequence) != len(wild_type) or not 1 <= local_position <= len(wild_type):
                    raise ValueError(f"Mutation coordinate does not map on row {line_number}")
                if (
                    wild_type[local_position - 1] != reference
                    or mutant_sequence[local_position - 1] != alternate
                ):
                    raise ValueError(f"Mutation/sequence mismatch on row {line_number}")
                if alternate == "*" or alternate == reference:
                    continue
                variant_id = f"{reference}{local_position}{alternate}"
                key = f"{target_id}\t{variant_id}"
                if key not in expected:
                    raise ValueError(f"Domainome outcome is outside the frozen universe: {key}")
                try:
                    effect = float(fields[6].decode("ascii", errors="strict"))
                except ValueError:
                    continue
                if math.isfinite(effect):
                    values.setdefault(key, []).append(effect)
    rows = []
    counts = {target_id: 0 for target_id in selected}
    for key, observed in sorted(values.items()):
        target_id, variant_id = key.split("\t")
        rows.append(
            {
                "protocol_id": CONFIRMATION_PROTOCOL_ID,
                "panel_id": "human-domainome-v1",
                "dataset_id": "human-domainome-v1",
                "assay_id": target_id,
                "target_id": target_id,
                "variant_id": variant_id,
                "effect": statistics.median(observed),
                "direction": 1,
            }
        )
        counts[target_id] += 1
    audit = pd.DataFrame(
        {
            "panel_id": "human-domainome-v1",
            "assay_id": sorted(selected),
            "target_id": sorted(selected),
            "status": ["parsed" if counts[item] else "no_finite_outcomes" for item in sorted(selected)],
            "retained_unique_variants": [counts[item] for item in sorted(selected)],
            "source_rows_total": source_rows,
            "effect_column": "scaled_fitness",
            "direction": 1,
        }
    )
    return pd.DataFrame(rows), audit


def retrieve_registered_confirmation_outcomes(
    outcome_lock_path: Path,
    task_registry_path: Path,
    domainome_targets_path: Path,
    domainome_variants_path: Path,
    venus_targets_path: Path,
    venus_assay_audit_path: Path,
    venus_protocol_path: Path,
    output_dir: Path,
    *,
    domainome_source: DomainomePredictorSource | None = None,
    fetch_bytes: Callable[[str], bytes] = _fetch_bytes,
) -> dict[str, Path]:
    """Retrieve exactly Domainome and untouched Venus outcomes under a registered lock."""
    outcome_lock_path = Path(outcome_lock_path)
    task_registry_path = Path(task_registry_path)
    domainome_targets_path = Path(domainome_targets_path)
    domainome_variants_path = Path(domainome_variants_path)
    venus_targets_path = Path(venus_targets_path)
    lock = read_outcome_lock(outcome_lock_path)
    _verify_registered_inputs(
        lock,
        [task_registry_path, domainome_targets_path, domainome_variants_path, venus_targets_path],
    )
    tasks = pd.read_csv(task_registry_path)
    tasks = tasks.loc[tasks["panel_id"].astype(str).isin(CONFIRMATION_PANELS)].copy()
    if not tasks["included"].astype(bool).all() or tasks["outcomes_accessed"].astype(bool).any():
        raise ValueError("Registered confirmation tasks are not an unopened included cohort")
    if set(tasks["panel_id"].astype(str)) != set(CONFIRMATION_PANELS):
        raise ValueError("Both registered confirmation panels are required")
    output_dir = Path(output_dir)
    outputs = {
        "outcomes": output_dir / "confirmation-outcomes.csv.gz",
        "audit": output_dir / "outcome-parsing-audit.csv",
        "ledger": output_dir / "outcome-access-ledger.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError("Confirmation outcomes have already been materialized")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = domainome_source or DomainomePredictorSource()
    domainome_raw = raw_dir / "human-domainome-v1" / "registered-outcomes.zip"
    domainome_payload, domainome_receipt = _acquire_once(
        source.url, domainome_raw, fetch_bytes=fetch_bytes
    )
    domainome_tasks = tasks.loc[tasks["panel_id"].eq("human-domainome-v1")]
    domainome, domainome_audit = _domainome_outcomes(
        domainome_payload,
        domainome_tasks,
        domainome_targets_path,
        domainome_variants_path,
        source,
    )
    domainome_receipt.update(
        {"panel_id": "human-domainome-v1", "parse_status": "parsed"}
    )

    protocol = json.loads(Path(venus_protocol_path).read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "variantshift-venusmuthub-confirmation-v1":
        raise ValueError("Unexpected VenusMutHub frozen protocol")
    source = protocol.get("source")
    if not isinstance(source, dict) or not source.get("revision"):
        raise ValueError("VenusMutHub target-freeze protocol lacks its source revision")
    revision = str(source["revision"])
    venus_targets = pd.read_csv(venus_targets_path).set_index("target_id")
    venus_sources = pd.read_csv(venus_assay_audit_path).set_index("dataset_id")
    venus_rows: list[dict[str, object]] = []
    venus_audits: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = [domainome_receipt]
    venus_tasks = tasks.loc[tasks["panel_id"].eq("venusmuthub-v1")].sort_values("assay_id")
    for task in venus_tasks.itertuples(index=False):
        assay_id, target_id = str(task.assay_id), str(task.target_id)
        source_row = venus_sources.loc[assay_id]
        source_path = str(source_row.source_path)
        url = f"{VENUS_SOURCE_BASE}/{revision}/{quote(source_path, safe='/')}?download=true"
        raw_path = raw_dir / "venusmuthub-v1" / f"{hashlib.sha256(assay_id.encode()).hexdigest()[:20]}.csv"
        payload, receipt = _acquire_once(url, raw_path, fetch_bytes=fetch_bytes)
        receipt.update({"panel_id": "venusmuthub-v1", "assay_id": assay_id})
        try:
            canonical, audit = _canonicalize_venus(
                pd.read_csv(BytesIO(payload), low_memory=False),
                assay_id=assay_id,
                sequence=str(venus_targets.loc[target_id, "sequence"]),
                direction=int(task.direction),
            )
            for row in canonical[["variant_id", "effect"]].itertuples(index=False):
                venus_rows.append(
                    {
                        "protocol_id": CONFIRMATION_PROTOCOL_ID,
                        "panel_id": "venusmuthub-v1",
                        "dataset_id": "venusmuthub-v1",
                        "assay_id": assay_id,
                        "target_id": target_id,
                        "variant_id": str(row.variant_id),
                        "effect": float(row.effect),
                        "direction": int(task.direction),
                    }
                )
            venus_audits.append(
                {"panel_id": "venusmuthub-v1", "assay_id": assay_id, "target_id": target_id,
                 "status": "parsed", **audit}
            )
            receipt["parse_status"] = "parsed"
        except (KeyError, TypeError, ValueError) as error:
            venus_audits.append(
                {"panel_id": "venusmuthub-v1", "assay_id": assay_id, "target_id": target_id,
                 "status": "excluded_by_registered_automatic_parser",
                 "error_type": type(error).__name__, "error": str(error)}
            )
            receipt["parse_status"] = "excluded"
        receipts.append(receipt)

    outcomes = pd.concat([domainome, pd.DataFrame(venus_rows)], ignore_index=True)
    OUTCOME_SCHEMA.validate(outcomes)
    outcomes = outcomes.sort_values(
        ["panel_id", "assay_id", "target_id", "variant_id"]
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(
        outputs["outcomes"], index=False, lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )
    audit = pd.concat([domainome_audit, pd.DataFrame(venus_audits)], ignore_index=True)
    audit.to_csv(outputs["audit"], index=False, lineterminator="\n")
    ledger = {
        "schema_version": 1,
        "protocol_id": CONFIRMATION_PROTOCOL_ID,
        "classification": "single registered confirmation outcome reveal",
        "registration": lock["registration"],
        "lock_sha256_before_reveal": sha256_file(outcome_lock_path),
        "retrieval_completed_at_utc": _now(),
        "panels_requested": list(CONFIRMATION_PANELS),
        "mavedb_outcomes_requested": False,
        "venus_source_revision": revision,
        "venus_source_protocol": str(venus_protocol_path),
        "venus_source_protocol_sha256": sha256_file(venus_protocol_path),
        "requests": receipts,
        "artifacts": {
            str(outputs["outcomes"]): sha256_file(outputs["outcomes"]),
            str(outputs["audit"]): sha256_file(outputs["audit"]),
        },
    }
    outputs["ledger"].write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
