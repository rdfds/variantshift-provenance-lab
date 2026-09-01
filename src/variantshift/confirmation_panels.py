"""Target-only confirmation panel acquisition.

This module may call registry and metadata endpoints. It deliberately has no code
path to a MaveDB score endpoint, so cohort construction cannot reveal outcomes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import statistics
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import BinaryIO
from zipfile import ZipFile

import pandas as pd

from .external_validation import (
    MAVEDB_API_BASE,
    _canonical_json_bytes,
    _fetch_json,
    _sha256_bytes,
    _target_record,
    calibration_orientation,
    enumerate_published_score_sets,
    fetch_score_set_metadata,
)
from .outcome_lock import assert_target_only
from .provenance import sha256_file
from .schemas import TARGET_SCHEMA, validate_targets, write_table

_PROTEINGYM_MAVEDB = re.compile(r"urn_mavedb_(?P<accession>[0-9]+-[A-Za-z0-9-]+)_scores")
_DOMAINOME_HEADER_PREFIX = (b"dom_ID", b"PFAM_ID", b"wt_seq")
_CANONICAL_PROTEIN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")
_DOMAINOME_MUTATION = re.compile(
    r"(?P<reference>[ACDEFGHIKLMNPQRSTVWY])(?P<position>[0-9]+)(?P<alternate>[ACDEFGHIKLMNPQRSTVWY*])$"
)
_DOMAINOME_PREDICTOR_COLUMNS = (
    "ESM1v_domain",
    "RaSP",
    "ddmut",
    "FoldX",
    "popEVE",
    "EVE",
    "Tranception",
    "EVE_domain",
    "rsasa",
    "thermoMPNN",
    "AlphaMissense",
    "ESM1v_full-length",
)
_DOMAINOME_PREDICTOR_HEADER = (
    "domain_ID",
    "uniprot_ID",
    "uniprot_ID_mutation",
    "aa_seq",
    "fitness",
    "fitness_sigma",
    "scaled_fitness",
    "scaled_fitness_sigma",
    *_DOMAINOME_PREDICTOR_COLUMNS,
    "Organism",
    "Gene Names (primary)",
    "Gene Names (synonym)",
)


@dataclass(frozen=True)
class DomainomeTargetSource:
    """Pinned target source whose trailing columns are treated as sealed outcomes."""

    url: str = (
        "https://zenodo.org/api/records/14356805/files/"
        "Supplementary_Table_3_esm1v_residuals.txt/content"
    )
    zenodo_record: str = "14356805"
    doi: str = "10.5281/zenodo.14356805"
    expected_md5: str = "4401c96dd6f3083a50f13796eaa4a837"


@dataclass(frozen=True)
class DomainomePredictorSource:
    """Pinned mixed archive used only through an outcome-skipping column allowlist."""

    url: str = (
        "https://zenodo.org/api/records/14356805/files/"
        "Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip/content"
    )
    zenodo_record: str = "14356805"
    doi: str = "10.5281/zenodo.14356805"
    expected_md5: str = "2f3ff661e25aa5e6d92b664dc0709c17"
    member: str = "Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt"


@dataclass(frozen=True)
class MaveDBComplementCriteria:
    frozen_on_or_before: str = "2026-08-30"
    minimum_reported_variants: int = 100
    minimum_sequence_length: int = 20
    maximum_sequence_length: int = 2_500
    post_reveal_minimum_single_substitutions: int = 100
    post_reveal_minimum_assayed_positions: int = 10


def _extract_domainome_target_rows(stream: BinaryIO) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extract only the first three byte-delimited fields from the mixed source table.

    The complete response is hashed, but bytes after the third tab on every data row are never
    decoded or retained. This permits target acquisition without exposing the residual column.
    """
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    targets: dict[str, tuple[str, str]] = {}
    source_row_count = 0
    bytes_read = 0
    for line_number, line in enumerate(stream, start=1):
        sha256.update(line)
        md5.update(line)
        bytes_read += len(line)
        fields = line.rstrip(b"\r\n").split(b"\t", 3)
        if len(fields) != 4:
            raise ValueError(f"Domainome source row {line_number} has fewer than four fields")
        if line_number == 1:
            if tuple(fields[:3]) != _DOMAINOME_HEADER_PREFIX:
                raise ValueError("Domainome source target-column header does not match the pin")
            continue
        source_row_count += 1
        try:
            domain_id, pfam_id, sequence = (
                field.decode("ascii", errors="strict") for field in fields[:3]
            )
        except UnicodeDecodeError as exception:
            raise ValueError(
                f"Domainome target field is not ASCII on row {line_number}"
            ) from exception
        if not domain_id or not pfam_id or not _CANONICAL_PROTEIN.fullmatch(sequence):
            raise ValueError(f"Invalid Domainome target fields on row {line_number}")
        previous = targets.setdefault(domain_id, (pfam_id, sequence))
        if previous != (pfam_id, sequence):
            raise ValueError(f"Conflicting target metadata for Domainome domain {domain_id}")
    if source_row_count == 0:
        raise ValueError("Domainome source contains no data rows")
    rows = []
    for domain_id, (pfam_id, sequence) in sorted(targets.items()):
        rows.append(
            {
                "panel_id": "human-domainome-v1",
                "target_id": domain_id,
                "protein_id": domain_id.split("_", 1)[0],
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "sequence_length": len(sequence),
                "pfam_id": pfam_id,
            }
        )
    frame = validate_targets(pd.DataFrame(rows))
    receipt = {
        "source_bytes": bytes_read,
        "source_rows": source_row_count,
        "target_count": len(frame),
        "source_sha256": sha256.hexdigest(),
        "source_md5": md5.hexdigest(),
        "decoded_columns": [field.decode("ascii") for field in _DOMAINOME_HEADER_PREFIX],
        "discarded_field_policy": (
            "Bytes after the third tab of every data row were hashed but never decoded or stored."
        ),
        "outcomes_accessed": False,
    }
    return frame, receipt


def freeze_domainome_targets(
    output_dir: Path,
    *,
    source: DomainomeTargetSource | None = None,
    stream: BinaryIO | None = None,
) -> dict[str, Path]:
    """Freeze Human Domainome target sequences without decoding the mixed table's outcomes."""
    source = source or DomainomeTargetSource()
    supplied_stream = stream is not None
    response = stream or urllib.request.urlopen(source.url, timeout=120)
    try:
        targets, receipt = _extract_domainome_target_rows(response)
    finally:
        if not supplied_stream:
            response.close()
    if not supplied_stream and receipt["source_md5"] != source.expected_md5:
        raise ValueError(
            "Domainome source checksum differs from the pinned Zenodo artifact; refusing freeze"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": output_dir / "targets.csv",
        "receipt": output_dir / "target-extraction-receipt.json",
    }
    write_table(targets, outputs["targets"])
    receipt.update(
        {
            "schema_version": 1,
            "source_url": source.url,
            "zenodo_record": source.zenodo_record,
            "source_doi": source.doi,
            "expected_source_md5": source.expected_md5,
            "target_sha256": sha256_file(outputs["targets"]),
        }
    )
    outputs["receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_domainome_official_predictions(
    archive_path: Path,
    targets_path: Path,
    variants_path: Path,
    output_dir: Path,
    *,
    source: DomainomePredictorSource | None = None,
) -> dict[str, Path]:
    """Extract published label-free predictions without decoding adjacent outcome columns."""
    source = source or DomainomePredictorSource()
    archive_path = Path(archive_path)
    if _md5_file(archive_path) != source.expected_md5:
        raise ValueError("Domainome predictor archive checksum differs from the Zenodo pin")
    targets = pd.read_csv(targets_path)
    validate_targets(targets)
    assert_target_only(targets)
    target_by_id = targets.set_index("target_id")["sequence"].astype(str).to_dict()
    expected_frame = pd.read_csv(variants_path, usecols=["target_id", "variant_id"])
    expected = set(
        expected_frame["target_id"].astype(str) + "\t" + expected_frame["variant_id"].astype(str)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": output_dir / "official-predictions.csv.gz",
        "receipt": output_dir / "official-prediction-extraction-receipt.json",
    }
    partial = outputs["predictions"].with_suffix(".csv.gz.partial")
    seen: set[str] = set()
    coverage = {column: 0 for column in _DOMAINOME_PREDICTOR_COLUMNS}
    selected_source_rows = 0
    duplicate_source_rows = 0
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        if source.member not in members:
            raise ValueError("Pinned Domainome predictor member is absent from the archive")
        with archive.open(source.member) as handle:
            header_fields = handle.readline().rstrip(b"\r\n").split(b"\t")
            try:
                header = tuple(field.decode("ascii", errors="strict") for field in header_fields)
            except UnicodeDecodeError as exception:
                raise ValueError("Domainome predictor header is not ASCII") from exception
            if header != _DOMAINOME_PREDICTOR_HEADER:
                raise ValueError("Domainome predictor columns differ from the pinned schema")
            with (
                partial.open("wb") as raw_output,
                gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed,
                io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_output,
            ):
                writer = csv.writer(text_output, lineterminator="\n")
                writer.writerow(("target_id", "variant_id", *_DOMAINOME_PREDICTOR_COLUMNS))
                current_key: str | None = None
                current_target_id = ""
                current_variant_id = ""
                current_scores: list[list[str]] = []

                def flush_prediction_group() -> None:
                    nonlocal current_key, duplicate_source_rows
                    if current_key is None:
                        return
                    if current_key in seen:
                        raise ValueError(
                            f"Non-contiguous duplicate Domainome prediction: {current_key}"
                        )
                    aggregated = []
                    for column_index, column in enumerate(_DOMAINOME_PREDICTOR_COLUMNS):
                        values = [
                            row[column_index]
                            for row in current_scores
                            if row[column_index].strip().lower() not in {"", "na", "nan"}
                        ]
                        if values:
                            value = (
                                values[0]
                                if len(values) == 1
                                else format(
                                    statistics.median(float(item) for item in values), ".17g"
                                )
                            )
                            coverage[column] += 1
                        else:
                            value = ""
                        aggregated.append(value)
                    writer.writerow((current_target_id, current_variant_id, *aggregated))
                    duplicate_source_rows += len(current_scores) - 1
                    seen.add(current_key)

                for line_number, line in enumerate(handle, start=2):
                    fields = line.rstrip(b"\r\n").split(b"\t")
                    if len(fields) != len(header):
                        raise ValueError(
                            f"Domainome predictor row {line_number} has wrong field count"
                        )
                    try:
                        target_id = fields[0].decode("ascii", errors="strict")
                    except UnicodeDecodeError as exception:
                        raise ValueError(
                            f"Domainome target ID is not ASCII on row {line_number}"
                        ) from exception
                    if target_id not in target_by_id:
                        continue
                    selected_source_rows += 1
                    try:
                        protein_id = fields[1].decode("ascii", errors="strict")
                        mutation_text = fields[2].decode("ascii", errors="strict")
                        mutant_sequence = fields[3].decode("ascii", errors="strict")
                    except UnicodeDecodeError as exception:
                        raise ValueError(
                            f"Allowlisted Domainome field is not ASCII on row {line_number}"
                        ) from exception
                    wild_type = target_by_id[target_id]
                    if mutation_text == f"{protein_id}_NANANA":
                        if mutant_sequence != wild_type:
                            raise ValueError(
                                f"Wild-type row differs from {target_id} on row {line_number}"
                            )
                        continue
                    match = _DOMAINOME_MUTATION.search(mutation_text)
                    if match is None or not mutation_text.startswith(f"{protein_id}_"):
                        raise ValueError(f"Unrecognized Domainome mutation on row {line_number}")
                    reference = match.group("reference")
                    alternate = match.group("alternate")
                    absolute_position = int(match.group("position"))
                    try:
                        domain_start = int(target_id.rsplit("_", 1)[1])
                    except (IndexError, ValueError) as exception:
                        raise ValueError(
                            f"Domainome target lacks a start coordinate: {target_id}"
                        ) from exception
                    local_position = absolute_position - domain_start + 1
                    if len(mutant_sequence) != len(wild_type) or not (
                        1 <= local_position <= len(wild_type)
                    ):
                        raise ValueError(
                            f"Mutation coordinate does not map to {target_id} on row {line_number}"
                        )
                    if (
                        wild_type[local_position - 1] != reference
                        or mutant_sequence[local_position - 1] != alternate
                    ):
                        raise ValueError(
                            f"Mutation/sequence mismatch for {target_id} on row {line_number}"
                        )
                    if alternate == "*" or alternate == reference:
                        continue
                    try:
                        scores = [
                            fields[index].decode("ascii", errors="strict")
                            for index in range(8, 8 + len(_DOMAINOME_PREDICTOR_COLUMNS))
                        ]
                    except UnicodeDecodeError as exception:
                        raise ValueError(
                            f"Predictor field is not ASCII on row {line_number}"
                        ) from exception
                    variant_id = f"{reference}{local_position}{alternate}"
                    key = f"{target_id}\t{variant_id}"
                    if key not in expected:
                        raise ValueError(
                            f"Published prediction is outside the frozen universe: {key}"
                        )
                    if current_key is not None and key != current_key:
                        flush_prediction_group()
                        current_scores = []
                    if key in seen:
                        raise ValueError(f"Non-contiguous duplicate Domainome prediction: {key}")
                    current_key = key
                    current_target_id = target_id
                    current_variant_id = variant_id
                    current_scores.append(scores)
                flush_prediction_group()
    if seen != expected:
        missing = len(expected.difference(seen))
        extra = len(seen.difference(expected))
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Official Domainome predictions do not match the frozen universe: "
            f"{missing} missing, {extra} extra"
        )
    partial.replace(outputs["predictions"])
    receipt = {
        "schema_version": 1,
        "source_url": source.url,
        "source_doi": source.doi,
        "zenodo_record": source.zenodo_record,
        "source_member": source.member,
        "source_bytes": archive_path.stat().st_size,
        "source_md5": _md5_file(archive_path),
        "source_sha256": sha256_file(archive_path),
        "decoded_target_columns": list(_DOMAINOME_PREDICTOR_HEADER[:4]),
        "decoded_prediction_columns": list(_DOMAINOME_PREDICTOR_COLUMNS),
        "skipped_outcome_columns": list(_DOMAINOME_PREDICTOR_HEADER[4:8]),
        "skipped_outcome_policy": (
            "Fields 5-8 were separated as bytes but never decoded, parsed, logged, or stored."
        ),
        "outcomes_accessed": False,
        "selected_source_rows": selected_source_rows,
        "duplicate_source_rows_collapsed": duplicate_source_rows,
        "duplicate_prediction_policy": "per-model median before outcome access",
        "prediction_rows": len(seen),
        "target_count": len(target_by_id),
        "nonmissing_prediction_count": coverage,
        "prediction_coverage": {column: count / len(seen) for column, count in coverage.items()},
        "target_sha256": sha256_file(targets_path),
        "variant_sha256": sha256_file(variants_path),
        "prediction_sha256": sha256_file(outputs["predictions"]),
    }
    outputs["receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def _proteingym_mavedb_urns(reference: pd.DataFrame) -> set[str]:
    urns = set()
    for filename in reference["raw_DMS_filename"].dropna().astype(str):
        match = _PROTEINGYM_MAVEDB.search(filename)
        if match:
            urns.add(f"urn:mavedb:{match.group('accession')}")
    return urns


def _publication_dois(metadata: dict[str, object]) -> set[str]:
    dois = set()
    experiment = metadata.get("experiment") or {}
    sources = [metadata, experiment] if isinstance(experiment, dict) else [metadata]
    for source in sources:
        for key in ("primaryPublicationIdentifiers", "secondaryPublicationIdentifiers"):
            for publication in source.get(key) or []:
                if isinstance(publication, dict) and publication.get("doi"):
                    dois.add(str(publication["doi"]).strip().lower())
    return dois


def _prior_selected_urns(protocol_path: Path) -> set[str]:
    payload = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    return set(map(str, payload.get("panel", {}).get("selected_urns", [])))


def freeze_mavedb_complement_targets(
    reference_path: Path,
    prior_protocol_path: Path,
    output_dir: Path,
    *,
    criteria: MaveDBComplementCriteria | None = None,
    registry: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    """Freeze all direction-resolved MaveDB complement targets without scores."""
    criteria = criteria or MaveDBComplementCriteria()
    cutoff = date.fromisoformat(criteria.frozen_on_or_before)
    reference = pd.read_csv(reference_path)
    prior = _prior_selected_urns(prior_protocol_path)
    proteingym_urns = _proteingym_mavedb_urns(reference)
    proteingym_dois = set(reference["jo"].dropna().astype(str).str.strip().str.lower())
    supplied_registry = registry is not None
    registry = registry if supplied_registry else enumerate_published_score_sets()

    candidate_rows: list[dict[str, object]] = []
    detail_urns: list[str] = []
    for row in registry:
        urn = str(row["urn"])
        published_text = str(row.get("publishedDate") or "")
        try:
            published = date.fromisoformat(published_text)
        except ValueError:
            continue
        target = _target_record(row)
        if target is None:
            continue
        reported = int(row.get("numVariants") or 0)
        if published > cutoff or reported < criteria.minimum_reported_variants:
            continue
        if not criteria.minimum_sequence_length <= int(target["sequence_length"]):
            continue
        if int(target["sequence_length"]) > criteria.maximum_sequence_length:
            continue
        preexcluded = []
        if urn in prior:
            preexcluded.append("existing_mavedb_development_v1")
        if urn in proteingym_urns:
            preexcluded.append("proteingym_score_set")
        candidate_rows.append(
            {
                "urn": urn,
                "title": str(row.get("title") or ""),
                "published_date": published.isoformat(),
                "reported_variants": reported,
                **target,
                "registry_metadata_sha256": _sha256_bytes(_canonical_json_bytes(row)),
                "exclusion_reasons": ";".join(preexcluded),
            }
        )
        if not preexcluded:
            detail_urns.append(urn)
    if not candidate_rows:
        raise ValueError("No protein MaveDB score sets met target-only candidate rules")

    details = fetch_score_set_metadata(detail_urns) if detail_urns else []
    detail_by_urn = {str(detail["urn"]): detail for detail in details}
    audit = pd.DataFrame(candidate_rows)
    selected_details: list[dict[str, object]] = []
    for index, row in audit.iterrows():
        urn = str(row["urn"])
        if row["exclusion_reasons"]:
            continue
        detail = detail_by_urn[urn]
        reasons = []
        detail_target = _target_record(detail)
        if detail_target is None or detail_target["sequence_sha256"] != row["sequence_sha256"]:
            reasons.append("search_detail_target_mismatch")
        if detail.get("metaAnalyzesScoreSetUrns"):
            reasons.append("meta_analysis_score_set")
        if _publication_dois(detail).intersection(proteingym_dois):
            reasons.append("proteingym_publication_overlap")
        orientation = calibration_orientation(detail)
        if orientation not in {-1, 1}:
            reasons.append("ambiguous_metadata_direction")
        audit.at[index, "metadata_orientation"] = orientation
        audit.at[index, "publication_dois"] = ";".join(sorted(_publication_dois(detail)))
        audit.at[index, "detail_metadata_sha256"] = _sha256_bytes(_canonical_json_bytes(detail))
        audit.at[index, "exclusion_reasons"] = ";".join(reasons)
        if not reasons:
            selected_details.append(detail)
    audit["selected"] = audit["exclusion_reasons"].fillna("").eq("")
    selected = audit.loc[audit["selected"]].copy()
    if selected.empty:
        raise ValueError("No MaveDB complement score set has metadata-resolved direction")

    target_rows = []
    for sequence_digest, group in selected.groupby("sequence_sha256", sort=True):
        representative = group.iloc[0]
        target_rows.append(
            {
                "panel_id": "mavedb-complement-v1",
                "target_id": f"mavedb-{sequence_digest[:16]}",
                "protein_id": str(representative["target_name"]),
                "sequence": str(representative["target_sequence"]),
                "sequence_sha256": str(sequence_digest),
                "sequence_length": int(representative["sequence_length"]),
                "source_score_set_urns": ";".join(sorted(group["urn"].astype(str))),
            }
        )
    targets = validate_targets(pd.DataFrame(target_rows))
    target_id_map = targets.set_index("sequence_sha256")["target_id"]
    audit["target_id"] = audit["sequence_sha256"].map(target_id_map)
    audit = audit.drop(columns="target_sequence")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": output_dir / "targets.csv",
        "audit": output_dir / "score-set-audit.csv",
        "metadata": output_dir / "detailed-metadata-snapshot.json",
        "protocol": output_dir / "target-freeze-protocol.json",
    }
    write_table(
        targets.loc[:, [*TARGET_SCHEMA.required, "source_score_set_urns"]], outputs["targets"]
    )
    write_table(audit.sort_values("urn"), outputs["audit"])
    outputs["metadata"].write_text(
        json.dumps(selected_details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_digest = _sha256_bytes(_canonical_json_bytes(registry))
    protocol = {
        "schema_version": 1,
        "protocol_id": "variantshift-mavedb-complement-confirmation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcomes_accessed": False,
        "outcome_endpoint_requests": 0,
        "source": {
            "api": MAVEDB_API_BASE,
            "api_version": "test-fixture" if supplied_registry else _fetch_json("api/version"),
            "registry_records": len(registry),
            "registry_sha256": registry_digest,
        },
        "criteria": asdict(criteria),
        "preexisting_exclusions": {
            "mavedb_development_v1_urns": len(prior),
            "proteingym_mavedb_urns": len(proteingym_urns),
            "proteingym_publication_dois": len(proteingym_dois),
        },
        "panel": {
            "target_only_candidates": len(audit),
            "selected_score_sets": int(audit["selected"].sum()),
            "selected_target_sequences": len(targets),
            "selected_urns": sorted(selected["urn"].astype(str).tolist()),
        },
        "post_reveal_rules": {
            "minimum_single_amino_acid_substitutions": criteria.post_reveal_minimum_single_substitutions,
            "minimum_assayed_positions": criteria.post_reveal_minimum_assayed_positions,
            "duplicate_protein_consequence": "median",
            "direction": "frozen metadata_orientation only; no outcome-based reversal",
        },
    }
    outputs["protocol"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol["artifact_sha256"] = {
        key: sha256_file(path) for key, path in outputs.items() if key != "protocol"
    }
    outputs["protocol"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
