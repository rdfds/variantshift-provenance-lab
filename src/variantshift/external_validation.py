"""Label-blind MaveDB cohort freezing and one-shot external validation.

The module deliberately separates metadata access from outcome access.  ``freeze_external_panel``
never calls a score endpoint; the resulting protocol can therefore be committed before labels are
downloaded.  Score ingestion is a separate command with an explicit timestamped access ledger.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .metrics import top_selection_metrics
from .proteingym import read_reference_index
from .provenance import sha256_file

MAVEDB_API_BASE = "https://api.mavedb.org/api/v1"
PROTEINGYM_V13_RELEASE = date(2025, 4, 28)
DEFAULT_FREEZE_CUTOFF = date(2026, 8, 29)

AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
_PROTEIN_SUBSTITUTION = re.compile(
    r"^p\.(?P<reference>[A-Z][a-z]{2})(?P<position>[1-9][0-9]*)"
    r"(?P<alternate>[A-Z][a-z]{2}|Ter|=)$"
)


@dataclass(frozen=True)
class ExternalSelectionCriteria:
    """Outcome-independent rules for the temporally external MaveDB panel."""

    published_after: str = PROTEINGYM_V13_RELEASE.isoformat()
    frozen_on_or_before: str = DEFAULT_FREEZE_CUTOFF.isoformat()
    minimum_reported_variants: int = 500
    minimum_sequence_length: int = 40
    maximum_sequence_length: int = 2_000
    sequence_identity_threshold: float = 0.30
    bidirectional_coverage_threshold: float = 0.80
    search_identity_floor: float = 0.15
    search_coverage_floor: float = 0.50

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalOutcomeCriteria:
    """Rules applied once, without revision, after score-table access."""

    minimum_single_missense_variants: int = 500
    minimum_mutated_positions: int = 20
    minimum_unique_scores: int = 10
    minimum_orientation_controls: int = 10
    duplicate_protein_consequence_rule: str = "median"
    primary_model: str = "esm2_t6_8M_UR50D_masked_marginal"
    secondary_model: str = "esm2_t6_8M_UR50D_wild_type_marginal"
    primary_metric: str = "protein-balanced mean within-assay Spearman"
    primary_success_rule: str = "nested-bootstrap 95% lower bound > 0"
    bootstrap_repeats: int = 10_000
    bootstrap_seed: int = 2_026_0829

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fetch_json(
    path: str,
    *,
    payload: dict[str, object] | None = None,
    attempts: int = 4,
) -> object:
    data = _canonical_json_bytes(payload) if payload is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "VariantShift/0.5 external-validation client",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{MAVEDB_API_BASE}/{path.lstrip('/')}", data=data, headers=headers)
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, http.client.IncompleteRead, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable MaveDB retry state")


def enumerate_published_score_sets(
    *,
    page_size: int = 100,
    workers: int = 6,
    snapshot_attempts: int = 3,
) -> list[dict[str, object]]:
    """Return a count-consistent public metadata snapshot without requesting scores."""
    if not 1 <= page_size <= 100:
        raise ValueError("MaveDB page size must lie in [1, 100]")
    if workers < 1 or snapshot_attempts < 1:
        raise ValueError("Registry workers and snapshot attempts must be positive")

    def fetch_page(offset: int) -> dict[str, object]:
        payload = _fetch_json(
            "score-sets/search",
            payload={"published": True, "limit": page_size, "offset": offset},
        )
        if not isinstance(payload, dict):
            raise TypeError("MaveDB search returned an unexpected payload")
        if not isinstance(payload.get("scoreSets"), list):
            raise TypeError("MaveDB search payload is missing scoreSets")
        return payload

    for _ in range(snapshot_attempts):
        first = fetch_page(0)
        expected = int(first["numScoreSets"])
        offsets = list(range(0, expected, page_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pages = list(executor.map(fetch_page, offsets))
        if any(int(page["numScoreSets"]) != expected for page in pages):
            continue
        rows = [row for page in pages for row in page["scoreSets"]]
        urns = [str(row["urn"]) for row in rows]
        if len(rows) == expected and len(urns) == len(set(urns)):
            return sorted(rows, key=lambda row: str(row["urn"]))
    raise RuntimeError("MaveDB registry changed during every snapshot attempt")


def _target_record(row: dict[str, object]) -> dict[str, object] | None:
    targets = row.get("targetGenes")
    if not isinstance(targets, list) or len(targets) != 1:
        return None
    target = targets[0]
    if not isinstance(target, dict):
        return None
    sequence_record = target.get("targetSequence")
    if not isinstance(sequence_record, dict) or sequence_record.get("sequenceType") != "protein":
        return None
    original_sequence = str(sequence_record.get("sequence") or "").upper()
    sequence = original_sequence.removesuffix("*")
    if not sequence or set(sequence).difference(set("ACDEFGHIKLMNPQRSTVWY")):
        return None
    return {
        "target_name": str(target.get("name") or row["urn"]),
        "target_sequence": sequence,
        "sequence_length": len(sequence),
        "sequence_sha256": _sha256_bytes(sequence.encode("ascii")),
        "terminal_stop_removed": original_sequence.endswith("*"),
    }


def select_metadata_candidates(
    rows: list[dict[str, object]],
    *,
    criteria: ExternalSelectionCriteria,
) -> pd.DataFrame:
    """Apply only publication, target, sequence, and reported-size rules."""
    release = date.fromisoformat(criteria.published_after)
    cutoff = date.fromisoformat(criteria.frozen_on_or_before)
    selected: list[dict[str, object]] = []
    for row in rows:
        published_text = str(row.get("publishedDate") or "")
        try:
            published = date.fromisoformat(published_text)
        except ValueError:
            continue
        target = _target_record(row)
        if target is None:
            continue
        reported = int(row.get("numVariants") or 0)
        if not release < published <= cutoff:
            continue
        if reported < criteria.minimum_reported_variants:
            continue
        if not criteria.minimum_sequence_length <= int(target["sequence_length"]):
            continue
        if int(target["sequence_length"]) > criteria.maximum_sequence_length:
            continue
        compact = {
            "urn": str(row["urn"]),
            "title": str(row.get("title") or ""),
            "published_date": published.isoformat(),
            "modification_date": str(row.get("modificationDate") or ""),
            "reported_variants": reported,
            "license": str((row.get("license") or {}).get("shortName") or ""),
            **target,
            "metadata_sha256": _sha256_bytes(_canonical_json_bytes(row)),
        }
        selected.append(compact)
    frame = pd.DataFrame(selected)
    if frame.empty:
        raise ValueError("No MaveDB metadata records met the external-panel rules")
    if frame["urn"].duplicated().any():
        raise RuntimeError("Selected MaveDB score-set URNs are not unique")
    return frame.sort_values(["published_date", "urn"]).reset_index(drop=True)


def _write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with Path(path).open("w", encoding="ascii") as handle:
        handle.writelines(f">{identifier}\n{sequence}\n" for identifier, sequence in records)


def audit_proteingym_family_overlap(
    candidates: pd.DataFrame,
    reference_path: Path,
    *,
    criteria: ExternalSelectionCriteria,
    binary: str = "mmseqs",
    threads: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Search candidate targets against every ProteinGym v1.3 target sequence."""
    executable = shutil.which(binary)
    if executable is None:
        raise RuntimeError("MMseqs2 is required to freeze the external panel")
    version = subprocess.run(
        [executable, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    unique_candidates = candidates.drop_duplicates("sequence_sha256").copy()
    unique_candidates["query_id"] = [f"external{index:03d}" for index in range(len(unique_candidates))]
    query_lookup = unique_candidates.set_index("query_id")
    reference = read_reference_index(reference_path).drop_duplicates("target_seq").copy()
    reference["target_id"] = [f"proteingym{index:03d}" for index in range(len(reference))]
    target_lookup = reference.set_index("target_id")
    with tempfile.TemporaryDirectory(prefix="variantshift-external-mmseqs-") as temporary:
        root = Path(temporary)
        query_fasta = root / "external.fasta"
        target_fasta = root / "proteingym.fasta"
        output = root / "alignments.tsv"
        _write_fasta(
            list(zip(unique_candidates["query_id"], unique_candidates["target_sequence"], strict=True)),
            query_fasta,
        )
        _write_fasta(list(zip(reference["target_id"], reference["target_seq"], strict=True)), target_fasta)
        command = [
            executable,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(output),
            str(root / "work"),
            "--exhaustive-search",
            "1",
            "--min-seq-id",
            str(criteria.search_identity_floor),
            "-c",
            str(criteria.search_coverage_floor),
            "--cov-mode",
            "0",
            "--max-seqs",
            "10000",
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--threads",
            str(threads),
            "-v",
            "1",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        columns = [
            "query_id",
            "target_id",
            "sequence_identity",
            "alignment_length",
            "query_coverage",
            "target_coverage",
            "e_value",
            "bit_score",
        ]
        if output.stat().st_size:
            alignments = pd.read_csv(output, sep="\t", names=columns)
        else:
            alignments = pd.DataFrame(columns=columns)
    if not alignments.empty:
        alignments = alignments.merge(
            query_lookup[["sequence_sha256", "target_name"]],
            left_on="query_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        ).merge(
            target_lookup[["DMS_id", "UniProt_ID"]],
            left_on="target_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
    alignments["qualifies_family_overlap"] = (
        pd.to_numeric(alignments.get("sequence_identity"), errors="coerce").ge(
            criteria.sequence_identity_threshold
        )
        & pd.to_numeric(alignments.get("query_coverage"), errors="coerce").ge(
            criteria.bidirectional_coverage_threshold
        )
        & pd.to_numeric(alignments.get("target_coverage"), errors="coerce").ge(
            criteria.bidirectional_coverage_threshold
        )
    )
    overlapping = set(
        alignments.loc[alignments["qualifies_family_overlap"], "sequence_sha256"].astype(str)
    )
    audited = candidates.copy()
    audited["proteingym_family_overlap"] = audited["sequence_sha256"].isin(overlapping)
    audited["selected_for_external_validation"] = ~audited["proteingym_family_overlap"]
    return audited, alignments, version


def fetch_score_set_metadata(
    urns: list[str],
    *,
    workers: int = 6,
) -> list[dict[str, object]]:
    """Freeze detailed public metadata, including calibrations, without score tables."""
    if workers < 1:
        raise ValueError("Metadata worker count must be positive")

    def fetch(urn: str) -> dict[str, object]:
        payload = _fetch_json(f"score-sets/{urn}")
        if not isinstance(payload, dict) or str(payload.get("urn")) != urn:
            raise RuntimeError(f"MaveDB returned inconsistent metadata for {urn}")
        return payload

    ordered = sorted(set(map(str, urns)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        details = list(executor.map(fetch, ordered))
    return details


def freeze_external_panel(
    reference_path: Path,
    output_dir: Path,
    *,
    criteria: ExternalSelectionCriteria | None = None,
    outcome_criteria: ExternalOutcomeCriteria | None = None,
    binary: str = "mmseqs",
    threads: int = 8,
) -> dict[str, Path]:
    """Freeze a metadata-only cohort and protocol without touching outcome endpoints."""
    criteria = criteria or ExternalSelectionCriteria()
    outcome_criteria = outcome_criteria or ExternalOutcomeCriteria()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = enumerate_published_score_sets()
    candidates = select_metadata_candidates(registry, criteria=criteria)
    audited, alignments, mmseqs_version = audit_proteingym_family_overlap(
        candidates,
        reference_path,
        criteria=criteria,
        binary=binary,
        threads=threads,
    )
    details = fetch_score_set_metadata(audited["urn"].astype(str).tolist())
    detail_by_urn = {str(detail["urn"]): detail for detail in details}
    audited["detail_metadata_sha256"] = audited["urn"].map(
        {urn: _sha256_bytes(_canonical_json_bytes(detail)) for urn, detail in detail_by_urn.items()}
    )
    audited["metadata_orientation"] = audited["urn"].map(
        {urn: calibration_orientation(detail) for urn, detail in detail_by_urn.items()}
    )
    for row in audited.itertuples(index=False):
        detail_target = _target_record(detail_by_urn[str(row.urn)])
        if detail_target is None or detail_target["sequence_sha256"] != row.sequence_sha256:
            raise RuntimeError(f"MaveDB search/detail target mismatch for {row.urn}")
    selected = audited.loc[audited["selected_for_external_validation"]]
    if selected["sequence_sha256"].nunique() < 10:
        raise RuntimeError("The frozen external panel contains fewer than ten independent targets")
    api_version = _fetch_json("api/version")
    registry_digest = _sha256_bytes(_canonical_json_bytes(registry))
    protocol = {
        "protocol": "VariantShift locked-box MaveDB external validation",
        "version": "1.0.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcomes_accessed": False,
        "outcome_access_timestamp_utc": None,
        "source": {
            "api": MAVEDB_API_BASE,
            "api_version": api_version,
            "published_registry_records": len(registry),
            "registry_canonical_sha256": registry_digest,
            "proteingym_release": "v1.3",
            "proteingym_release_date": PROTEINGYM_V13_RELEASE.isoformat(),
        },
        "selection": criteria.to_dict(),
        "outcome_rules": outcome_criteria.to_dict(),
        "orientation_rule": {
            "first": (
                "use MaveDB normal-versus-abnormal calibration ranges when their ordering "
                "unambiguously defines whether larger scores are more functional"
            ),
            "fallback": (
                "if at least 10 synonymous and 10 termination controls exist, orient toward "
                "the synonymous-control median"
            ),
            "otherwise": "exclude from directed primary analysis and retain in an unsigned audit",
        },
        "models": {
            "primary": {
                "identifier": "esm2_t6_8M_UR50D",
                "scoring": "masked marginal log-odds",
                "long_sequence_rule": (
                    "1022-residue windows with 256-residue overlap; assign each position to the "
                    "containing window that maximizes distance from a boundary"
                ),
            },
            "secondary": {
                "identifier": "esm2_t6_8M_UR50D",
                "scoring": "wild-type marginal log-odds using the identical window assignment",
            },
        },
        "estimands": {
            "primary": outcome_criteria.primary_metric,
            "within_protein": "average assay estimates before averaging proteins",
            "uncertainty": (
                "nested bootstrap: resample proteins, assays within proteins, and mutated "
                "positions within assays"
            ),
            "secondary": [
                "within-assay Spearman",
                "predicted top-decile recall",
                "top-decile experimental selection gain in assay SD units",
                "masked-minus-wild-type marginal paired differences",
                "cross-condition stability for repeated protein assays",
            ],
        },
        "panel": {
            "metadata_candidates": len(audited),
            "excluded_proteingym_family_assays": int(audited["proteingym_family_overlap"].sum()),
            "selected_assays": len(selected),
            "selected_target_names": int(selected["target_name"].nunique()),
            "selected_target_sequences": int(selected["sequence_sha256"].nunique()),
            "metadata_oriented_selected_assays": int(selected["metadata_orientation"].notna().sum()),
            "selected_urns": selected["urn"].astype(str).tolist(),
        },
        "audit_commit_requirement": (
            "Commit and push this protocol with outcomes_accessed=false before any selected "
            "score endpoint is requested. Publish every later inclusion and exclusion."
        ),
        "mmseqs_version": mmseqs_version,
    }
    outputs = {
        "protocol": output_dir / "protocol.json",
        "registry": output_dir / "metadata-registry.csv",
        "family_alignments": output_dir / "proteingym-family-alignments.csv",
        "metadata_details": output_dir / "detailed-metadata-snapshot.json",
    }
    outputs["protocol"].write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audited.drop(columns="target_sequence").to_csv(outputs["registry"], index=False)
    alignments.to_csv(outputs["family_alignments"], index=False)
    outputs["metadata_details"].write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_mave_hgvs_protein_substitution(code: str) -> tuple[str, int, str]:
    """Parse one unambiguous MAVE-HGVS protein substitution."""
    match = _PROTEIN_SUBSTITUTION.fullmatch(str(code))
    if match is None:
        raise ValueError(f"Not a single MAVE-HGVS protein substitution: {code}")
    reference = AA3_TO_AA1.get(match.group("reference"))
    alternate_text = match.group("alternate")
    if reference is None:
        raise ValueError(f"Unsupported reference amino acid in {code}")
    if alternate_text == "Ter":
        alternate = "*"
    elif alternate_text == "=":
        alternate = reference
    else:
        alternate = AA3_TO_AA1.get(alternate_text)
        if alternate is None:
            raise ValueError(f"Unsupported alternate amino acid in {code}")
    return reference, int(match.group("position")), alternate


def calibration_orientation(metadata: dict[str, object]) -> int | None:
    """Return +1 when larger raw scores mean more function, or -1 when smaller do."""
    for calibration in metadata.get("scoreCalibrations") or []:
        classifications = calibration.get("functionalClassifications") or []
        normal = [item for item in classifications if item.get("functionalClassification") == "normal"]
        abnormal = [
            item for item in classifications if item.get("functionalClassification") == "abnormal"
        ]
        for normal_item in normal:
            for abnormal_item in abnormal:
                normal_range = normal_item.get("range") or [None, None]
                abnormal_range = abnormal_item.get("range") or [None, None]
                if (
                    normal_range[0] is not None
                    and abnormal_range[1] is not None
                    and float(normal_range[0]) >= float(abnormal_range[1])
                ):
                    return 1
                if (
                    normal_range[1] is not None
                    and abnormal_range[0] is not None
                    and float(normal_range[1]) <= float(abnormal_range[0])
                ):
                    return -1
    return None


def canonicalize_mavedb_scores(
    frame: pd.DataFrame,
    *,
    sequence: str,
    orientation: int | None,
    minimum_controls: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reduce a MaveDB table to unique finite single-missense protein consequences."""
    required = {"hgvs_pro", "score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"MaveDB score table is missing columns: {', '.join(missing)}")
    sequence = sequence.upper()
    parsed_rows: list[dict[str, object]] = []
    invalid = 0
    reference_mismatches = 0
    for code, raw_score in frame[["hgvs_pro", "score"]].itertuples(index=False, name=None):
        score = pd.to_numeric(pd.Series([raw_score]), errors="coerce").iat[0]
        if not np.isfinite(score):
            continue
        try:
            reference, position, alternate = parse_mave_hgvs_protein_substitution(str(code))
        except ValueError:
            invalid += 1
            continue
        if position > len(sequence) or sequence[position - 1] != reference:
            reference_mismatches += 1
            continue
        effect = "synonymous" if alternate == reference else "termination" if alternate == "*" else "missense"
        parsed_rows.append(
            {
                "hgvs_pro": str(code),
                "mutation_codes": f"{reference}{position}{alternate}",
                "position": position,
                "effect": effect,
                "raw_score": float(score),
            }
        )
    parsed = pd.DataFrame(parsed_rows)
    if parsed.empty:
        raise ValueError("MaveDB score table has no valid finite protein substitutions")
    if orientation is None:
        synonymous = parsed.loc[parsed["effect"].eq("synonymous"), "raw_score"]
        termination = parsed.loc[parsed["effect"].eq("termination"), "raw_score"]
        if len(synonymous) >= minimum_controls and len(termination) >= minimum_controls:
            delta = float(synonymous.median() - termination.median())
            orientation = 1 if delta > 0 else -1 if delta < 0 else None
    missense = parsed.loc[parsed["effect"].eq("missense")].copy()
    duplicate_rows = int(missense["mutation_codes"].duplicated(keep=False).sum())
    missense = (
        missense.groupby(["mutation_codes", "position"], as_index=False)
        .agg(raw_score=("raw_score", "median"), protein_consequence_rows=("raw_score", "size"))
        .sort_values(["position", "mutation_codes"])
        .reset_index(drop=True)
    )
    missense["DMS_score"] = missense["raw_score"] * orientation if orientation else np.nan
    audit = {
        "input_rows": len(frame),
        "parsed_rows": len(parsed),
        "invalid_or_non_single_protein_codes": invalid,
        "reference_mismatches": reference_mismatches,
        "single_missense_variants": len(missense),
        "mutated_positions": int(missense["position"].nunique()),
        "unique_scores": int(missense["raw_score"].nunique()),
        "duplicate_protein_consequence_rows": duplicate_rows,
        "synonymous_controls": int(parsed["effect"].eq("synonymous").sum()),
        "termination_controls": int(parsed["effect"].eq("termination").sum()),
        "orientation": orientation,
        "directed_analysis_eligible": orientation in (-1, 1),
    }
    return missense, audit


def evaluate_external_predictions(frame: pd.DataFrame) -> dict[str, float]:
    """Evaluate aligned external predictions using the protocol's declared metrics."""
    required = {"DMS_score", "prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"External predictions are missing columns: {', '.join(missing)}")
    observed = frame["DMS_score"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("External evaluation requires finite aligned values")
    if len(frame) < 2 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        spearman = 0.0
    else:
        spearman = float(pd.Series(observed).corr(pd.Series(predicted), method="spearman"))
    return {"spearman": spearman, **top_selection_metrics(observed, predicted, fraction=0.10)}


def download_selected_score_tables(protocol_path: Path, output_dir: Path) -> dict[str, Path]:
    """Access every frozen score table exactly once and write a timestamped ledger."""
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("outcomes_accessed"):
        raise ValueError("Protocol already records outcome access")
    urns = list(protocol["panel"]["selected_urns"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    ledger = []
    accessed_at = datetime.now(timezone.utc).isoformat()
    for urn in urns:
        request = Request(
            f"{MAVEDB_API_BASE}/score-sets/{urn}/scores",
            headers={"Accept": "text/csv", "User-Agent": "VariantShift/0.5 locked-box client"},
        )
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        # Parse before accepting the download so an HTML error cannot enter the frozen cache.
        frame = pd.read_csv(BytesIO(payload))
        if not {"hgvs_pro", "score"}.issubset(frame.columns):
            raise RuntimeError(f"Score endpoint for {urn} lacks hgvs_pro and score columns")
        safe_name = urn.replace(":", "_") + ".csv"
        path = output_dir / safe_name
        path.write_bytes(payload)
        outputs[urn] = path
        ledger.append(
            {
                "urn": urn,
                "path": str(path),
                "bytes": len(payload),
                "rows": len(frame),
                "sha256": _sha256_bytes(payload),
                "accessed_at_utc": accessed_at,
            }
        )
    ledger_path = output_dir / "outcome-access-ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    outputs["ledger"] = ledger_path
    return outputs


def _protein_identifier(metadata: dict[str, object]) -> str:
    target = metadata["targetGenes"][0]
    mapped_gene = target.get("mappedHgncName")
    if mapped_gene:
        return str(mapped_gene)
    target_name = str(target.get("name") or "").strip()
    if target_name:
        # Domain constructs such as "TSC2 RapGAP" must share the TSC2 protein unit with the
        # corresponding full-length target.  This rule uses frozen target metadata only.
        return target_name.split()[0]
    mapped_uniprot = target.get("uniprotIdFromMappedMetadata")
    if mapped_uniprot:
        return str(mapped_uniprot)
    for external in target.get("externalIdentifiers") or []:
        identifier = external.get("identifier") or {}
        if identifier.get("dbName") == "UniProt" and identifier.get("identifier"):
            return str(identifier["identifier"])
    return str(metadata["urn"])


def build_external_outcome_cohort(
    protocol_dir: Path,
    score_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen outcome rules and retain every assay-level audit decision."""
    protocol_dir = Path(protocol_dir)
    score_dir = Path(score_dir)
    protocol = json.loads((protocol_dir / "protocol.json").read_text())
    outcome_rules = protocol["outcome_rules"]
    registry = pd.read_csv(protocol_dir / "metadata-registry.csv")
    selected = registry.loc[registry["selected_for_external_validation"].astype(bool)].copy()
    details = {
        str(metadata["urn"]): metadata
        for metadata in json.loads((protocol_dir / "detailed-metadata-snapshot.json").read_text())
    }
    cohort_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        urn = str(row.urn)
        metadata = details[urn]
        target = _target_record(metadata)
        if target is None:
            raise RuntimeError(f"Frozen MaveDB target is no longer parseable: {urn}")
        score_path = score_dir / f"{urn.replace(':', '_')}.csv"
        if not score_path.is_file():
            raise FileNotFoundError(f"Frozen score table is missing: {score_path}")
        frozen_orientation = None
        if pd.notna(row.metadata_orientation):
            frozen_orientation = int(row.metadata_orientation)
        canonical, audit = canonicalize_mavedb_scores(
            pd.read_csv(score_path, low_memory=False),
            sequence=str(target["target_sequence"]),
            orientation=frozen_orientation,
            minimum_controls=int(outcome_rules["minimum_orientation_controls"]),
        )
        exclusion_reasons = []
        if audit["single_missense_variants"] < int(
            outcome_rules["minimum_single_missense_variants"]
        ):
            exclusion_reasons.append("too_few_single_missense_variants")
        if audit["mutated_positions"] < int(outcome_rules["minimum_mutated_positions"]):
            exclusion_reasons.append("too_few_mutated_positions")
        if audit["unique_scores"] < int(outcome_rules["minimum_unique_scores"]):
            exclusion_reasons.append("too_few_unique_scores")
        if not audit["directed_analysis_eligible"]:
            exclusion_reasons.append("score_direction_not_predeclared_or_control_identifiable")
        eligible = not exclusion_reasons
        audit_rows.append(
            {
                "urn": urn,
                "target_name": str(row.target_name),
                "protein_id": _protein_identifier(metadata),
                "sequence_sha256": str(row.sequence_sha256),
                "sequence_length": int(row.sequence_length),
                "score_sha256": sha256_file(score_path),
                **audit,
                "eligible": eligible,
                "exclusion_reasons": ";".join(exclusion_reasons),
            }
        )
        if eligible:
            canonical["urn"] = urn
            canonical["target_name"] = str(row.target_name)
            canonical["protein_id"] = _protein_identifier(metadata)
            canonical["sequence_sha256"] = str(row.sequence_sha256)
            cohort_frames.append(canonical)
    audit_frame = pd.DataFrame(audit_rows).sort_values("urn").reset_index(drop=True)
    if not cohort_frames:
        raise RuntimeError("No frozen MaveDB assays passed the predeclared outcome rules")
    cohort = pd.concat(cohort_frames, ignore_index=True).sort_values(
        ["urn", "position", "mutation_codes"]
    )
    if cohort[["urn", "mutation_codes"]].duplicated().any():
        raise RuntimeError("Canonical external cohort contains duplicate assay variants")
    if cohort["DMS_score"].isna().any():
        raise RuntimeError("Directed external cohort contains unoriented scores")
    return cohort.reset_index(drop=True), audit_frame


def score_external_cohort(
    cohort: pd.DataFrame,
    protocol_dir: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every unique eligible sequence once under both frozen ESM-2 strategies."""
    import esm
    import torch

    from .plm import esm2_position_log_probabilities, score_single_substitutions

    protocol_dir = Path(protocol_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t6_8M_UR50D.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resolved ESM-2 checkpoint is missing: {checkpoint}")
    details = json.loads((protocol_dir / "detailed-metadata-snapshot.json").read_text())
    sequence_by_digest = {}
    for metadata in details:
        target = _target_record(metadata)
        if target is not None:
            sequence_by_digest[str(target["sequence_sha256"])] = str(target["target_sequence"])
    predictions = []
    audit_rows = []
    for digest, sequence_frame in cohort.groupby("sequence_sha256", sort=True):
        sequence = sequence_by_digest[str(digest)]
        variants = sequence_frame["mutation_codes"].drop_duplicates().astype(str).tolist()
        positions = sorted(sequence_frame["position"].astype(int).unique())
        strategy_scores = {}
        for strategy in ("masked-marginal", "wild-type-marginal"):
            probabilities, token_index = esm2_position_log_probabilities(
                sequence,
                positions,
                model_name="esm2_t6_8M_UR50D",
                strategy=strategy,
                device=device,
                batch_size=batch_size,
                window_size=1022,
                overlap=256,
            )
            scored = score_single_substitutions(probabilities, variants, token_index)
            strategy_scores[strategy] = scored.rename(
                columns={"prediction": strategy.replace("-", "_")}
            )
        merged = strategy_scores["masked-marginal"].merge(
            strategy_scores["wild-type-marginal"],
            on="mutation_codes",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(variants):
            raise RuntimeError(f"ESM strategy join lost variants for sequence {digest}")
        merged["sequence_sha256"] = str(digest)
        predictions.append(merged)
        audit_rows.append(
            {
                "sequence_sha256": str(digest),
                "sequence_length": len(sequence),
                "mutated_positions": len(positions),
                "unique_variants": len(variants),
                "model": "esm2_t6_8M_UR50D",
                "strategies": "masked-marginal;wild-type-marginal",
                "window_size": 1022,
                "window_overlap": 256,
                "batch_size": batch_size,
                "device": device,
                "torch_version": torch.__version__,
                "fair_esm_version": esm.__version__,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if prediction_frame[["sequence_sha256", "mutation_codes"]].duplicated().any():
        raise RuntimeError("External ESM cache contains duplicate sequence variants")
    return prediction_frame, pd.DataFrame(audit_rows)


def _fixed_rank_position_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observed_rank = frame["DMS_score"].rank(method="average").to_numpy(dtype=float)
    predicted_rank = frame[prediction_column].rank(method="average").to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=int)
    unique_positions, inverse = np.unique(positions, return_inverse=True)
    count = np.bincount(inverse).astype(float)
    sum_x = np.bincount(inverse, weights=observed_rank)
    sum_y = np.bincount(inverse, weights=predicted_rank)
    sum_x2 = np.bincount(inverse, weights=observed_rank**2)
    sum_y2 = np.bincount(inverse, weights=predicted_rank**2)
    sum_xy = np.bincount(inverse, weights=observed_rank * predicted_rank)
    output = np.empty(repeats, dtype=float)
    chunk_size = 512
    for start in range(0, repeats, chunk_size):
        size = min(chunk_size, repeats - start)
        draws = rng.integers(0, len(unique_positions), size=(size, len(unique_positions)))
        n = count[draws].sum(axis=1)
        sx = sum_x[draws].sum(axis=1)
        sy = sum_y[draws].sum(axis=1)
        sxx = sum_x2[draws].sum(axis=1) - sx**2 / n
        syy = sum_y2[draws].sum(axis=1) - sy**2 / n
        sxy = sum_xy[draws].sum(axis=1) - sx * sy / n
        denominator = np.sqrt(sxx * syy)
        output[start : start + size] = np.divide(
            sxy,
            denominator,
            out=np.zeros_like(sxy),
            where=denominator > 0,
        )
    return output


def evaluate_external_cohort(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 2_026_0829,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute assay, protein-balanced, and nested position-bootstrap results."""
    merged = cohort.merge(
        predictions,
        on=["sequence_sha256", "mutation_codes"],
        how="left",
        validate="many_to_one",
    )
    prediction_columns = ["masked_marginal", "wild_type_marginal"]
    if merged[prediction_columns].isna().any().any():
        raise RuntimeError("External predictions are incomplete after the sequence/variant join")
    assay_rows = []
    grouped_assays = list(merged.groupby("urn", sort=True))
    for urn, assay in grouped_assays:
        identity = assay.iloc[0]
        for prediction_column in prediction_columns:
            current = assay.rename(columns={prediction_column: "prediction"})
            metrics = evaluate_external_predictions(current)
            assay_rows.append(
                {
                    "urn": urn,
                    "target_name": identity["target_name"],
                    "protein_id": identity["protein_id"],
                    "model": prediction_column,
                    "variants": len(assay),
                    "mutated_positions": int(assay["position"].nunique()),
                    **metrics,
                }
            )
    assay_metrics = pd.DataFrame(assay_rows)
    metric_columns = ["spearman", "top_recall", "selection_gain_sd", "best_variant_regret_sd"]
    protein_metrics = (
        assay_metrics.groupby(["protein_id", "model"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["model", "protein_id"])
    )
    point_summary = (
        protein_metrics.groupby("model", as_index=False)[metric_columns]
        .mean()
        .rename(columns={column: f"mean_{column}" for column in metric_columns})
    )

    assay_bootstraps: dict[tuple[str, str], np.ndarray] = {}
    for assay_index, (urn, assay) in enumerate(grouped_assays):
        for prediction_column in prediction_columns:
            assay_bootstraps[(str(urn), prediction_column)] = _fixed_rank_position_bootstrap(
                assay,
                prediction_column=prediction_column,
                repeats=bootstrap_repeats,
                rng=np.random.default_rng(
                    np.random.SeedSequence([bootstrap_seed, assay_index])
                ),
            )
    assays_by_protein = {
        str(protein): sorted(group["urn"].astype(str).unique())
        for protein, group in assay_metrics.groupby("protein_id", sort=True)
    }
    proteins = sorted(assays_by_protein)
    nested_rng = np.random.default_rng(np.random.SeedSequence([bootstrap_seed, 999_999]))
    assay_draws = {
        protein: nested_rng.integers(
            0,
            len(urns),
            size=(bootstrap_repeats, len(urns)),
        )
        for protein, urns in assays_by_protein.items()
    }
    protein_draws = nested_rng.integers(
        0, len(proteins), size=(bootstrap_repeats, len(proteins))
    )
    nested: dict[str, np.ndarray] = {}
    for model in prediction_columns:
        protein_values = np.empty((len(proteins), bootstrap_repeats), dtype=float)
        for protein_index, protein in enumerate(proteins):
            urns = assays_by_protein[protein]
            stacked = np.vstack([assay_bootstraps[(urn, model)] for urn in urns]).T
            protein_values[protein_index] = np.take_along_axis(
                stacked, assay_draws[protein], axis=1
            ).mean(axis=1)
        nested[model] = np.take_along_axis(protein_values.T, protein_draws, axis=1).mean(axis=1)
    bootstrap_frame = pd.DataFrame(
        {
            "bootstrap_index": np.arange(bootstrap_repeats),
            **nested,
            "masked_minus_wild_type": (
                nested["masked_marginal"] - nested["wild_type_marginal"]
            ),
        }
    )
    interval_rows = []
    for model in prediction_columns:
        values = nested[model]
        interval_rows.append(
            {
                "model": model,
                "mean_spearman": float(
                    point_summary.loc[point_summary["model"].eq(model), "mean_spearman"].item()
                ),
                "bootstrap_mean": float(values.mean()),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "proteins": len(proteins),
                "assays": int(assay_metrics.loc[assay_metrics["model"].eq(model), "urn"].nunique()),
                "success_lower_bound_above_zero": bool(np.quantile(values, 0.025) > 0),
            }
        )
    difference = bootstrap_frame["masked_minus_wild_type"].to_numpy()
    interval_rows.append(
        {
            "model": "masked_minus_wild_type",
            "mean_spearman": float(
                point_summary.loc[
                    point_summary["model"].eq("masked_marginal"), "mean_spearman"
                ].item()
                - point_summary.loc[
                    point_summary["model"].eq("wild_type_marginal"), "mean_spearman"
                ].item()
            ),
            "bootstrap_mean": float(difference.mean()),
            "ci_lower": float(np.quantile(difference, 0.025)),
            "ci_upper": float(np.quantile(difference, 0.975)),
            "proteins": len(proteins),
            "assays": int(assay_metrics["urn"].nunique()),
            "success_lower_bound_above_zero": bool(np.quantile(difference, 0.025) > 0),
        }
    )
    interval_summary = pd.DataFrame(interval_rows)
    return assay_metrics, protein_metrics, point_summary, interval_summary, bootstrap_frame
