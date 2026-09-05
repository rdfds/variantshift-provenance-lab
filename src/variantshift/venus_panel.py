"""Outcome-free target acquisition for the VenusMutHub stress panel."""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .confirmation_panels import _publication_dois
from .provenance import sha256_file
from .schemas import AMINO_ACIDS, TARGET_SCHEMA, sequence_sha256, validate_targets, write_table

HF_DATASET = "AI4Protein/VenusMutHub"
HF_REVISION = "main"
HF_TREE_URL = (
    "https://huggingface.co/api/datasets/AI4Protein/VenusMutHub/tree/"
    "main/single_mutant?recursive=true&expand=false&limit=1000"
)
HF_DATASET_API = "https://huggingface.co/api/datasets/AI4Protein/VenusMutHub"
HF_DOI_URL = (
    "https://huggingface.co/datasets/AI4Protein/VenusMutHub/resolve/main/"
    "mutant/doi.csv?download=true"
)
RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
_PDB_TOKEN = re.compile(r"(?:^|_)(?P<pdb>[0-9][A-Za-z0-9]{3})(?:_|$)")
_PDB_CHAIN = re.compile(
    r"^PPB_Affinity_(?P<pdb>[0-9][A-Za-z0-9]{3})_(?P<chain>[A-Za-z0-9])(?:_|$)"
)
_UNIPROT_TOKEN = re.compile(
    r"(?:^|_)(?P<uniprot>(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|A0A[A-Z0-9]{7}))(?:_|$)"
)


def _fetch_bytes(url: str, *, attempts: int = 4) -> bytes:
    request = Request(url, headers={"User-Agent": "VariantShift/1.0 target-only client"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (HTTPError, URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable target-metadata retry state")


@lru_cache(maxsize=4096)
def _fetch_json(url: str) -> object:
    return json.loads(_fetch_bytes(url))


def normalize_doi(value: str) -> str:
    text = str(value).strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.rstrip("/.,")


def parse_venus_source_identifier(dataset_id: str) -> tuple[str, str] | None:
    text = str(dataset_id)
    text = text.removeprefix("vb1432_")
    chain_match = _PDB_CHAIN.match(text)
    if chain_match:
        return (
            "pdb_chain",
            f"{chain_match.group('pdb').upper()}:{chain_match.group('chain').upper()}",
        )
    match = _PDB_TOKEN.search(text)
    if match:
        return "pdb", match.group("pdb").upper()
    match = _UNIPROT_TOKEN.search(text)
    if match:
        return "uniprot", match.group("uniprot").upper()
    return None


def _normalize_sequence(value: str) -> str | None:
    sequence = re.sub(r"\s+", "", str(value)).upper().removesuffix("*")
    return sequence if sequence and not set(sequence).difference(AMINO_ACIDS) else None


def resolve_pdb_sequence(pdb_id: str, *, chain: str | None = None) -> dict[str, object]:
    entry = _fetch_json(f"{RCSB_BASE}/entry/{pdb_id}")
    entity_ids = entry["rcsb_entry_container_identifiers"]["polymer_entity_ids"]
    sequences = []
    references = set()
    for entity_id in entity_ids:
        entity = _fetch_json(f"{RCSB_BASE}/polymer_entity/{pdb_id}/{entity_id}")
        entity_poly = entity.get("entity_poly") or {}
        if entity_poly.get("rcsb_entity_polymer_type") != "Protein":
            continue
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        if chain is not None and chain not in set(identifiers.get("auth_asym_ids") or []):
            continue
        sequence = _normalize_sequence(entity_poly.get("pdbx_seq_one_letter_code_can") or "")
        if sequence:
            sequences.append(sequence)
        for reference in identifiers.get("reference_sequence_identifiers") or []:
            if reference.get("database_name") == "UniProt":
                references.add(str(reference.get("database_accession")))
    unique = sorted(set(sequences))
    return {
        "source_type": "pdb_chain" if chain else "pdb",
        "source_identifier": f"{pdb_id}:{chain}" if chain else pdb_id,
        "sequence": unique[0] if len(unique) == 1 else None,
        "sequence_status": (
            "resolved"
            if len(unique) == 1
            else "chain_not_found"
            if chain and not unique
            else "ambiguous_polymer_entities"
        ),
        "protein_polymer_sequences": len(unique),
        "reference_accessions": ";".join(sorted(references)),
    }


def resolve_uniprot_sequence(accession: str) -> dict[str, object]:
    payload = _fetch_bytes(f"{UNIPROT_BASE}/{accession}.fasta").decode("utf-8")
    sequence = _normalize_sequence(
        "".join(line for line in payload.splitlines() if not line.startswith(">"))
    )
    return {
        "source_type": "uniprot",
        "source_identifier": accession,
        "sequence": sequence,
        "sequence_status": "resolved" if sequence else "noncanonical_sequence",
        "protein_polymer_sequences": 1 if sequence else 0,
        "reference_accessions": accession,
    }


def resolve_venus_sequences(
    identifiers: list[tuple[str, str]], *, workers: int = 12
) -> pd.DataFrame:
    unique = sorted(set(identifiers))

    def resolve(item: tuple[str, str]) -> dict[str, object]:
        source_type, identifier = item
        try:
            if source_type == "pdb":
                return resolve_pdb_sequence(identifier)
            if source_type == "pdb_chain":
                pdb_id, chain = identifier.split(":", 1)
                return resolve_pdb_sequence(pdb_id, chain=chain)
            return resolve_uniprot_sequence(identifier)
        except Exception as exception:  # noqa: BLE001 - audit every unresolved target
            return {
                "source_type": source_type,
                "source_identifier": identifier,
                "sequence": None,
                "sequence_status": f"failed:{type(exception).__name__}",
                "protein_polymer_sequences": 0,
                "reference_accessions": "",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(resolve, unique))
    return pd.DataFrame(rows).sort_values(["source_type", "source_identifier"])


def _development_sequences(
    proteingym_reference: pd.DataFrame, mavedb_metadata: list[dict[str, object]]
) -> set[str]:
    sequences = set(
        proteingym_reference["target_seq"].dropna().astype(str).str.upper().str.removesuffix("*")
    )
    for detail in mavedb_metadata:
        for target in detail.get("targetGenes") or []:
            record = target.get("targetSequence") or {}
            if record.get("sequenceType") == "protein":
                sequence = _normalize_sequence(record.get("sequence") or "")
                if sequence:
                    sequences.add(sequence)
    return sequences


def freeze_venusmuthub_targets(
    proteingym_reference_path: Path,
    mavedb_development_metadata_path: Path,
    output_dir: Path,
    *,
    tree_rows: list[dict[str, object]] | None = None,
    doi_frame: pd.DataFrame | None = None,
    resolutions: pd.DataFrame | None = None,
    workers: int = 12,
) -> dict[str, Path]:
    """Freeze unambiguous VenusMutHub targets without reading any mutation file."""
    fixture_mode = tree_rows is not None
    if tree_rows is None:
        tree_payload = _fetch_json(HF_TREE_URL)
        if not isinstance(tree_payload, list):
            raise TypeError("VenusMutHub target tree returned an unexpected payload")
        tree_rows = tree_payload
    if doi_frame is None:
        doi_frame = pd.read_csv(BytesIO(_fetch_bytes(HF_DOI_URL)))
    required_doi = {"mutant_file_id", "doi"}
    if not required_doi.issubset(doi_frame.columns):
        raise ValueError("VenusMutHub DOI table lacks mutant_file_id and doi")

    paths = sorted(
        str(row["path"])
        for row in tree_rows
        if str(row.get("path", "")).startswith("single_mutant/")
        and str(row.get("path", "")).endswith(".csv")
    )
    assay_rows = []
    identifiers = []
    for path in paths:
        parts = path.split("/")
        dataset_id = Path(path).stem
        parsed = parse_venus_source_identifier(dataset_id)
        row = {
            "dataset_id": dataset_id,
            "category": parts[1],
            "source_path": path,
            "source_type": parsed[0] if parsed else "",
            "source_identifier": parsed[1] if parsed else "",
        }
        assay_rows.append(row)
        if parsed:
            identifiers.append(parsed)
    assays = pd.DataFrame(assay_rows)
    if resolutions is None:
        resolutions = resolve_venus_sequences(identifiers, workers=workers)
    assays = assays.merge(
        resolutions,
        on=["source_type", "source_identifier"],
        how="left",
        validate="many_to_one",
    )

    doi_lookup = doi_frame.copy()
    doi_lookup["doi_normalized"] = doi_lookup["doi"].map(normalize_doi)
    assays = assays.merge(
        doi_lookup.loc[:, ["mutant_file_id", "doi_normalized"]],
        left_on="dataset_id",
        right_on="mutant_file_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="mutant_file_id")

    reference = pd.read_csv(proteingym_reference_path)
    mavedb_metadata = json.loads(
        Path(mavedb_development_metadata_path).read_text(encoding="utf-8")
    )
    development_dois = set(reference["jo"].dropna().map(normalize_doi))
    for detail in mavedb_metadata:
        development_dois.update(map(normalize_doi, _publication_dois(detail)))
    development_sequences = _development_sequences(reference, mavedb_metadata)

    reasons = []
    for row in assays.itertuples(index=False):
        row_reasons = []
        if not row.source_identifier:
            row_reasons.append("unparsed_target_identifier")
        elif row.sequence_status != "resolved" or not isinstance(row.sequence, str):
            row_reasons.append(str(row.sequence_status or "unresolved_target_sequence"))
        if not isinstance(row.doi_normalized, str) or not row.doi_normalized:
            row_reasons.append("missing_publication_doi")
        elif row.doi_normalized in development_dois:
            row_reasons.append("development_publication_overlap")
        if isinstance(row.sequence, str) and row.sequence in development_sequences:
            row_reasons.append("development_sequence_overlap")
        reasons.append(";".join(row_reasons))
    assays["exclusion_reasons"] = reasons
    assays["selected"] = assays["exclusion_reasons"].eq("")
    selected = assays.loc[assays["selected"]].copy()
    if selected.empty:
        raise ValueError("No VenusMutHub target passed the target-only overlap audit")

    target_rows = []
    for digest, group in selected.assign(
        sequence_sha256=selected["sequence"].map(sequence_sha256)
    ).groupby("sequence_sha256", sort=True):
        representative = group.iloc[0]
        target_rows.append(
            {
                "panel_id": "venusmuthub-v1",
                "target_id": f"venus-{digest[:16]}",
                "protein_id": str(representative["source_identifier"]),
                "sequence": str(representative["sequence"]),
                "sequence_sha256": digest,
                "sequence_length": len(str(representative["sequence"])),
                "source_type": str(representative["source_type"]),
                "source_assays": ";".join(sorted(group["dataset_id"].astype(str))),
                "publication_dois": ";".join(sorted(group["doi_normalized"].astype(str))),
            }
        )
    targets = validate_targets(pd.DataFrame(target_rows))
    target_map = targets.set_index("sequence_sha256")["target_id"]
    assays["sequence_sha256"] = assays["sequence"].map(
        lambda value: sequence_sha256(value) if isinstance(value, str) else ""
    )
    assays["target_id"] = assays["sequence_sha256"].map(target_map)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": output_dir / "targets.csv",
        "assay_audit": output_dir / "assay-audit.csv",
        "sequence_resolution": output_dir / "sequence-resolution.csv",
        "source_tree": output_dir / "source-tree.json",
        "doi_snapshot": output_dir / "doi.csv",
        "protocol": output_dir / "target-freeze-protocol.json",
    }
    extra = ["source_type", "source_assays", "publication_dois"]
    write_table(targets.loc[:, [*TARGET_SCHEMA.required, *extra]], outputs["targets"])
    write_table(assays.sort_values("dataset_id"), outputs["assay_audit"])
    write_table(resolutions, outputs["sequence_resolution"])
    outputs["source_tree"].write_text(
        json.dumps(tree_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_table(doi_frame, outputs["doi_snapshot"])
    source_revision = "test-fixture"
    if not fixture_mode:
        metadata = _fetch_json(HF_DATASET_API)
        source_revision = str(metadata.get("sha", "undocumented"))
    protocol = {
        "schema_version": 1,
        "protocol_id": "variantshift-venusmuthub-confirmation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": HF_DATASET,
            "revision": source_revision,
            "tree_url": HF_TREE_URL,
        },
        "outcomes_accessed": False,
        "mutation_file_requests": 0,
        "selection": {
            "source": "single_mutant path metadata only",
            "sequence": "unambiguous RCSB protein entity or canonical UniProt sequence",
            "publication": "DOI required and absent from development",
            "overlap": "exact sequence and DOI absent from ProteinGym and MaveDB development",
            "post_reveal": "retain only valid single substitutions matching the frozen sequence",
        },
        "panel": {
            "source_assays": len(assays),
            "selected_assays": int(assays["selected"].sum()),
            "selected_target_sequences": len(targets),
        },
        "artifact_sha256": {
            key: sha256_file(path) for key, path in outputs.items() if key != "protocol"
        },
    }
    outputs["protocol"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
