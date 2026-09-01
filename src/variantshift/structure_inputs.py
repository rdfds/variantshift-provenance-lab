"""Outcome-blind AlphaFold structure acquisition for frozen Domainome targets."""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from .outcome_lock import assert_target_only
from .provenance import sha256_file
from .schemas import validate_targets, write_table

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
UNIPROT_SEARCH_API = "https://rest.uniprot.org/uniprotkb/search"
_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",
}


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    return gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload


def crop_alphafold_pdb(
    payload: bytes,
    *,
    start: int,
    sequence: str,
) -> tuple[bytes, float]:
    """Crop and locally renumber one AlphaFold PDB after sequence validation."""
    end = start + len(sequence) - 1
    atom_lines: list[str] = []
    residues: dict[int, str] = {}
    plddt: dict[int, float] = {}
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exception:
        raise ValueError("AlphaFold PDB is not ASCII") from exception
    for line in text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 66:
            continue
        if line[21] != "A":
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        if not start <= residue_number <= end:
            continue
        residue_name = line[17:20].strip()
        amino_acid = _THREE_TO_ONE.get(residue_name)
        if amino_acid is None:
            raise ValueError(f"Unsupported residue {residue_name} in AlphaFold PDB")
        previous = residues.setdefault(residue_number, amino_acid)
        if previous != amino_acid:
            raise ValueError(f"Conflicting residue identity at position {residue_number}")
        local_number = residue_number - start + 1
        atom_lines.append(f"{line[:22]}{local_number:4d}{line[26:]}")
        if line[12:16].strip() == "CA":
            plddt[residue_number] = float(line[60:66])
    observed_positions = list(range(start, end + 1))
    observed_sequence = "".join(residues.get(position, "") for position in observed_positions)
    if observed_sequence != sequence:
        raise ValueError("AlphaFold crop sequence differs from the frozen target")
    if len(plddt) != len(sequence):
        raise ValueError("AlphaFold crop lacks one or more CA confidence values")
    cropped = (
        "REMARK 950 VARIANTSHIFT OUTCOME-BLIND ALPHAFOLD CROP\n"
        + "\n".join(atom_lines)
        + "\nTER\nEND\n"
    ).encode("ascii")
    return cropped, sum(plddt.values()) / len(plddt)


def crop_exact_pdb_chain(payload: bytes, *, sequence: str) -> tuple[bytes, str]:
    """Select, validate, and locally renumber an exact experimental PDB chain.

    Only residues supported by coordinate records are considered.  This deliberately
    rejects SEQRES-only residues because structure-conditioned models and Foldseek
    cannot use unresolved coordinates.
    """
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exception:
        raise ValueError("PDB is not ASCII") from exception

    residues_by_chain: dict[str, dict[tuple[int, str], dict[str, object]]] = {}
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        residue_name = line[17:20].strip()
        if line.startswith("HETATM") and residue_name != "MSE":
            continue
        amino_acid = _THREE_TO_ONE.get(residue_name)
        if amino_acid is None:
            continue
        alternate_location = line[16]
        if alternate_location not in {" ", "A"}:
            continue
        chain_id = line[21]
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        key = (residue_number, line[26])
        chain = residues_by_chain.setdefault(chain_id, {})
        residue = chain.setdefault(
            key,
            {"amino_acid": amino_acid, "atoms": set(), "lines": []},
        )
        if residue["amino_acid"] != amino_acid:
            raise ValueError(
                f"Conflicting residue identity for chain {chain_id!r} residue {key}"
            )
        atom_name = line[12:16].strip()
        atoms = residue["atoms"]
        assert isinstance(atoms, set)
        atoms.add(atom_name)
        lines = residue["lines"]
        assert isinstance(lines, list)
        lines.append(line)

    exact_chains: list[tuple[str, list[dict[str, object]]]] = []
    incomplete_exact_chains: list[str] = []
    for chain_id, residue_map in residues_by_chain.items():
        ordered = list(residue_map.values())
        observed = "".join(str(residue["amino_acid"]) for residue in ordered)
        if observed != sequence:
            continue
        complete = all(
            {"N", "CA", "C"}.issubset(residue["atoms"])  # type: ignore[arg-type]
            for residue in ordered
        )
        if complete:
            exact_chains.append((chain_id, ordered))
        else:
            incomplete_exact_chains.append(chain_id)

    if not exact_chains:
        if incomplete_exact_chains:
            rendered = ", ".join(repr(chain) for chain in sorted(incomplete_exact_chains))
            raise ValueError(f"Exact chain lacks complete N/CA/C backbone: {rendered}")
        raise ValueError("No coordinate-resolved PDB chain exactly matches the frozen target")

    chain_id, residues = min(exact_chains, key=lambda item: item[0])
    output_lines = [
        f"REMARK 950 VARIANTSHIFT OUTCOME-BLIND RCSB EXACT-CHAIN {chain_id or 'blank'}"
    ]
    for local_number, residue in enumerate(residues, start=1):
        lines = residue["lines"]
        assert isinstance(lines, list)
        for line in lines:
            # Normalize every selected chain to A and remove insertion codes so all
            # downstream tools share the same one-based local coordinate system.
            output_lines.append(f"{line[:21]}A{local_number:4d} {line[27:]}")
    output_lines.extend(["TER", "END"])
    return ("\n".join(output_lines) + "\n").encode("ascii"), chain_id


def freeze_domainome_structures(
    targets_path: Path,
    output_dir: Path,
    *,
    workers: int = 12,
) -> dict[str, Path]:
    """Fetch current AlphaFold models and freeze validated target-domain crops."""
    targets_path = Path(targets_path)
    targets = pd.read_csv(targets_path)
    validate_targets(targets)
    assert_target_only(targets)
    protein_ids = sorted(targets["protein_id"].astype(str).unique())
    metadata_by_protein: dict[str, dict[str, object]] = {}
    metadata_errors: dict[str, str] = {}

    def metadata_job(protein_id: str) -> tuple[str, dict[str, object]]:
        records = _fetch_json(f"{ALPHAFOLD_API}/{protein_id}")
        if not isinstance(records, list) or not records:
            raise ValueError("no AlphaFold record")
        exact = [
            record
            for record in records
            if str(record.get("uniprotAccession")) == protein_id and record.get("pdbUrl")
        ]
        if not exact:
            raise ValueError("no exact-accession AlphaFold PDB")
        exact.sort(
            key=lambda record: (
                int(record.get("sequenceEnd") or 0),
                str(record.get("modelCreatedDate") or ""),
            ),
            reverse=True,
        )
        return protein_id, exact[0]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(metadata_job, protein_id): protein_id for protein_id in protein_ids
        }
        for future in as_completed(futures):
            protein_id = futures[future]
            try:
                key, metadata = future.result()
                metadata_by_protein[key] = metadata
            except Exception as exception:  # noqa: BLE001 - retain per-target exclusions
                metadata_errors[protein_id] = f"{type(exception).__name__}: {exception}"

    pdb_by_url: dict[str, bytes] = {}
    pdb_errors: dict[str, str] = {}
    urls = sorted({str(item["pdbUrl"]) for item in metadata_by_protein.values()})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_bytes, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                pdb_by_url[url] = future.result()
            except Exception as exception:  # noqa: BLE001 - retain per-target exclusions
                pdb_errors[url] = f"{type(exception).__name__}: {exception}"

    output_dir = Path(output_dir)
    structure_dir = output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for target in targets.sort_values("target_id").itertuples(index=False):
        protein_id = str(target.protein_id)
        metadata = metadata_by_protein.get(protein_id)
        row = {
            "target_id": str(target.target_id),
            "protein_id": protein_id,
            "sequence_sha256": str(target.sequence_sha256),
            "source": "AlphaFold DB",
            "source_api": f"{ALPHAFOLD_API}/{protein_id}",
            "status": "excluded",
            "exclusion_reason": metadata_errors.get(protein_id, ""),
        }
        if metadata is None:
            audit_rows.append(row)
            continue
        url = str(metadata["pdbUrl"])
        payload = pdb_by_url.get(url)
        row.update(
            {
                "entry_id": str(metadata.get("entryId") or ""),
                "model_created_date": str(metadata.get("modelCreatedDate") or ""),
                "pdb_url": url,
                "pdb_source_sha256": (
                    hashlib.sha256(payload).hexdigest() if payload is not None else ""
                ),
            }
        )
        if payload is None:
            row["exclusion_reason"] = pdb_errors.get(url, "PDB download failed")
            audit_rows.append(row)
            continue
        try:
            start = int(str(target.target_id).rsplit("_", 1)[1])
            cropped, mean_plddt = crop_alphafold_pdb(
                payload,
                start=start,
                sequence=str(target.sequence),
            )
            destination = structure_dir / f"{target.target_id}.pdb"
            destination.write_bytes(cropped)
            row.update(
                {
                    "status": "usable",
                    "exclusion_reason": "",
                    "domain_start": start,
                    "domain_end": start + len(str(target.sequence)) - 1,
                    "mean_plddt": mean_plddt,
                    "structure_path": str(destination),
                    "structure_sha256": hashlib.sha256(cropped).hexdigest(),
                }
            )
        except Exception as exception:  # noqa: BLE001 - retain sequence mismatches
            row["exclusion_reason"] = f"{type(exception).__name__}: {exception}"
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    outputs = {
        "audit": output_dir / "structure-input-audit.csv",
        "manifest": output_dir / "structure-input-manifest.json",
    }
    write_table(audit, outputs["audit"])
    usable = audit.loc[audit["status"].eq("usable")]
    manifest = {
        "schema_version": 1,
        "source": "AlphaFold Protein Structure Database API",
        "source_api": ALPHAFOLD_API,
        "outcomes_accessed": False,
        "target_sha256": sha256_file(targets_path),
        "target_count": len(targets),
        "usable_target_count": len(usable),
        "coverage": len(usable) / len(targets),
        "structure_hashes": dict(zip(usable["target_id"], usable["structure_sha256"], strict=True)),
        "audit_sha256": sha256_file(outputs["audit"]),
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def _uniprot_candidates(protein_id: str, sequence: str) -> list[dict[str, str]]:
    token = protein_id.split()[0]
    accession_like = (
        token.split("-", maxsplit=1)[0].isalnum()
        and len(token.split("-", maxsplit=1)[0]) in {6, 10}
        and any(character.isdigit() for character in token)
    )
    query = f"accession:{token}" if accession_like else f"gene_exact:{token} AND organism_id:9606"
    parameters = urlencode(
        {
            "query": query,
            "format": "json",
            "fields": "accession,id,sequence,reviewed",
            "size": "100",
        }
    )
    url = f"{UNIPROT_SEARCH_API}?{parameters}"
    payload = _fetch_json(url)
    rows = []
    for result in payload.get("results", []):
        observed = str(result.get("sequence", {}).get("value") or "").upper()
        if observed != sequence:
            continue
        rows.append(
            {
                "accession": str(result["primaryAccession"]),
                "entry_name": str(result.get("uniProtkbId") or ""),
                "reviewed": str(result.get("entryType") or "").lower().startswith(
                    "uniprotkb reviewed"
                ),
                "resolution_url": url,
            }
        )
    return sorted(rows, key=lambda row: (not bool(row["reviewed"]), row["accession"]))


def freeze_confirmation_structures(
    targets_path: Path,
    output_dir: Path,
    *,
    workers: int = 12,
) -> dict[str, Path]:
    """Freeze full-length AlphaFold models after exact UniProt-sequence resolution."""
    targets_path = Path(targets_path)
    targets = pd.read_csv(targets_path)
    validate_targets(targets)
    assert_target_only(targets)

    def resolve_job(row: object) -> tuple[str, dict[str, object]]:
        target_id = str(row.target_id)
        if str(getattr(row, "source_type", "")).lower() == "pdb":
            pdb_id = str(row.protein_id).upper()
            return target_id, {
                "source_kind": "pdb",
                "pdb_id": pdb_id,
                "pdb_url": f"https://files.rcsb.org/download/{pdb_id}.pdb",
            }
        candidates = _uniprot_candidates(str(row.protein_id), str(row.sequence))
        if not candidates:
            raise ValueError("no exact-sequence UniProt accession")
        if len(candidates) > 1 and candidates[0]["reviewed"] == candidates[1]["reviewed"]:
            raise ValueError("multiple equally ranked exact-sequence UniProt accessions")
        resolved = candidates[0]
        records = _fetch_json(f"{ALPHAFOLD_API}/{resolved['accession']}")
        exact = [
            record
            for record in records
            if str(record.get("uniprotAccession")) == resolved["accession"]
            and record.get("pdbUrl")
        ]
        if not exact:
            raise ValueError("no exact-accession AlphaFold PDB")
        exact.sort(
            key=lambda record: (
                int(record.get("sequenceEnd") or 0),
                str(record.get("modelCreatedDate") or ""),
            ),
            reverse=True,
        )
        return target_id, {**resolved, "source_kind": "alphafold", "alphafold": exact[0]}

    resolved: dict[str, dict[str, object]] = {}
    resolution_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(resolve_job, row): str(row.target_id)
            for row in targets.itertuples(index=False)
        }
        for future in as_completed(futures):
            target_id = futures[future]
            try:
                key, record = future.result()
                resolved[key] = record
            except Exception as exception:  # noqa: BLE001 - explicit exclusion audit
                resolution_errors[target_id] = f"{type(exception).__name__}: {exception}"

    urls = sorted(
        {
            (
                str(item["pdb_url"])
                if item["source_kind"] == "pdb"
                else str(item["alphafold"]["pdbUrl"])
            )
            for item in resolved.values()
        }
    )
    payloads: dict[str, bytes] = {}
    payload_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_bytes, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                payloads[url] = future.result()
            except Exception as exception:  # noqa: BLE001 - explicit exclusion audit
                payload_errors[url] = f"{type(exception).__name__}: {exception}"

    output_dir = Path(output_dir)
    structure_dir = output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for target in targets.sort_values("target_id").itertuples(index=False):
        target_id = str(target.target_id)
        resolution = resolved.get(target_id)
        row: dict[str, object] = {
            "panel_id": str(target.panel_id),
            "target_id": target_id,
            "protein_id": str(target.protein_id),
            "sequence_sha256": str(target.sequence_sha256),
            "source": "AlphaFold DB",
            "status": "excluded",
            "exclusion_reason": resolution_errors.get(target_id, ""),
        }
        if resolution is None:
            audit_rows.append(row)
            continue
        if resolution["source_kind"] == "pdb":
            url = str(resolution["pdb_url"])
            payload = payloads.get(url)
            row.update(
                {
                    "source": "RCSB Protein Data Bank",
                    "source_api": url,
                    "entry_id": resolution["pdb_id"],
                    "pdb_url": url,
                    "pdb_source_sha256": (
                        hashlib.sha256(payload).hexdigest() if payload is not None else ""
                    ),
                }
            )
            if payload is None:
                row["exclusion_reason"] = payload_errors.get(url, "PDB download failed")
                audit_rows.append(row)
                continue
            try:
                cropped, chain_id = crop_exact_pdb_chain(payload, sequence=str(target.sequence))
                destination = structure_dir / f"{target_id}.pdb"
                destination.write_bytes(cropped)
                row.update(
                    {
                        "status": "usable",
                        "exclusion_reason": "",
                        "selected_chain_id": chain_id,
                        "domain_start": 1,
                        "domain_end": len(str(target.sequence)),
                        "structure_path": str(destination),
                        "structure_sha256": hashlib.sha256(cropped).hexdigest(),
                    }
                )
            except Exception as exception:  # noqa: BLE001 - explicit exclusion audit
                row["exclusion_reason"] = f"{type(exception).__name__}: {exception}"
            audit_rows.append(row)
            continue
        metadata = resolution["alphafold"]
        url = str(metadata["pdbUrl"])
        payload = payloads.get(url)
        row.update(
            {
                "resolved_uniprot_accession": resolution["accession"],
                "resolved_uniprot_entry_name": resolution["entry_name"],
                "resolved_uniprot_reviewed": resolution["reviewed"],
                "resolution_url": resolution["resolution_url"],
                "source_api": f"{ALPHAFOLD_API}/{resolution['accession']}",
                "entry_id": str(metadata.get("entryId") or ""),
                "model_created_date": str(metadata.get("modelCreatedDate") or ""),
                "pdb_url": url,
                "pdb_source_sha256": (
                    hashlib.sha256(payload).hexdigest() if payload is not None else ""
                ),
            }
        )
        if payload is None:
            row["exclusion_reason"] = payload_errors.get(url, "PDB download failed")
            audit_rows.append(row)
            continue
        try:
            cropped, mean_plddt = crop_alphafold_pdb(
                payload, start=1, sequence=str(target.sequence)
            )
            destination = structure_dir / f"{target_id}.pdb"
            destination.write_bytes(cropped)
            row.update(
                {
                    "status": "usable",
                    "exclusion_reason": "",
                    "domain_start": 1,
                    "domain_end": len(str(target.sequence)),
                    "mean_plddt": mean_plddt,
                    "structure_path": str(destination),
                    "structure_sha256": hashlib.sha256(cropped).hexdigest(),
                }
            )
        except Exception as exception:  # noqa: BLE001 - explicit exclusion audit
            row["exclusion_reason"] = f"{type(exception).__name__}: {exception}"
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    outputs = {
        "audit": output_dir / "structure-input-audit.csv",
        "manifest": output_dir / "structure-input-manifest.json",
    }
    write_table(audit, outputs["audit"])
    usable = audit.loc[audit["status"].eq("usable")]
    manifest = {
        "schema_version": 1,
        "source": "AlphaFold Protein Structure Database and UniProt REST APIs",
        "outcomes_accessed": False,
        "exact_sequence_resolution_required": True,
        "target_sha256": sha256_file(targets_path),
        "target_count": len(targets),
        "usable_target_count": len(usable),
        "coverage": len(usable) / len(targets),
        "structure_hashes": dict(
            zip(usable["target_id"], usable["structure_sha256"], strict=True)
        ),
        "audit_sha256": sha256_file(outputs["audit"]),
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
