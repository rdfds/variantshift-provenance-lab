"""Freeze a small, outcome-independent ProteinGym panel for implementation parity."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .provenance import sha256_file
from .schemas import (
    TARGET_SCHEMA,
    all_single_substitutions,
    sequence_sha256,
    validate_targets,
    write_table,
)

_MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")
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
}


def _pdb_chain_a_sequence(payload: bytes) -> str:
    residues: dict[tuple[str, str], str] = {}
    for line in payload.decode("ascii", errors="strict").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27 or line[21:22] != "A":
            continue
        residue = _THREE_TO_ONE.get(line[17:20].strip())
        if residue is None:
            continue
        key = (line[22:26].strip(), line[26:27].strip())
        previous = residues.setdefault(key, residue)
        if previous != residue:
            raise ValueError(f"Conflicting residue identity at PDB key {key}")
    return "".join(residues.values())


def _validate_reference_mutations(frame: pd.DataFrame, sequence: str, dms_id: str) -> None:
    for mutation in frame["variant_id"].astype(str):
        match = _MUTATION.fullmatch(mutation)
        if match is None:
            raise ValueError(f"Unexpected ProteinGym mutation for {dms_id}: {mutation}")
        reference, position, alternate = match.groups()
        index = int(position) - 1
        if index < 0 or index >= len(sequence) or sequence[index] != reference:
            raise ValueError(f"ProteinGym mutation disagrees with {dms_id} sequence: {mutation}")
        if alternate == reference:
            raise ValueError(f"ProteinGym mutation is not a substitution: {mutation}")


def freeze_proteingym_parity_panel(
    qualification_config: Path,
    metadata_path: Path,
    score_archive: Path,
    structure_archive: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Freeze targets, structures, and model-score references without reading DMS outcomes."""
    qualification_config = Path(qualification_config)
    metadata_path = Path(metadata_path)
    score_archive = Path(score_archive)
    structure_archive = Path(structure_archive)
    output_dir = Path(output_dir)
    config = json.loads(qualification_config.read_text(encoding="utf-8"))
    panel = config["parity_panel"]
    requested = list(map(str, panel["targets"]))
    if len(requested) != len(set(requested)):
        raise ValueError("Qualification parity targets must be unique")
    models = config["models"]
    model_ids = [str(item["model_id"]) for item in models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Qualification model identifiers must be unique")

    metadata = pd.read_csv(metadata_path)
    selected = metadata.loc[metadata["DMS_id"].astype(str).isin(requested)].copy()
    missing = sorted(set(requested).difference(selected["DMS_id"].astype(str)))
    if missing:
        raise ValueError(f"ProteinGym metadata is missing parity targets: {missing}")
    selected = selected.set_index("DMS_id").loc[requested].reset_index()
    if selected["UniProt_ID"].astype(str).duplicated().any():
        raise ValueError("Parity targets must map one-to-one to ProteinGym proteins")

    target_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        sequence = str(row.target_seq).removesuffix("*").upper()
        target_rows.append(
            {
                "panel_id": str(panel["panel_id"]),
                "target_id": str(row.UniProt_ID),
                "protein_id": str(row.UniProt_ID),
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "sequence_length": len(sequence),
            }
        )
    targets = pd.DataFrame(target_rows, columns=TARGET_SCHEMA.required)
    validate_targets(targets)
    variants = all_single_substitutions(targets)

    structure_dir = output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    structure_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(structure_archive) as structures:
        members = set(structures.namelist())
        for row in selected.itertuples(index=False):
            member = f"ProteinGym_AF2_structures/{row.pdb_file}"
            if member not in members:
                raise ValueError(f"ProteinGym structure archive is missing {member}")
            payload = structures.read(member)
            sequence = str(row.target_seq).removesuffix("*").upper()
            observed = _pdb_chain_a_sequence(payload)
            if observed != sequence:
                raise ValueError(
                    f"ProteinGym structure sequence mismatch for {row.DMS_id}: "
                    f"{len(observed)} versus {len(sequence)} residues"
                )
            destination = structure_dir / f"{row.UniProt_ID}.pdb"
            destination.write_bytes(payload)
            structure_rows.append(
                {
                    "dms_id": str(row.DMS_id),
                    "target_id": str(row.UniProt_ID),
                    "source_member": member,
                    "sequence_sha256": sequence_sha256(sequence),
                    "structure_sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "usable",
                    "error": "",
                }
            )

    official_rows: list[pd.DataFrame] = []
    score_columns = [str(item["official_column"]) for item in models]
    with zipfile.ZipFile(score_archive) as scores:
        members = set(scores.namelist())
        for row in selected.itertuples(index=False):
            filename = str(row.DMS_filename)
            if filename not in members:
                raise ValueError(f"ProteinGym score archive is missing {filename}")
            current = pd.read_csv(scores.open(filename), usecols=["mutant", *score_columns])
            current = current.loc[
                current["mutant"].astype(str).str.fullmatch(_MUTATION.pattern)
            ].copy()
            sequence = str(row.target_seq).removesuffix("*").upper()
            for model in models:
                model_id = str(model["model_id"])
                score_column = str(model["official_column"])
                reference = current.loc[current[score_column].notna(), ["mutant", score_column]]
                reference = reference.rename(
                    columns={"mutant": "variant_id", score_column: "official_score"}
                )
                _validate_reference_mutations(reference, sequence, str(row.DMS_id))
                reference.insert(0, "model_id", model_id)
                reference.insert(0, "target_id", str(row.UniProt_ID))
                reference.insert(0, "dms_id", str(row.DMS_id))
                official_rows.append(reference)
    official = pd.concat(official_rows, ignore_index=True)
    if official.duplicated(["dms_id", "target_id", "variant_id", "model_id"]).any():
        raise ValueError("ProteinGym parity references contain duplicate model-variant rows")

    outputs = {
        "targets": output_dir / "targets.csv",
        "variants": output_dir / "variants.csv",
        "official_scores": output_dir / "official-scores.csv.gz",
        "structure_audit": output_dir / "structure-audit.csv",
        "structure_manifest": output_dir / "structure-manifest.json",
        "manifest": output_dir / "panel-manifest.json",
    }
    write_table(targets, outputs["targets"])
    write_table(variants, outputs["variants"])
    write_table(official, outputs["official_scores"])
    structure_audit = pd.DataFrame(structure_rows)
    write_table(structure_audit, outputs["structure_audit"])
    structure_manifest = {
        "schema_version": 1,
        "source": "ProteinGym_AF2_structures.zip",
        "source_sha256": sha256_file(structure_archive),
        "target_count": len(targets),
        "usable_target_count": len(structure_audit),
        "structure_hashes": dict(
            zip(
                structure_audit["target_id"],
                structure_audit["structure_sha256"],
                strict=True,
            )
        ),
        "audit_sha256": sha256_file(outputs["structure_audit"]),
    }
    outputs["structure_manifest"].write_text(
        json.dumps(structure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": str(config["protocol_id"]),
        "panel_id": str(panel["panel_id"]),
        "target_count": len(targets),
        "variant_count": len(variants),
        "official_score_rows": len(official),
        "models": model_ids,
        "dms_ids": requested,
        "columns_read_from_score_archive": ["mutant", *score_columns],
        "dms_outcome_columns_read": False,
        "confirmation_outcomes_accessed": False,
        "inputs": {
            "qualification_config_sha256": sha256_file(qualification_config),
            "metadata_sha256": sha256_file(metadata_path),
            "score_archive_sha256": sha256_file(score_archive),
            "structure_archive_sha256": sha256_file(structure_archive),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                outputs["targets"],
                outputs["variants"],
                outputs["official_scores"],
                outputs["structure_audit"],
                outputs["structure_manifest"],
            )
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
