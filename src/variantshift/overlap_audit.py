"""Outcome-free novelty, family, and model-exposure audits for confirmation targets."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .family_clusters import _DisjointSet
from .model_adapters import load_model_specifications
from .proteingym import read_reference_index
from .provenance import sha256_file
from .schemas import TARGET_SCHEMA, validate_targets, write_table

ALIGNMENT_COLUMNS = (
    "query_sequence_id",
    "target_sequence_id",
    "sequence_identity",
    "alignment_length",
    "query_coverage",
    "target_coverage",
    "e_value",
    "bit_score",
)


def _development_targets(reference_path: Path, eligibility_path: Path) -> pd.DataFrame:
    reference = read_reference_index(reference_path)
    eligibility = pd.read_csv(eligibility_path)
    eligible = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    selected = reference.loc[
        reference["DMS_id"].astype(str).isin(eligible),
        ["DMS_id", "UniProt_ID", "target_seq"],
    ].copy()
    selected = selected.rename(
        columns={"DMS_id": "assay_id", "UniProt_ID": "protein_id", "target_seq": "sequence"}
    )
    selected["sequence"] = selected["sequence"].astype(str).str.removesuffix("*").str.upper()
    selected["sequence_sha256"] = selected["sequence"].map(
        lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    )
    selected["target_id"] = "development-" + selected["sequence_sha256"].str[:16]
    selected["panel_id"] = "proteingym-v1.3-development"
    selected["sequence_length"] = selected["sequence"].str.len()
    return selected.drop_duplicates("target_id").reset_index(drop=True)


def _combined_confirmation_targets(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one confirmation target table is required")
    frames = [validate_targets(pd.read_csv(path)).copy() for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    TARGET_SCHEMA.validate(combined)
    if combined.duplicated(["panel_id", "target_id"]).any():
        raise ValueError("Confirmation target identifiers must be unique across inputs")
    return combined


def _write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with Path(path).open("w", encoding="ascii") as handle:
        handle.writelines(
            f">{row.sequence_id}\n{row.sequence}\n"
            for row in frame.itertuples(index=False)
        )


def _mmseqs_confirmation_search(
    confirmation: pd.DataFrame,
    development: pd.DataFrame,
    *,
    binary: str,
    threads: int,
) -> tuple[pd.DataFrame, str]:
    executable = shutil.which(binary)
    if executable is None:
        raise RuntimeError("MMseqs2 is required for the confirmation sequence audit")
    version = subprocess.run(
        [executable, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    queries = confirmation.reset_index(drop=True).copy()
    queries["sequence_id"] = [f"q{index:05d}" for index in range(len(queries))]
    targets = pd.concat(
        [
            development.assign(source="development"),
            confirmation.assign(source="confirmation"),
        ],
        ignore_index=True,
    )
    targets["sequence_id"] = [f"t{index:05d}" for index in range(len(targets))]
    with tempfile.TemporaryDirectory(prefix="variantshift-confirmation-mmseqs-") as temporary:
        root = Path(temporary)
        query_fasta = root / "confirmation.fasta"
        target_fasta = root / "development-and-confirmation.fasta"
        output = root / "alignments.tsv"
        _write_fasta(queries, query_fasta)
        _write_fasta(targets, target_fasta)
        subprocess.run(
            [
                executable,
                "easy-search",
                str(query_fasta),
                str(target_fasta),
                str(output),
                str(root / "work"),
                "--exhaustive-search",
                "1",
                "--min-seq-id",
                "0.15",
                "-c",
                "0.5",
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
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        alignments = pd.read_csv(output, sep="\t", names=ALIGNMENT_COLUMNS)
    query_lookup = queries[
        ["sequence_id", "panel_id", "target_id", "protein_id", "sequence_sha256"]
    ].add_prefix("query_")
    target_lookup = targets[
        [
            "sequence_id",
            "panel_id",
            "target_id",
            "protein_id",
            "sequence_sha256",
            "source",
        ]
    ].add_prefix("target_")
    alignments = alignments.merge(
        query_lookup,
        on="query_sequence_id",
        how="left",
        validate="many_to_one",
    ).merge(
        target_lookup,
        on="target_sequence_id",
        how="left",
        validate="many_to_one",
    )
    if alignments[["query_target_id", "target_target_id"]].isna().any().any():
        raise RuntimeError("MMseqs2 returned identifiers outside the frozen target tables")
    alignments["qualifies_family_edge"] = (
        alignments["sequence_identity"].ge(0.30)
        & alignments["query_coverage"].ge(0.80)
        & alignments["target_coverage"].ge(0.80)
    )
    return alignments, version


def _confirmation_family_ids(
    confirmation: pd.DataFrame, alignments: pd.DataFrame
) -> dict[tuple[str, str], str]:
    keys = [
        f"{row.panel_id}::{row.target_id}" for row in confirmation.itertuples(index=False)
    ]
    groups = _DisjointSet(keys)
    qualifying = alignments.loc[
        alignments["qualifies_family_edge"]
        & alignments["target_source"].eq("confirmation")
    ]
    for row in qualifying.itertuples(index=False):
        groups.union(
            f"{row.query_panel_id}::{row.query_target_id}",
            f"{row.target_panel_id}::{row.target_target_id}",
        )
    members: dict[str, list[str]] = {}
    for key in keys:
        members.setdefault(groups.find(key), []).append(key)
    identifiers = {}
    for values in members.values():
        digest = hashlib.sha256("|".join(sorted(values)).encode("utf-8")).hexdigest()[:12]
        for value in values:
            panel_id, target_id = value.split("::", maxsplit=1)
            identifiers[(panel_id, target_id)] = f"confirmation-family-{digest}"
    return identifiers


def _optional_overlap(
    audit: pd.DataFrame,
    confirmation_annotations_path: Path | None,
    development_annotations_path: Path | None,
    *,
    annotation_column: str,
    prefix: str,
) -> pd.DataFrame:
    output = audit.copy()
    status_column = f"{prefix}_status"
    unseen_column = f"{prefix}_unseen"
    overlap_column = f"{prefix}_overlap_ids"
    if confirmation_annotations_path is None or development_annotations_path is None:
        output[status_column] = "undocumented"
        output[unseen_column] = pd.Series(pd.NA, index=output.index, dtype="boolean")
        output[overlap_column] = ""
        return output
    confirmation = pd.read_csv(confirmation_annotations_path)
    development = pd.read_csv(development_annotations_path)
    required_confirmation = {"panel_id", "target_id", annotation_column}
    required_development = {annotation_column}
    missing = required_confirmation.difference(confirmation.columns) | required_development.difference(
        development.columns
    )
    if missing:
        raise ValueError(f"{prefix} annotations are missing columns: {sorted(missing)}")
    development_ids = set(development[annotation_column].dropna().astype(str))
    grouped = (
        confirmation.dropna(subset=[annotation_column])
        .groupby(["panel_id", "target_id"])[annotation_column]
        .agg(lambda values: sorted(set(map(str, values))))
    )
    statuses = []
    unseen = []
    overlaps = []
    for row in output.itertuples(index=False):
        values = grouped.get((str(row.panel_id), str(row.target_id)), [])
        shared = sorted(set(values).intersection(development_ids))
        statuses.append("audited" if values else "undocumented")
        unseen.append(not shared if values else pd.NA)
        overlaps.append(";".join(shared))
    output[status_column] = statuses
    output[unseen_column] = pd.array(unseen, dtype="boolean")
    output[overlap_column] = overlaps
    return output


def _model_exposure(
    audit: pd.DataFrame,
    model_config_path: Path,
) -> pd.DataFrame:
    rows = []
    for specification in load_model_specifications(model_config_path):
        declared = specification.exposure_status.lower()
        for target in audit.itertuples(index=False):
            if declared in {"clean", "possible", "known"}:
                category = declared
                reason = "model configuration declaration"
            elif not bool(target.exact_sequence_unseen):
                category = "possible"
                reason = "exact sequence occurs in the development benchmark"
            else:
                category = "undocumented"
                reason = "training data exposure is not documented"
            rows.append(
                {
                    "panel_id": target.panel_id,
                    "target_id": target.target_id,
                    "model_id": specification.model_id,
                    "training_cutoff": specification.training_cutoff,
                    "exposure_category": category,
                    "exposure_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def audit_confirmation_overlap(
    reference_path: Path,
    eligibility_path: Path,
    confirmation_target_paths: list[Path],
    model_config_path: Path,
    output_dir: Path,
    *,
    confirmation_pfam_path: Path | None = None,
    development_pfam_path: Path | None = None,
    confirmation_structure_path: Path | None = None,
    development_structure_path: Path | None = None,
    mmseqs_binary: str = "mmseqs",
    threads: int = 8,
) -> dict[str, Path]:
    """Audit all novelty strata without reading a measurement or effect column."""
    confirmation = _combined_confirmation_targets(confirmation_target_paths)
    development = _development_targets(reference_path, eligibility_path)
    alignments, mmseqs_version = _mmseqs_confirmation_search(
        confirmation,
        development,
        binary=mmseqs_binary,
        threads=threads,
    )
    family_ids = _confirmation_family_ids(confirmation, alignments)
    exact_development = set(development["sequence_sha256"].astype(str))
    development_edges = alignments.loc[
        alignments["target_source"].eq("development")
        & alignments["qualifies_family_edge"]
    ]
    family_overlap = set(
        zip(
            development_edges["query_panel_id"].astype(str),
            development_edges["query_target_id"].astype(str),
            strict=True,
        )
    )
    audit = confirmation.loc[
        :, ["panel_id", "target_id", "protein_id", "sequence_sha256", "sequence_length"]
    ].copy()
    audit["family_id"] = [
        family_ids[(str(row.panel_id), str(row.target_id))]
        for row in audit.itertuples(index=False)
    ]
    audit["exact_sequence_unseen"] = ~audit["sequence_sha256"].astype(str).isin(
        exact_development
    )
    audit["mmseqs_family_unseen"] = [
        (str(row.panel_id), str(row.target_id)) not in family_overlap
        for row in audit.itertuples(index=False)
    ]
    audit = _optional_overlap(
        audit,
        confirmation_structure_path,
        development_structure_path,
        annotation_column="structure_family_id",
        prefix="foldseek_structure_family",
    )
    audit = _optional_overlap(
        audit,
        confirmation_pfam_path,
        development_pfam_path,
        annotation_column="clan_accession",
        prefix="pfam_clan",
    )
    exposure = _model_exposure(audit, model_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audit": output_dir / "confirmation-overlap-audit.csv",
        "alignments": output_dir / "confirmation-sequence-alignments.csv.gz",
        "exposure": output_dir / "model-target-exposure.csv",
        "summary": output_dir / "confirmation-overlap-summary.json",
    }
    write_table(audit, outputs["audit"])
    write_table(alignments, outputs["alignments"])
    write_table(exposure, outputs["exposure"])
    summary = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "mmseqs_version": mmseqs_version,
        "sequence_family_rule": "identity >= 0.30 and bidirectional coverage >= 0.80",
        "confirmation_targets": len(audit),
        "confirmation_families": int(audit["family_id"].nunique()),
        "exact_sequence_unseen": int(audit["exact_sequence_unseen"].sum()),
        "mmseqs_family_unseen": int(audit["mmseqs_family_unseen"].sum()),
        "foldseek_status": {
            str(key): int(value)
            for key, value in audit["foldseek_structure_family_status"].value_counts().items()
        },
        "pfam_status": {
            str(key): int(value)
            for key, value in audit["pfam_clan_status"].value_counts().items()
        },
        "exposure_categories": {
            str(key): int(value)
            for key, value in exposure["exposure_category"].value_counts().items()
        },
        "input_sha256": {
            "reference": sha256_file(reference_path),
            "eligibility": sha256_file(eligibility_path),
            "model_config": sha256_file(model_config_path),
            **{
                str(path): sha256_file(path) for path in confirmation_target_paths
            },
        },
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
