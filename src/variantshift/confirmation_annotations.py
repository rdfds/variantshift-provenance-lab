"""Outcome-blind Pfam-clan and Foldseek annotations for confirmation targets."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .curated_families import _pfam_clan, _protein_pfams
from .family_clusters import _DisjointSet
from .outcome_lock import assert_target_only
from .provenance import sha256_file
from .schemas import validate_targets, write_table
from .structure_clusters import _run_foldseek, eligible_structure_inputs


def curated_family_id(pfam_accession: str, clan_accession: object) -> str:
    """Return the preregisterable clan grouping, falling back to the Pfam family."""
    clan = "" if pd.isna(clan_accession) else str(clan_accession).strip()
    return f"clan:{clan}" if clan else f"pfam:{pfam_accession}"


def _load_confirmation_targets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = validate_targets(pd.read_csv(path)).copy()
        assert_target_only(frame)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["panel_id", "target_id"]).any():
        raise ValueError("Confirmation target identifiers are not unique")
    return combined


def _load_structure_resolution(
    audit_paths: list[Path], targets: pd.DataFrame
) -> pd.DataFrame:
    panel_by_target = dict(
        zip(targets["target_id"].astype(str), targets["panel_id"].astype(str), strict=True)
    )
    frames = []
    for path in audit_paths:
        frame = pd.read_csv(path)
        if "panel_id" not in frame:
            frame["panel_id"] = frame["target_id"].astype(str).map(panel_by_target)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined[["panel_id", "target_id"]].isna().any().any():
        raise ValueError("A structure audit target cannot be mapped to a confirmation panel")
    if combined.duplicated(["panel_id", "target_id"]).any():
        raise ValueError("Structure audits contain duplicate confirmation targets")
    return combined


def freeze_confirmation_pfam_annotations(
    target_paths: list[Path],
    structure_audit_paths: list[Path],
    development_domain_path: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    workers: int = 12,
) -> dict[str, Path]:
    """Freeze Pfam/clan annotations without opening any confirmation outcomes."""
    targets = _load_confirmation_targets(target_paths)
    resolutions = _load_structure_resolution(structure_audit_paths, targets)
    resolution_by_target = resolutions.set_index(["panel_id", "target_id"])
    direct_pfams: dict[tuple[str, str], str] = {}
    protein_requests: dict[str, str] = {}
    target_accessions: dict[tuple[str, str], str] = {}
    for target in targets.itertuples(index=False):
        key = (str(target.panel_id), str(target.target_id))
        pfam = str(getattr(target, "pfam_id", "") or "").split(".", maxsplit=1)[0]
        if pfam.startswith("PF"):
            direct_pfams[key] = pfam
            continue
        if key not in resolution_by_target.index:
            continue
        resolution = resolution_by_target.loc[key]
        accession = str(resolution.get("resolved_uniprot_accession") or "").strip()
        if not accession or accession.lower() == "nan":
            continue
        entry_name = str(resolution.get("resolved_uniprot_entry_name") or accession)
        target_accessions[key] = accession
        protein_requests.setdefault(accession, entry_name)

    protein_domains: dict[str, list[dict[str, object]]] = {}
    protein_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_protein_pfams, accession, entry_name, cache_dir): accession
            for accession, entry_name in sorted(protein_requests.items())
        }
        for future in as_completed(futures):
            accession = futures[future]
            try:
                protein_domains[accession] = future.result()
            except Exception as exception:  # noqa: BLE001 - retain annotation failures
                protein_errors[accession] = f"{type(exception).__name__}: {exception}"

    all_pfams = set(direct_pfams.values())
    for domains in protein_domains.values():
        all_pfams.update(str(domain["pfam_accession"]) for domain in domains)
    clans: dict[str, dict[str, object]] = {}
    clan_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_pfam_clan, pfam, cache_dir): pfam for pfam in sorted(all_pfams)
        }
        for future in as_completed(futures):
            pfam = futures[future]
            try:
                clans[pfam] = future.result()
            except Exception as exception:  # noqa: BLE001 - retain annotation failures
                clan_errors[pfam] = f"{type(exception).__name__}: {exception}"

    annotation_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for target in targets.sort_values(["panel_id", "target_id"]).itertuples(index=False):
        key = (str(target.panel_id), str(target.target_id))
        annotations: list[dict[str, object]] = []
        if key in direct_pfams:
            annotations = [{"pfam_accession": direct_pfams[key], "annotation_source": "panel"}]
        elif key in target_accessions:
            accession = target_accessions[key]
            annotations = [
                {**domain, "annotation_source": "InterPro/UniProt"}
                for domain in protein_domains.get(accession, [])
            ]
        errors = []
        if key in target_accessions and target_accessions[key] in protein_errors:
            errors.append(protein_errors[target_accessions[key]])
        for annotation in annotations:
            pfam = str(annotation["pfam_accession"])
            clan = clans.get(pfam, {})
            if pfam in clan_errors:
                errors.append(clan_errors[pfam])
            annotation_rows.append(
                {
                    "panel_id": key[0],
                    "target_id": key[1],
                    "protein_id": str(target.protein_id),
                    "pfam_accession": pfam,
                    "pfam_name": annotation.get("pfam_name", ""),
                    "clan_accession": clan.get("clan_accession"),
                    "clan_name": clan.get("clan_name"),
                    "pfam_clan_family_id": curated_family_id(
                        pfam, clan.get("clan_accession")
                    ),
                    "domain_start": annotation.get("domain_start"),
                    "domain_end": annotation.get("domain_end"),
                    "annotation_source": annotation["annotation_source"],
                }
            )
        if errors:
            status = "annotation_error"
        elif annotations:
            status = "audited_with_annotation"
        elif key in target_accessions:
            status = "audited_no_pfam"
        else:
            status = "undocumented_no_exact_uniprot"
        audit_rows.append(
            {
                "panel_id": key[0],
                "target_id": key[1],
                "annotation_status": status,
                "pfam_count": len(annotations),
                "resolved_uniprot_accession": target_accessions.get(key, ""),
                "error": "; ".join(sorted(set(errors))),
                "outcomes_accessed": False,
            }
        )

    annotations = pd.DataFrame(annotation_rows)
    audit = pd.DataFrame(audit_rows)
    development = pd.read_csv(development_domain_path)
    development = development.loc[development["qualifies_curated_domain"].astype(bool)].copy()
    development["pfam_clan_family_id"] = [
        curated_family_id(str(row.pfam_accession), row.clan_accession)
        for row in development.itertuples(index=False)
    ]
    development = development[
        [
            "assay_id",
            "uniprot_id",
            "pfam_accession",
            "clan_accession",
            "pfam_clan_family_id",
        ]
    ].drop_duplicates()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "confirmation": output_dir / "confirmation-pfam-annotations.csv",
        "development": output_dir / "development-pfam-annotations.csv",
        "audit": output_dir / "pfam-annotation-audit.csv",
        "manifest": output_dir / "pfam-annotation-manifest.json",
    }
    write_table(annotations, outputs["confirmation"])
    write_table(development, outputs["development"])
    write_table(audit, outputs["audit"])
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "grouping_rule": "Pfam clan when available; otherwise the individual Pfam family",
        "confirmation_target_count": len(targets),
        "targets_with_annotations": int(audit["pfam_count"].gt(0).sum()),
        "targets_without_exact_uniprot_mapping": int(
            audit["annotation_status"].eq("undocumented_no_exact_uniprot").sum()
        ),
        "annotation_error_count": int(audit["annotation_status"].eq("annotation_error").sum()),
        "inputs": {
            "targets": {str(path): sha256_file(path) for path in target_paths},
            "structure_audits": {
                str(path): sha256_file(path) for path in structure_audit_paths
            },
            "development_domains": sha256_file(development_domain_path),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def _reciprocal_foldseek_edges(
    directed: pd.DataFrame,
    *,
    minimum_tm_score: float,
    minimum_coverage: float,
    minimum_homology_probability: float,
) -> pd.DataFrame:
    nonself = directed.loc[
        directed["query_uniprot_id"].astype(str).ne(
            directed["target_uniprot_id"].astype(str)
        )
    ].copy()
    nonself["structure_a"] = nonself[
        ["query_uniprot_id", "target_uniprot_id"]
    ].min(axis=1)
    nonself["structure_b"] = nonself[
        ["query_uniprot_id", "target_uniprot_id"]
    ].max(axis=1)
    nonself["minimum_tm_score"] = nonself[["query_tm_score", "target_tm_score"]].min(
        axis=1
    )
    nonself["minimum_coverage"] = nonself[["query_coverage", "target_coverage"]].min(
        axis=1
    )
    edges = (
        nonself.groupby(["structure_a", "structure_b"], as_index=False)
        .agg(
            direction_count=("query_uniprot_id", "nunique"),
            reciprocal_minimum_tm_score=("minimum_tm_score", "min"),
            reciprocal_minimum_coverage=("minimum_coverage", "min"),
            reciprocal_minimum_homology_probability=("homology_probability", "min"),
            maximum_sequence_identity=("sequence_identity", "max"),
            minimum_e_value=("e_value", "min"),
        )
        .sort_values(["structure_a", "structure_b"])
        .reset_index(drop=True)
    )
    edges["qualifies_structure_edge"] = (
        edges["direction_count"].eq(2)
        & edges["reciprocal_minimum_tm_score"].ge(minimum_tm_score)
        & edges["reciprocal_minimum_coverage"].ge(minimum_coverage)
        & edges["reciprocal_minimum_homology_probability"].ge(
            minimum_homology_probability
        )
    )
    return edges


def freeze_confirmation_structure_families(
    structure_archive_path: Path,
    reference_path: Path,
    eligibility_path: Path,
    confirmation_target_paths: list[Path],
    confirmation_structure_audit_paths: list[Path],
    output_dir: Path,
    *,
    foldseek_binary: str = "foldseek",
    threads: int = 8,
    minimum_tm_score: float = 0.50,
    minimum_coverage: float = 0.80,
    minimum_homology_probability: float = 0.95,
) -> dict[str, Path]:
    """Freeze joint development/confirmation Foldseek-family assignments."""
    targets = _load_confirmation_targets(confirmation_target_paths)
    confirmation_audit = _load_structure_resolution(
        confirmation_structure_audit_paths, targets
    )
    eligibility = pd.read_csv(eligibility_path)
    development_inputs, development_payloads = eligible_structure_inputs(
        structure_archive_path, reference_path, eligibility
    )

    records: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for index, row in enumerate(development_inputs.sort_values("uniprot_id").itertuples()):
        structure_id = f"d{index:05d}"
        payload = development_payloads[str(row.uniprot_id)]
        payloads[structure_id] = payload
        records.append(
            {
                "structure_id": structure_id,
                "source": "development",
                "panel_id": "proteingym-v1.3-development",
                "target_id": "",
                "uniprot_id": str(row.uniprot_id),
                "structure_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    usable = confirmation_audit.loc[confirmation_audit["status"].eq("usable")].sort_values(
        ["panel_id", "target_id"]
    )
    for index, row in enumerate(usable.itertuples(index=False)):
        structure_id = f"c{index:05d}"
        path = Path(str(row.structure_path))
        payload = path.read_bytes()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != str(row.structure_sha256):
            raise ValueError(f"Structure hash drift for {row.panel_id}/{row.target_id}")
        payloads[structure_id] = payload
        records.append(
            {
                "structure_id": structure_id,
                "source": "confirmation",
                "panel_id": str(row.panel_id),
                "target_id": str(row.target_id),
                "uniprot_id": "",
                "structure_sha256": observed_sha256,
            }
        )
    structure_inputs = pd.DataFrame(records)
    foldseek_inputs = structure_inputs[["structure_id"]].rename(
        columns={"structure_id": "uniprot_id"}
    )
    directed, foldseek_version = _run_foldseek(
        foldseek_inputs, payloads, binary=foldseek_binary, threads=threads
    )
    edges = _reciprocal_foldseek_edges(
        directed,
        minimum_tm_score=minimum_tm_score,
        minimum_coverage=minimum_coverage,
        minimum_homology_probability=minimum_homology_probability,
    )
    identifiers = structure_inputs["structure_id"].astype(str).tolist()
    groups = _DisjointSet(identifiers)
    for row in edges.loc[edges["qualifies_structure_edge"]].itertuples(index=False):
        groups.union(str(row.structure_a), str(row.structure_b))
    members: dict[str, list[str]] = {}
    for identifier in identifiers:
        members.setdefault(groups.find(identifier), []).append(identifier)
    family_by_structure = {}
    for component in members.values():
        family_id = "foldseek-family-" + hashlib.sha256(
            "|".join(sorted(component)).encode("ascii")
        ).hexdigest()[:12]
        family_by_structure.update(dict.fromkeys(component, family_id))
    structure_inputs["structure_family_id"] = structure_inputs["structure_id"].map(
        family_by_structure
    )
    development = structure_inputs.loc[
        structure_inputs["source"].eq("development"),
        ["uniprot_id", "structure_family_id", "structure_sha256"],
    ]
    confirmation = structure_inputs.loc[
        structure_inputs["source"].eq("confirmation"),
        ["panel_id", "target_id", "structure_family_id", "structure_sha256"],
    ]
    assignment_lookup = confirmation.set_index(["panel_id", "target_id"])[
        "structure_family_id"
    ]
    audit_rows = []
    for target in targets.sort_values(["panel_id", "target_id"]).itertuples(index=False):
        key = (str(target.panel_id), str(target.target_id))
        family = assignment_lookup.get(key)
        structure_row = confirmation_audit.loc[
            confirmation_audit["panel_id"].astype(str).eq(key[0])
            & confirmation_audit["target_id"].astype(str).eq(key[1])
        ]
        exclusion = ""
        if family is None and not structure_row.empty:
            exclusion = str(structure_row.iloc[0].get("exclusion_reason") or "")
        audit_rows.append(
            {
                "panel_id": key[0],
                "target_id": key[1],
                "status": "audited" if family is not None else "undocumented_no_structure",
                "structure_family_id": family or "",
                "exclusion_reason": exclusion,
                "outcomes_accessed": False,
            }
        )
    audit = pd.DataFrame(audit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "confirmation": output_dir / "confirmation-structure-families.csv",
        "development": output_dir / "development-structure-families.csv",
        "inputs": output_dir / "structure-inputs.csv",
        "directed": output_dir / "foldseek-directed-alignments.csv.gz",
        "edges": output_dir / "foldseek-reciprocal-edges.csv.gz",
        "audit": output_dir / "structure-family-audit.csv",
        "manifest": output_dir / "structure-family-manifest.json",
    }
    write_table(confirmation, outputs["confirmation"])
    write_table(development, outputs["development"])
    write_table(structure_inputs, outputs["inputs"])
    write_table(directed, outputs["directed"])
    write_table(edges, outputs["edges"])
    write_table(audit, outputs["audit"])
    development_families = set(development["structure_family_id"].astype(str))
    shared = confirmation["structure_family_id"].astype(str).isin(development_families)
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "foldseek_version": foldseek_version,
        "thresholds": {
            "minimum_reciprocal_tm_score": minimum_tm_score,
            "minimum_bidirectional_coverage": minimum_coverage,
            "minimum_reciprocal_homology_probability": minimum_homology_probability,
        },
        "development_structure_count": len(development),
        "confirmation_target_count": len(targets),
        "confirmation_structure_count": len(confirmation),
        "confirmation_structure_unseen_count": int((~shared).sum()),
        "inputs": {
            "structure_archive": sha256_file(structure_archive_path),
            "reference": sha256_file(reference_path),
            "eligibility": sha256_file(eligibility_path),
            "confirmation_targets": {
                str(path): sha256_file(path) for path in confirmation_target_paths
            },
            "confirmation_structure_audits": {
                str(path): sha256_file(path)
                for path in confirmation_structure_audit_paths
            },
        },
        "artifacts": {
            path.name: sha256_file(path)
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
