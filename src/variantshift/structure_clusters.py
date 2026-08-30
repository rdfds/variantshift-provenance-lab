"""Structure-aware family clustering over official ProteinGym AlphaFold models."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

from .family_clusters import _DisjointSet
from .proteingym import read_reference_index

FOLDSEEK_COLUMNS = (
    "query_uniprot_id",
    "target_uniprot_id",
    "query_tm_score",
    "target_tm_score",
    "alignment_tm_score",
    "query_coverage",
    "target_coverage",
    "lddt",
    "homology_probability",
    "e_value",
    "bit_score",
    "alignment_length",
    "sequence_identity",
)


@dataclass(frozen=True)
class StructureClusteringResult:
    assignments: pd.DataFrame
    structure_inputs: pd.DataFrame
    alignments: pd.DataFrame
    sensitivity: pd.DataFrame
    audit: pd.DataFrame


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_structure_inputs(
    structure_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bytes]]:
    """Resolve one official structure for every eligible UniProt unit."""
    reference = read_reference_index(reference_path)
    required = {"pdb_file", "pdb_range"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(
            f"ProteinGym reference is missing structure fields: {', '.join(sorted(missing))}"
        )
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    selected = reference.loc[
        reference["DMS_id"].astype(str).isin(eligible_ids),
        ["DMS_id", "UniProt_ID", "pdb_file", "pdb_range"],
    ].copy()
    if len(selected) != len(eligible_ids):
        raise ValueError("Eligible assays are missing from the ProteinGym reference")
    if selected[["pdb_file", "pdb_range"]].isna().any().any():
        raise ValueError("Every eligible assay requires an official structure and range")
    selected = selected.rename(
        columns={
            "DMS_id": "assay_id",
            "UniProt_ID": "uniprot_id",
            "pdb_file": "structure_file",
            "pdb_range": "structure_range",
        }
    )
    conflicts = (
        selected.groupby("uniprot_id")[["structure_file", "structure_range"]]
        .nunique()
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        identifiers = ", ".join(conflicts.index[conflicts].astype(str))
        raise ValueError(f"UniProt IDs map to conflicting structures: {identifiers}")
    proteins = (
        selected.groupby("uniprot_id", as_index=False)
        .agg(
            structure_file=("structure_file", "first"),
            structure_range=("structure_range", "first"),
            assay_count=("assay_id", "nunique"),
        )
        .sort_values("uniprot_id")
        .reset_index(drop=True)
    )

    payloads: dict[str, bytes] = {}
    with ZipFile(structure_archive_path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.name.lower().endswith(".pdb"):
                if path.name in members:
                    raise ValueError(f"Structure archive contains duplicate filename: {path.name}")
                members[path.name] = name
        missing_files = sorted(set(proteins["structure_file"]) - set(members))
        if missing_files:
            raise ValueError(
                f"Official structures are absent from the archive: {', '.join(missing_files)}"
            )
        for row in proteins.itertuples(index=False):
            data = archive.read(members[str(row.structure_file)])
            if not data.startswith((b"ATOM", b"HEADER", b"MODEL", b"REMARK")):
                raise ValueError(f"Structure is not recognizable PDB data: {row.structure_file}")
            payloads[str(row.uniprot_id)] = data

    proteins["structure_bytes"] = proteins["uniprot_id"].map(
        {protein: len(data) for protein, data in payloads.items()}
    )
    proteins["structure_sha256"] = proteins["uniprot_id"].map(
        {protein: _sha256_bytes(data) for protein, data in payloads.items()}
    )
    return proteins, payloads


def _run_foldseek(
    structure_inputs: pd.DataFrame,
    payloads: dict[str, bytes],
    *,
    binary: str,
    threads: int,
) -> tuple[pd.DataFrame, str]:
    executable = shutil.which(binary)
    if executable is None:
        raise RuntimeError(
            "Foldseek is required for structure clustering; supply a 'foldseek' executable"
        )
    version = subprocess.run(
        [executable, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="variantshift-foldseek-") as temporary:
        root = Path(temporary)
        structures = root / "structures"
        structures.mkdir()
        for row in structure_inputs.itertuples(index=False):
            (structures / f"{row.uniprot_id}.pdb").write_bytes(payloads[str(row.uniprot_id)])
        output = root / "all-versus-all.tsv"
        command = [
            executable,
            "easy-search",
            str(structures),
            str(structures),
            str(output),
            str(root / "work"),
            "--exhaustive-search",
            "1",
            "--alignment-type",
            "1",
            "--exact-tmscore",
            "1",
            "-e",
            "100000",
            "--max-seqs",
            "10000",
            "--format-output",
            (
                "query,target,qtmscore,ttmscore,alntmscore,qcov,tcov,lddt,"
                "prob,evalue,bits,alnlen,fident"
            ),
            "--threads",
            str(threads),
            "-v",
            "1",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        directed = pd.read_csv(output, sep="\t", names=FOLDSEEK_COLUMNS)
    return directed, version


def reciprocal_structure_pairs(
    directed: pd.DataFrame,
    proteins: list[str],
) -> pd.DataFrame:
    """Collapse the exact directed score matrix to conservative reciprocal pairs."""
    required = set(FOLDSEEK_COLUMNS)
    missing = required.difference(directed.columns)
    if missing:
        raise ValueError(f"Foldseek results are missing: {', '.join(sorted(missing))}")
    proteins = sorted(map(str, proteins))
    expected = len(proteins) ** 2
    if len(directed) != expected:
        raise ValueError(
            f"Foldseek search is incomplete: expected {expected} rows, found {len(directed)}"
        )
    observed_queries = set(directed["query_uniprot_id"].astype(str))
    observed_targets = set(directed["target_uniprot_id"].astype(str))
    if observed_queries != set(proteins) or observed_targets != set(proteins):
        raise ValueError("Foldseek result identifiers do not match the structure cohort")
    if directed.duplicated(["query_uniprot_id", "target_uniprot_id"]).any():
        raise ValueError("Foldseek returned duplicate directed structure pairs")
    self_hits = directed.loc[directed["query_uniprot_id"].eq(directed["target_uniprot_id"])]
    if len(self_hits) != len(proteins):
        raise ValueError("Foldseek result is missing self alignments")

    nonself = directed.loc[directed["query_uniprot_id"].ne(directed["target_uniprot_id"])].copy()
    nonself["protein_a"] = nonself[["query_uniprot_id", "target_uniprot_id"]].min(axis=1)
    nonself["protein_b"] = nonself[["query_uniprot_id", "target_uniprot_id"]].max(axis=1)
    nonself["minimum_tm_score"] = nonself[["query_tm_score", "target_tm_score"]].min(axis=1)
    nonself["minimum_coverage"] = nonself[["query_coverage", "target_coverage"]].min(axis=1)
    pairs = (
        nonself.groupby(["protein_a", "protein_b"], as_index=False)
        .agg(
            direction_count=("query_uniprot_id", "size"),
            reciprocal_minimum_tm_score=("minimum_tm_score", "min"),
            reciprocal_minimum_coverage=("minimum_coverage", "min"),
            reciprocal_minimum_lddt=("lddt", "min"),
            reciprocal_minimum_homology_probability=("homology_probability", "min"),
            maximum_sequence_identity=("sequence_identity", "max"),
            maximum_alignment_length=("alignment_length", "max"),
            minimum_e_value=("e_value", "min"),
            minimum_bit_score=("bit_score", "min"),
        )
        .sort_values(["protein_a", "protein_b"])
        .reset_index(drop=True)
    )
    if pairs["direction_count"].ne(2).any():
        raise ValueError("Every non-self structure pair must have two directed alignments")
    return pairs


def _combine_families(
    sequence_assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    minimum_tm_score: float,
    minimum_coverage: float,
    minimum_homology_probability: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"uniprot_id", "family_id"}
    missing = required.difference(sequence_assignments.columns)
    if missing:
        raise ValueError(f"Sequence-family assignments are missing: {', '.join(sorted(missing))}")
    sequence = sequence_assignments.copy()
    if sequence["uniprot_id"].duplicated().any():
        raise ValueError("Each UniProt ID must have exactly one sequence-family assignment")
    proteins = sorted(sequence["uniprot_id"].astype(str))
    if set(pairs["protein_a"]).union(pairs["protein_b"]) != set(proteins):
        raise ValueError("Structure pairs and sequence assignments cover different proteins")
    disjoint = _DisjointSet(proteins)
    for _, group in sequence.groupby("family_id"):
        members = sorted(group["uniprot_id"].astype(str))
        for member in members[1:]:
            disjoint.union(members[0], member)
    audited = pairs.copy()
    audited["qualifies_structure_edge"] = (
        audited["reciprocal_minimum_tm_score"].ge(minimum_tm_score)
        & audited["reciprocal_minimum_coverage"].ge(minimum_coverage)
        & audited["reciprocal_minimum_homology_probability"].ge(minimum_homology_probability)
    )
    for row in audited.loc[audited["qualifies_structure_edge"]].itertuples(index=False):
        disjoint.union(str(row.protein_a), str(row.protein_b))

    components: dict[str, list[str]] = {}
    for protein in proteins:
        components.setdefault(disjoint.find(protein), []).append(protein)
    family_by_protein: dict[str, str] = {}
    members_by_family: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        ordered = tuple(sorted(members))
        digest = hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()[:12]
        family_id = f"ssf_{digest}"
        members_by_family[family_id] = ordered
        family_by_protein.update({protein: family_id for protein in ordered})

    assignments = sequence.rename(columns={"family_id": "sequence_family_id"})
    assignments["family_id"] = assignments["uniprot_id"].map(family_by_protein)
    assignments["family_size"] = assignments["family_id"].map(
        {family: len(members) for family, members in members_by_family.items()}
    )
    assignments["family_members"] = assignments["family_id"].map(
        {family: ";".join(members) for family, members in members_by_family.items()}
    )
    sequence_family_size = sequence.set_index("uniprot_id")["family_size"]
    assignments["structure_added_relationship"] = assignments.apply(
        lambda row: row["family_size"] > int(sequence_family_size.loc[str(row["uniprot_id"])]),
        axis=1,
    )
    preferred = [
        "uniprot_id",
        "family_id",
        "family_size",
        "family_members",
        "sequence_family_id",
        "structure_added_relationship",
    ]
    assignments = (
        assignments[
            preferred + [column for column in assignments.columns if column not in preferred]
        ]
        .sort_values(["family_id", "uniprot_id"])
        .reset_index(drop=True)
    )

    audited["protein_a_family_id"] = audited["protein_a"].map(family_by_protein)
    audited["protein_b_family_id"] = audited["protein_b"].map(family_by_protein)
    cross = audited.loc[
        audited["qualifies_structure_edge"]
        & audited["protein_a_family_id"].ne(audited["protein_b_family_id"])
    ]
    if not cross.empty:
        raise RuntimeError("A qualifying structure edge crosses combined-family clusters")
    return assignments, audited


def _summary(
    assignments: pd.DataFrame,
    audited: pd.DataFrame,
    *,
    minimum_tm_score: float,
    minimum_coverage: float,
    minimum_homology_probability: float,
) -> dict[str, int | float]:
    sizes = assignments.groupby("family_id")["uniprot_id"].nunique()
    qualifying = audited.loc[audited["qualifies_structure_edge"]]
    return {
        "minimum_tm_score": minimum_tm_score,
        "minimum_bidirectional_coverage": minimum_coverage,
        "minimum_reciprocal_homology_probability": minimum_homology_probability,
        "n_families": len(sizes),
        "singleton_families": int(sizes.eq(1).sum()),
        "multi_protein_families": int(sizes.gt(1).sum()),
        "proteins_in_multi_protein_families": int(sizes.loc[sizes.gt(1)].sum()),
        "largest_family_size": int(sizes.max()),
        "qualifying_structure_pairs": len(qualifying),
    }


def build_sequence_structure_family_clusters(
    structure_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    sequence_assignments: pd.DataFrame,
    *,
    minimum_tm_score: float = 0.50,
    minimum_coverage: float = 0.80,
    minimum_homology_probability: float = 0.95,
    binary: str = "foldseek",
    threads: int = 8,
) -> StructureClusteringResult:
    """Union sequence homology with conservative reciprocal structure matches."""
    for name, value in (
        ("minimum_tm_score", minimum_tm_score),
        ("minimum_coverage", minimum_coverage),
        ("minimum_homology_probability", minimum_homology_probability),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if threads < 1:
        raise ValueError("Foldseek thread count must be positive")
    structure_inputs, payloads = eligible_structure_inputs(
        structure_archive_path, reference_path, eligibility
    )
    directed, version = _run_foldseek(structure_inputs, payloads, binary=binary, threads=threads)
    pairs = reciprocal_structure_pairs(
        directed, structure_inputs["uniprot_id"].astype(str).tolist()
    )
    assignments, audited = _combine_families(
        sequence_assignments,
        pairs,
        minimum_tm_score=minimum_tm_score,
        minimum_coverage=minimum_coverage,
        minimum_homology_probability=minimum_homology_probability,
    )
    sensitivity_rows = []
    settings = {
        (0.50, 0.50, minimum_homology_probability),
        (0.50, minimum_coverage, 0.90),
        (minimum_tm_score, minimum_coverage, minimum_homology_probability),
        (0.50, minimum_coverage, 0.99),
        (0.70, minimum_coverage, minimum_homology_probability),
    }
    for tm_score, coverage, probability in sorted(settings):
        current, current_audit = _combine_families(
            sequence_assignments,
            pairs,
            minimum_tm_score=tm_score,
            minimum_coverage=coverage,
            minimum_homology_probability=probability,
        )
        sensitivity_rows.append(
            _summary(
                current,
                current_audit,
                minimum_tm_score=tm_score,
                minimum_coverage=coverage,
                minimum_homology_probability=probability,
            )
        )
    sensitivity = (
        pd.DataFrame(sensitivity_rows)
        .sort_values(
            [
                "minimum_bidirectional_coverage",
                "minimum_tm_score",
                "minimum_reciprocal_homology_probability",
            ]
        )
        .reset_index(drop=True)
    )
    summary = _summary(
        assignments,
        audited,
        minimum_tm_score=minimum_tm_score,
        minimum_coverage=minimum_coverage,
        minimum_homology_probability=minimum_homology_probability,
    )
    sequence_family_count = int(sequence_assignments["family_id"].nunique())
    audit = pd.DataFrame(
        [
            {
                "method": (
                    "sequence-family graph union reciprocal Foldseek structure-homology "
                    "connected components"
                ),
                "structure_source": "official ProteinGym AlphaFold structure archive",
                "foldseek_version": version,
                "alignment_mode": "exhaustive all-versus-all exact TM-align",
                "reciprocal_rule": "both directed alignments must pass every threshold",
                "structure_archive_bytes": Path(structure_archive_path).stat().st_size,
                "structure_archive_sha256": _sha256_path(structure_archive_path),
                "structures": len(structure_inputs),
                "structure_bytes": int(structure_inputs["structure_bytes"].sum()),
                "directed_alignment_rows": len(directed),
                "undirected_structure_pairs": len(pairs),
                "sequence_family_count": sequence_family_count,
                "families_merged_by_structure": sequence_family_count
                - int(assignments["family_id"].nunique()),
                **summary,
                "cross_family_qualifying_structure_pairs": int(
                    (
                        audited["qualifies_structure_edge"]
                        & audited["protein_a_family_id"].ne(audited["protein_b_family_id"])
                    ).sum()
                ),
            }
        ]
    )
    return StructureClusteringResult(assignments, structure_inputs, audited, sensitivity, audit)
