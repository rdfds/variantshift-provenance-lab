"""Auditable sequence-family proxies for cross-protein evaluation."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .proteingym import read_reference_index

MMSEQS_COLUMNS = (
    "query_sequence_id",
    "target_sequence_id",
    "sequence_identity",
    "alignment_length",
    "query_coverage",
    "target_coverage",
    "e_value",
    "bit_score",
)


@dataclass(frozen=True)
class FamilyClusteringResult:
    assignments: pd.DataFrame
    alignments: pd.DataFrame
    sensitivity: pd.DataFrame
    audit: pd.DataFrame


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _eligible_segments(reference_path: Path, eligibility: pd.DataFrame) -> pd.DataFrame:
    reference = read_reference_index(reference_path)
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    selected = reference.loc[
        reference["DMS_id"].astype(str).isin(eligible_ids),
        ["DMS_id", "UniProt_ID", "target_seq", "MSA_start", "MSA_end"],
    ].copy()
    selected = selected.sort_values("DMS_id").reset_index(drop=True)
    if len(selected) != len(eligible_ids):
        raise ValueError("Eligible assays are missing from the ProteinGym reference")
    if selected[["MSA_start", "MSA_end"]].isna().any().any():
        raise ValueError("Every eligible assay requires MSA_start and MSA_end")
    selected["MSA_start"] = selected["MSA_start"].astype(int)
    selected["MSA_end"] = selected["MSA_end"].astype(int)
    lengths = selected["target_seq"].astype(str).str.len()
    if selected["MSA_start"].lt(1).any() or selected["MSA_end"].gt(lengths).any():
        raise ValueError("Assayed-region coordinates exceed the target sequence")
    selected["sequence_id"] = [f"assay{index:03d}" for index in range(len(selected))]
    selected["assayed_sequence"] = [
        sequence[start - 1 : end]
        for sequence, start, end in zip(
            selected["target_seq"].astype(str),
            selected["MSA_start"],
            selected["MSA_end"],
            strict=True,
        )
    ]
    selected["sequence_sha256"] = selected["assayed_sequence"].map(_sequence_digest)
    selected["sequence_length"] = selected["assayed_sequence"].str.len()
    return selected.rename(
        columns={"DMS_id": "assay_id", "UniProt_ID": "uniprot_id"}
    )


def _write_fasta(segments: pd.DataFrame, path: Path) -> None:
    with Path(path).open("w", encoding="ascii") as handle:
        handle.writelines(
            f">{row.sequence_id}\n{row.assayed_sequence}\n"
            for row in segments.itertuples(index=False)
        )


def _run_mmseqs(
    segments: pd.DataFrame,
    *,
    binary: str,
    identity_floor: float,
    coverage_floor: float,
    threads: int,
) -> tuple[pd.DataFrame, str]:
    executable = shutil.which(binary)
    if executable is None:
        raise RuntimeError(
            "MMseqs2 is required for family clustering; install the 'mmseqs2' executable"
        )
    version = subprocess.run(
        [executable, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="variantshift-mmseqs-") as temporary:
        root = Path(temporary)
        fasta = root / "eligible-assayed-sequences.fasta"
        output = root / "alignments.tsv"
        _write_fasta(segments, fasta)
        command = [
            executable,
            "easy-search",
            str(fasta),
            str(fasta),
            str(output),
            str(root / "work"),
            "--exhaustive-search",
            "1",
            "--min-seq-id",
            str(identity_floor),
            "-c",
            str(coverage_floor),
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
        hits = pd.read_csv(output, sep="\t", names=MMSEQS_COLUMNS)
    lookup = segments[
        ["sequence_id", "assay_id", "uniprot_id", "sequence_sha256", "sequence_length"]
    ]
    hits = hits.merge(
        lookup.add_prefix("query_"),
        on="query_sequence_id",
        how="left",
        validate="many_to_one",
    ).merge(
        lookup.add_prefix("target_"),
        on="target_sequence_id",
        how="left",
        validate="many_to_one",
    )
    if hits[["query_uniprot_id", "target_uniprot_id"]].isna().any().any():
        raise RuntimeError("MMseqs2 returned an unknown sequence identifier")
    return hits, version


def cluster_from_alignments(
    segments: pd.DataFrame,
    alignments: pd.DataFrame,
    *,
    identity_threshold: float,
    coverage_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form connected components from every alignment meeting the family rule."""
    proteins = sorted(segments["uniprot_id"].astype(str).unique())
    disjoint = _DisjointSet(proteins)
    qualifying = alignments.loc[
        alignments["sequence_identity"].ge(identity_threshold)
        & alignments["query_coverage"].ge(coverage_threshold)
        & alignments["target_coverage"].ge(coverage_threshold)
    ].copy()
    for row in qualifying.itertuples(index=False):
        disjoint.union(str(row.query_uniprot_id), str(row.target_uniprot_id))

    components: dict[str, list[str]] = {}
    for protein in proteins:
        components.setdefault(disjoint.find(protein), []).append(protein)
    family_by_protein: dict[str, str] = {}
    members_by_family: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        ordered = tuple(sorted(members))
        digest = hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()[:12]
        family_id = f"sf_{digest}"
        members_by_family[family_id] = ordered
        family_by_protein.update({protein: family_id for protein in ordered})

    protein_counts = segments.groupby("uniprot_id", as_index=False).agg(
        assay_count=("assay_id", "nunique"),
        assayed_sequence_count=("sequence_sha256", "nunique"),
    )
    assignments = protein_counts.copy()
    assignments["family_id"] = assignments["uniprot_id"].map(family_by_protein)
    assignments["family_size"] = assignments["family_id"].map(
        {family: len(members) for family, members in members_by_family.items()}
    )
    assignments["family_members"] = assignments["family_id"].map(
        {family: ";".join(members) for family, members in members_by_family.items()}
    )
    assignments = assignments[
        [
            "uniprot_id",
            "family_id",
            "family_size",
            "family_members",
            "assay_count",
            "assayed_sequence_count",
        ]
    ].sort_values(["family_id", "uniprot_id"]).reset_index(drop=True)

    audited = alignments.copy()
    audited["qualifies_family_edge"] = (
        audited["sequence_identity"].ge(identity_threshold)
        & audited["query_coverage"].ge(coverage_threshold)
        & audited["target_coverage"].ge(coverage_threshold)
    )
    audited["query_family_id"] = audited["query_uniprot_id"].map(family_by_protein)
    audited["target_family_id"] = audited["target_uniprot_id"].map(family_by_protein)
    cross_family = audited.loc[
        audited["qualifies_family_edge"]
        & audited["query_family_id"].ne(audited["target_family_id"])
    ]
    if not cross_family.empty:
        raise RuntimeError("A qualifying homology edge crosses sequence-family clusters")
    return assignments, audited


def _cluster_summary(
    assignments: pd.DataFrame,
    *,
    identity_threshold: float,
    coverage_threshold: float,
) -> dict[str, int | float]:
    sizes = assignments.groupby("family_id")["uniprot_id"].nunique()
    return {
        "identity_threshold": identity_threshold,
        "coverage_threshold": coverage_threshold,
        "n_families": len(sizes),
        "singleton_families": int(sizes.eq(1).sum()),
        "multi_protein_families": int(sizes.gt(1).sum()),
        "proteins_in_multi_protein_families": int(sizes.loc[sizes.gt(1)].sum()),
        "largest_family_size": int(sizes.max()),
    }


def build_sequence_family_clusters(
    reference_path: Path,
    eligibility: pd.DataFrame,
    *,
    identity_threshold: float = 0.30,
    coverage_threshold: float = 0.80,
    search_identity_floor: float = 0.15,
    search_coverage_floor: float = 0.50,
    binary: str = "mmseqs",
    threads: int = 8,
) -> FamilyClusteringResult:
    """Cluster complete UniProt units through exhaustive assayed-sequence alignments."""
    if not 0 < search_identity_floor <= identity_threshold <= 1:
        raise ValueError("Identity thresholds must satisfy 0 < floor <= primary <= 1")
    if not 0 < search_coverage_floor <= coverage_threshold <= 1:
        raise ValueError("Coverage thresholds must satisfy 0 < floor <= primary <= 1")
    if threads < 1:
        raise ValueError("MMseqs2 thread count must be positive")
    segments = _eligible_segments(reference_path, eligibility)
    alignments, version = _run_mmseqs(
        segments,
        binary=binary,
        identity_floor=search_identity_floor,
        coverage_floor=search_coverage_floor,
        threads=threads,
    )
    assignments, audited = cluster_from_alignments(
        segments,
        alignments,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
    )
    sensitivity_rows = []
    for identity in sorted({0.20, 0.25, identity_threshold, 0.40}):
        if identity < search_identity_floor:
            continue
        for coverage in sorted({search_coverage_floor, coverage_threshold}):
            current, _ = cluster_from_alignments(
                segments,
                alignments,
                identity_threshold=identity,
                coverage_threshold=coverage,
            )
            sensitivity_rows.append(
                _cluster_summary(
                    current,
                    identity_threshold=identity,
                    coverage_threshold=coverage,
                )
            )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["coverage_threshold", "identity_threshold"]
    ).reset_index(drop=True)
    summary = _cluster_summary(
        assignments,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
    )
    qualifying = audited.loc[audited["qualifies_family_edge"]]
    protein_pairs = {
        tuple(sorted((str(row.query_uniprot_id), str(row.target_uniprot_id))))
        for row in qualifying.itertuples(index=False)
        if row.query_uniprot_id != row.target_uniprot_id
    }
    audit = pd.DataFrame(
        [
            {
                "method": "MMseqs2 exhaustive all-versus-all connected components",
                "mmseqs_version": version,
                "sequence_scope": "ProteinGym MSA_start:MSA_end assayed segment",
                "coverage_mode": "bidirectional",
                "search_identity_floor": search_identity_floor,
                "search_coverage_floor": search_coverage_floor,
                "eligible_assays": int(segments["assay_id"].nunique()),
                "proteins": int(segments["uniprot_id"].nunique()),
                "assayed_sequences": int(segments["sequence_sha256"].nunique()),
                **summary,
                "qualifying_protein_pairs": len(protein_pairs),
                "cross_family_qualifying_alignments": int(
                    (
                        qualifying["query_family_id"]
                        != qualifying["target_family_id"]
                    ).sum()
                ),
            }
        ]
    )
    return FamilyClusteringResult(assignments, audited, sensitivity, audit)
