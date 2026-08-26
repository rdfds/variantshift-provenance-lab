"""ProteinGym substitution-assay ingestion and eligibility auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from .mutations import apply_variant, parse_variant, validate_against_sequence

PROTEINGYM_VERSION = "v1.3"
PROTEINGYM_BASE_URL = (
    f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_VERSION}"
)
PROTEINGYM_ARCHIVE_NAME = "DMS_ProteinGym_substitutions.zip"
PROTEINGYM_ARCHIVE_URL = f"{PROTEINGYM_BASE_URL}/{PROTEINGYM_ARCHIVE_NAME}"
PROTEINGYM_SCORE_ARCHIVE_NAME = "zero_shot_substitutions_scores.zip"
PROTEINGYM_SCORE_ARCHIVE_URL = (
    f"{PROTEINGYM_BASE_URL}/{PROTEINGYM_SCORE_ARCHIVE_NAME}"
)
PROTEINGYM_REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    "main/reference_files/DMS_substitutions.csv"
)

REFERENCE_COLUMNS = {
    "DMS_id",
    "DMS_filename",
    "UniProt_ID",
    "target_seq",
    "taxon",
    "selection_type",
    "coarse_selection_type",
    "ProteinGym_version",
}
ASSAY_COLUMNS = {"mutant", "mutated_sequence", "DMS_score"}


@dataclass(frozen=True)
class EligibilityCriteria:
    """Predeclared inclusion criteria for the primary single-substitution study."""

    min_single_variants: int = 500
    min_positions: int = 20
    min_unique_scores: int = 10

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _download(url: str, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "VariantShift/0.2 research client"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    partial.replace(destination)
    return destination


def download_proteingym(
    destination: Path,
    *,
    include_zero_shot_scores: bool = False,
) -> dict[str, Path]:
    """Download public ProteinGym inputs, optionally including the 1.9 GB score archive."""
    destination = Path(destination)
    outputs = {
        "archive": _download(PROTEINGYM_ARCHIVE_URL, destination / PROTEINGYM_ARCHIVE_NAME),
        "reference": _download(PROTEINGYM_REFERENCE_URL, destination / "DMS_substitutions.csv"),
    }
    if include_zero_shot_scores:
        outputs["zero_shot_scores"] = _download(
            PROTEINGYM_SCORE_ARCHIVE_URL,
            destination / PROTEINGYM_SCORE_ARCHIVE_NAME,
        )
    return outputs


def read_reference_index(path: Path) -> pd.DataFrame:
    """Read the assay index and enforce identifiers needed by the benchmark."""
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(REFERENCE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ProteinGym index is missing columns: {', '.join(missing)}")
    if frame["DMS_id"].duplicated().any() or frame["DMS_filename"].duplicated().any():
        raise ValueError("ProteinGym assay identifiers and filenames must be unique")
    return frame.copy()


def _archive_members(archive: ZipFile) -> dict[str, str]:
    members = {
        PurePosixPath(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".csv")
    }
    if len(members) != len(set(members)):
        raise ValueError("ProteinGym archive contains duplicate assay filenames")
    return members


def read_assay_member(archive: ZipFile, member: str) -> pd.DataFrame:
    """Read one processed assay from an already-open ProteinGym archive."""
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, low_memory=False)
    missing = sorted(ASSAY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"ProteinGym assay is missing columns: {', '.join(missing)}")
    return frame


def canonicalize_assay(frame: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:
    """Convert finite single substitutions to VariantShift's canonical schema."""
    scores = pd.to_numeric(frame["DMS_score"], errors="coerce")
    depth = frame["mutant"].astype(str).str.count(":") + 1
    selected = frame.loc[scores.notna() & depth.eq(1), ["mutant", "DMS_score"]].copy()
    selected["mutation_codes"] = selected.pop("mutant").astype(str).str.replace(
        ":", "/", regex=False
    )
    selected["DMS_score"] = pd.to_numeric(selected["DMS_score"], errors="raise")
    selected["goi_amino_mutations"] = 1
    selected["assay_id"] = str(metadata["DMS_id"])
    selected["uniprot_id"] = str(metadata["UniProt_ID"])
    selected["taxon"] = str(metadata["taxon"])
    selected["selection_type"] = str(metadata["selection_type"])
    selected["coarse_selection_type"] = str(metadata["coarse_selection_type"])
    return selected.reset_index(drop=True)


def _audit_assay(
    frame: pd.DataFrame,
    metadata: pd.Series,
    criteria: EligibilityCriteria,
) -> dict[str, object]:
    target_sequence = str(metadata["target_seq"])
    scores = pd.to_numeric(frame["DMS_score"], errors="coerce")
    mutations = frame["mutant"].astype(str)
    depth = mutations.str.count(":") + 1
    single_mask = scores.notna() & depth.eq(1)
    singles = frame.loc[single_mask, ["mutant", "mutated_sequence"]]

    invalid_codes = 0
    reference_mismatches = 0
    mutated_sequence_mismatches = 0
    positions: set[int] = set()
    for mutation_code, mutated_sequence in singles.itertuples(index=False, name=None):
        try:
            parsed = parse_variant(str(mutation_code).replace(":", "/"))
            validate_against_sequence(str(mutation_code).replace(":", "/"), target_sequence)
        except ValueError as error:
            if "Reference mismatch" in str(error) or "exceeds sequence length" in str(error):
                reference_mismatches += 1
            else:
                invalid_codes += 1
            continue
        positions.add(parsed[0].position)
        if str(mutated_sequence) != apply_variant(str(mutation_code), target_sequence):
            mutated_sequence_mismatches += 1

    single_scores = scores.loc[single_mask]
    duplicate_variants = int(mutations.loc[single_mask].duplicated().sum())
    reasons: list[str] = []
    if len(single_scores) < criteria.min_single_variants:
        reasons.append("too_few_single_variants")
    if len(positions) < criteria.min_positions:
        reasons.append("too_few_positions")
    if single_scores.nunique() < criteria.min_unique_scores:
        reasons.append("too_few_unique_scores")
    if invalid_codes:
        reasons.append("invalid_mutation_codes")
    if reference_mismatches:
        reasons.append("reference_sequence_mismatch")
    if mutated_sequence_mismatches:
        reasons.append("mutated_sequence_mismatch")
    if duplicate_variants:
        reasons.append("duplicate_single_variants")

    return {
        "assay_id": str(metadata["DMS_id"]),
        "filename": str(metadata["DMS_filename"]),
        "uniprot_id": str(metadata["UniProt_ID"]),
        "taxon": str(metadata["taxon"]),
        "selection_type": str(metadata["selection_type"]),
        "coarse_selection_type": str(metadata["coarse_selection_type"]),
        "protein_length": len(target_sequence),
        "total_rows": len(frame),
        "finite_rows": int(scores.notna().sum()),
        "single_variants": len(single_scores),
        "multiple_variants": int((scores.notna() & depth.gt(1)).sum()),
        "mutated_positions": len(positions),
        "unique_scores": int(single_scores.nunique()),
        "invalid_mutation_codes": invalid_codes,
        "reference_mismatches": reference_mismatches,
        "mutated_sequence_mismatches": mutated_sequence_mismatches,
        "duplicate_single_variants": duplicate_variants,
        "eligible": not reasons,
        "exclusion_reasons": ";".join(reasons),
    }


def audit_archive(
    archive_path: Path,
    reference_path: Path,
    *,
    criteria: EligibilityCriteria | None = None,
) -> pd.DataFrame:
    """Audit every indexed assay and return an explicit inclusion ledger."""
    criteria = criteria or EligibilityCriteria()
    reference = read_reference_index(reference_path)
    results: list[dict[str, object]] = []
    with ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        for _, metadata in reference.iterrows():
            filename = str(metadata["DMS_filename"])
            member = members.get(filename)
            if member is None:
                results.append(
                    {
                        "assay_id": str(metadata["DMS_id"]),
                        "filename": filename,
                        "uniprot_id": str(metadata["UniProt_ID"]),
                        "eligible": False,
                        "exclusion_reasons": "missing_archive_member",
                    }
                )
                continue
            results.append(_audit_assay(read_assay_member(archive, member), metadata, criteria))
    return pd.DataFrame(results).sort_values("assay_id").reset_index(drop=True)


def iter_eligible_assays(
    archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
):
    """Yield canonical data and metadata for each assay marked eligible."""
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    with ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        for assay_id in eligible_ids:
            metadata = reference.loc[assay_id]
            filename = str(metadata["DMS_filename"])
            yield canonicalize_assay(read_assay_member(archive, members[filename]), metadata), metadata
