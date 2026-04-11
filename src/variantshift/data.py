"""Data access and schema validation for the TEV GROQ-seq release."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


DATASET_URL = (
    "https://zenodo.org/records/19713341/files/"
    "TEV_Pilot_SSVL_EP_output_v1.1.zip?download=1"
)
DATASET_FILENAME = "TEV_Pilot_SSVL_EP_output_v1.1.csv"
DATASET_PAGE = "https://data.alignbio.org/groqseq/groqseq-014/"

REQUIRED_COLUMNS = {
    "variant",
    "mutation_codes",
    "goi_amino_mutations",
    "has_amino_indel",
    "goi_amino_seq",
    "library_type",
    "log_ec50_prot_Sal10",
    "log_ec50_prot_Sal25",
    "total_counts",
}


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    wild_type_rows: int
    substitution_rows: int
    indel_rows: int
    ep_rows: int
    ssvl_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def download_dataset(destination: Path, *, accept_data_use_agreement: bool) -> Path:
    """Download and extract the source CSV after an explicit DUA acknowledgement."""
    if not accept_data_use_agreement:
        raise ValueError(
            "Download requires --accept-data-use-agreement. Review terms at " + DATASET_PAGE
        )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / DATASET_FILENAME
    if csv_path.exists():
        return csv_path

    archive_path = destination / "tev-groqseq-v1.1.zip"
    request = Request(DATASET_URL, headers={"User-Agent": "VariantShift/0.1 research client"})
    with urlopen(request, timeout=120) as response, archive_path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)

    with ZipFile(archive_path) as archive:
        member = archive.getinfo(DATASET_FILENAME)
        if Path(member.filename).name != member.filename:
            raise ValueError("Unexpected archive path")
        archive.extract(member, destination)
    archive_path.unlink()
    return csv_path


def read_tev_dataset(path: Path) -> pd.DataFrame:
    """Load the Align CSV and enforce the columns needed by VariantShift."""
    frame = pd.read_csv(path, comment="#", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["mutation_codes"] = frame["mutation_codes"].fillna("").astype(str)
    frame["goi_amino_mutations"] = pd.to_numeric(
        frame["goi_amino_mutations"], errors="raise"
    ).astype(int)
    frame["has_amino_indel"] = frame["has_amino_indel"].astype(bool)
    frame["total_counts"] = pd.to_numeric(frame["total_counts"], errors="coerce")
    return frame


def quality_filter(
    frame: pd.DataFrame,
    *,
    min_total_counts: int = 1_000,
    substitutions_only: bool = True,
    exclude_stop: bool = True,
) -> pd.DataFrame:
    """Apply transparent, conservative filters without imputing measurements."""
    mask = frame["total_counts"].ge(min_total_counts)
    if substitutions_only:
        mask &= ~frame["has_amino_indel"]
    if exclude_stop:
        mask &= ~frame["mutation_codes"].str.contains("*", regex=False)
    return frame.loc[mask].reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> DatasetSummary:
    wild_type = frame["goi_amino_mutations"].eq(0)
    indel = frame["has_amino_indel"]
    return DatasetSummary(
        rows=len(frame),
        columns=len(frame.columns),
        wild_type_rows=int(wild_type.sum()),
        substitution_rows=int((~wild_type & ~indel).sum()),
        indel_rows=int(indel.sum()),
        ep_rows=int(frame["library_type"].eq("EP").sum()),
        ssvl_rows=int(frame["library_type"].eq("SSVL").sum()),
    )


def condition_columns(frame: pd.DataFrame, prefix: str = "mean_y") -> list[str]:
    """Return measured condition columns while excluding uncertainty companions."""
    return [
        column
        for column in frame.columns
        if column.startswith(prefix + "_S") and not column.endswith("_err")
    ]

