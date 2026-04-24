from pathlib import Path

import pandas as pd
import pytest

from variantshift.data import quality_filter, read_tev_dataset, summarize


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["wt", "single", "stop", "indel"],
            "mutation_codes": ["", "A1C", "A1*", "A1del"],
            "goi_amino_mutations": [0, 1, 1, 1],
            "has_amino_indel": [False, False, False, True],
            "goi_amino_seq": ["A*", "C*", "**", "*"],
            "library_type": ["SSVL", "SSVL", "EP", "EP"],
            "log_ec50_prot_Sal10": [1.0, 2.0, 3.0, 4.0],
            "log_ec50_prot_Sal25": [1.1, 2.1, 3.1, 4.1],
            "total_counts": [10_000, 2_000, 2_000, 2_000],
        }
    )


def test_quality_filter_excludes_indels_and_stops() -> None:
    filtered = quality_filter(example_frame())
    assert filtered["variant"].to_list() == ["wt", "single"]


def test_summary_counts_variant_classes() -> None:
    summary = summarize(example_frame())
    assert summary.rows == 4
    assert summary.wild_type_rows == 1
    assert summary.substitution_rows == 2
    assert summary.indel_rows == 1


def test_reader_fails_fast_on_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("variant,value\na,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        read_tev_dataset(path)

