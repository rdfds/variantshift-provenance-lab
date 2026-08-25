from pathlib import Path

import pandas as pd
import pytest

from variantshift.visualize import render_shift_figure


def test_shift_figure_is_standalone_and_data_driven(tmp_path: Path) -> None:
    gaps = pd.DataFrame(
        {
            "seed": [42, 43, 42, 43],
            "target": ["Sal10", "Sal10", "Sal25", "Sal25"],
            "model": ["additive_ridge"] * 4,
            "spearman_gap": [0.3, 0.4, 0.2, 0.25],
        }
    )
    rows = []
    for source in ["mean_y_S1", "mean_y_S2"]:
        for target in ["mean_y_S1", "mean_y_S2"]:
            rows.append(
                {
                    "split": "position_holdout",
                    "model": "additive_ridge",
                    "source_condition": source,
                    "target_condition": target,
                    "transfer_spearman": 0.5 if source == target else 0.3,
                    "exact_variant_overlap": 0,
                }
            )
    output = render_shift_figure(gaps, pd.DataFrame(rows), tmp_path / "shift.svg")
    contents = output.read_text()

    assert contents.startswith("<svg")
    assert "ROBUSTNESS UNDER BIOLOGICAL SHIFT" in contents
    assert "2×2" in contents
    assert "0 exact variants shared" not in contents
    assert "https://" not in contents


def test_shift_figure_rejects_incomplete_transfer_matrix(tmp_path: Path) -> None:
    gaps = pd.DataFrame(
        {
            "seed": [42, 43],
            "target": ["Sal10", "Sal10"],
            "model": ["additive_ridge"] * 2,
            "spearman_gap": [0.3, 0.4],
        }
    )
    transfer = pd.DataFrame(
        {
            "split": ["position_holdout"],
            "model": ["additive_ridge"],
            "source_condition": ["mean_y_S1"],
            "target_condition": ["mean_y_S2"],
            "transfer_spearman": [0.3],
            "exact_variant_overlap": [0],
        }
    )

    with pytest.raises(ValueError, match="incomplete"):
        render_shift_figure(gaps, transfer, tmp_path / "shift.svg")
