import numpy as np
import pandas as pd
import pytest

from variantshift.models import BiophysicalRidge
from variantshift.transfer import run_condition_transfer, summarize_condition_transfer


def transfer_frame() -> pd.DataFrame:
    singles = [f"A{position}C" for position in range(1, 31)]
    doubles = [f"A{position}C/A{position + 1}D" for position in range(1, 30, 2)]
    codes = singles + doubles
    base = np.linspace(-1.0, 1.0, len(codes))
    return pd.DataFrame(
        {
            "mutation_codes": codes,
            "goi_amino_mutations": [1] * len(singles) + [2] * len(doubles),
            "mean_y_S1": base,
            "mean_y_S2": 2.0 * base + 0.1,
            "mean_y_S3": -base,
        }
    )


def test_condition_transfer_builds_complete_matrices_without_leakage() -> None:
    results = run_condition_transfer(
        transfer_frame(),
        conditions=("mean_y_S1", "mean_y_S2", "mean_y_S3"),
        model_factory=BiophysicalRidge,
    )

    assert len(results) == 18
    assert set(results["split"]) == {"random_variant", "position_holdout"}
    assert (results["exact_variant_overlap"] == 0).all()
    assert results.loc[
        results["split"].eq("position_holdout"), "shared_position_count"
    ].eq(0).all()
    diagonal = results["source_condition"].eq(results["target_condition"])
    assert results.loc[diagonal, "transfer_gap"].to_numpy() == pytest.approx(0.0)

    summary = summarize_condition_transfer(results)
    assert set(summary["conditions"]) == {3}
    assert set(summary["split"]) == {"random_variant", "position_holdout"}


def test_condition_transfer_requires_multiple_conditions() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_condition_transfer(transfer_frame(), conditions=("mean_y_S1",))
