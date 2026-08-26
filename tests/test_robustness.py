import pandas as pd
import pytest

from variantshift.models import MeanBaseline
from variantshift.robustness import (
    generalization_gaps,
    run_repeated_benchmark,
    seed_schedule,
    summarize_generalization_gaps,
    summarize_robustness,
)


def analysis_frame() -> pd.DataFrame:
    singles = [f"A{position}C" for position in range(1, 31)]
    doubles = [f"A{position}C/A{position + 1}D" for position in range(1, 30, 2)]
    codes = singles + doubles
    return pd.DataFrame(
        {
            "mutation_codes": codes,
            "goi_amino_mutations": [1] * len(singles) + [2] * len(doubles),
            "activity": [float(index) for index in range(len(codes))],
        }
    )


def test_seed_schedule_is_transparent_and_validated() -> None:
    assert seed_schedule(start_seed=7, repeats=3) == (7, 8, 9)
    with pytest.raises(ValueError, match="at least two"):
        seed_schedule(repeats=1)


def test_repeated_benchmark_and_gap_summaries_preserve_seed_pairing() -> None:
    runs = run_repeated_benchmark(
        analysis_frame(),
        targets=("activity",),
        start_seed=7,
        repeats=3,
        model_factories={"mean": MeanBaseline},
    )

    assert set(runs["seed"]) == {7, 8, 9}
    assert len(runs) == 9
    assert (runs["exact_variant_overlap"] == 0).all()

    summary = summarize_robustness(runs)
    assert set(summary["n_seeds"]) == {3}
    assert set(summary["split"]) == {
        "random_variant",
        "position_holdout",
        "mutation_depth",
    }

    gaps = generalization_gaps(runs)
    assert len(gaps) == 3
    assert gaps["spearman_gap"].eq(
        gaps["random_spearman"] - gaps["position_spearman"]
    ).all()

    gap_summary = summarize_generalization_gaps(gaps)
    assert gap_summary["n_seeds"].item() == 3
