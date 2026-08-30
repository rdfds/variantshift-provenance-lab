import numpy as np
import pandas as pd

from variantshift.modern_zero_shot import (
    compare_modern_to_baseline,
    summarize_modern_zero_shot,
)


def _runs() -> pd.DataFrame:
    rows = []
    for model, offset in (("esm2_650m", 0.0), ("new_model", 0.1)):
        for protein in range(4):
            rows.append(
                {
                    "model": model,
                    "score_column": model,
                    "modality": "sequence",
                    "assay_id": f"A{protein}",
                    "uniprot_id": f"P{protein}",
                    "spearman": 0.2 + protein * 0.01 + offset,
                    "top_recall": 0.3 + offset,
                    "selection_gain_sd": 0.4 + offset,
                    "best_variant_regret_sd": 0.5 - offset,
                }
            )
    return pd.DataFrame(rows)


def test_modern_summary_bootstraps_proteins():
    summary = summarize_modern_zero_shot(_runs(), bootstrap_repeats=100)

    assert summary["n_proteins"].eq(4).all()
    assert summary.iloc[0]["model"] == "new_model"
    assert summary.filter(like="ci_").notna().all().all()


def test_modern_comparison_is_exactly_paired():
    comparison = compare_modern_to_baseline(_runs(), bootstrap_repeats=100)
    new_model = comparison.loc[comparison["model"].eq("new_model")].iloc[0]

    assert np.isclose(new_model["mean_paired_delta"], 0.1)
    assert new_model["n_proteins"] == 4
    assert new_model["probability_delta_above_zero"] == 1.0
    assert new_model["bootstrap_probability_best"] == 1.0
    assert new_model["simultaneous_delta_ci_low"] > 0
