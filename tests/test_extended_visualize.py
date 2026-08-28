from pathlib import Path

import pandas as pd

from variantshift.extended_visualize import render_extended_figure


def test_render_extended_figure(tmp_path: Path) -> None:
    official = pd.DataFrame(
        [
            {
                "model": model,
                "split": split,
                "mean_spearman": value,
                "n_assays": 3,
                "n_proteins": 2,
            }
            for model, value in (
                ("esm1v_embedding_probe", 0.5),
                ("protein_npt", 0.6),
                ("kermut", 0.7),
            )
            for split in ("random_variant", "contiguous_position")
        ]
    )
    probe = pd.DataFrame(
        [
            {
                "split": split,
                "calibration_method": method,
                "mean_spearman": 0.4,
                "mean_observed_coverage": 0.8,
                "mean_normalized_mean_interval_width": 1.5,
            }
            for split in (
                "random_variant",
                "position_holdout",
                "contiguous_position",
            )
            for method in (
                "standard_split",
                "mondrian_substitution",
                "position_distance_scaled",
            )
        ]
    )
    heldout = pd.DataFrame(
        [
            {
                "model": model,
                "calibration_method": "standard_split",
                "mean_spearman": value,
                "mean_observed_coverage": 0.8,
            }
            for model, value in (
                ("cross_protein_ridge", 0.5),
                ("cross_protein_histgb", 0.55),
            )
        ]
    )
    crossover = pd.DataFrame(
        [
            {
                "model": "logistic",
                "roc_auc": 0.8,
                "accuracy": 0.75,
                "majority_accuracy": 0.6,
                "n_examples": 15,
            }
        ]
    )

    output = render_extended_figure(
        official,
        probe,
        heldout,
        crossover,
        tmp_path / "figure.svg",
        heldout_family=heldout,
        heldout_structure_family=heldout,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "Harder splits change model rankings" in rendered
    assert "Crossover predictor" in rendered
    assert "seq+structure" in rendered
