from pathlib import Path

import numpy as np
import pandas as pd

from variantshift.evaluate import run_benchmark
from variantshift.models import MeanBaseline
from variantshift.report import render_report


def benchmark_frame() -> pd.DataFrame:
    singles = [f"A{position}C" for position in range(1, 21)]
    doubles = [
        f"A{position}C/A{position + 1}D" for position in range(1, 20, 2)
    ]
    codes = singles + doubles
    depth = [1] * len(singles) + [2] * len(doubles)
    signal = np.linspace(0.0, 1.0, len(codes))
    return pd.DataFrame(
        {
            "mutation_codes": codes,
            "goi_amino_mutations": depth,
            "log_ec50_prot_Sal10": signal,
        }
    )


def test_benchmark_runs_all_split_regimes() -> None:
    results = run_benchmark(
        benchmark_frame(),
        targets=("log_ec50_prot_Sal10",),
        model_factories={"mean": MeanBaseline},
    )
    assert set(results["split"]) == {
        "random_variant",
        "position_holdout",
        "mutation_depth",
    }
    assert (results["exact_variant_overlap"] == 0).all()
    assert results.loc[
        results["split"].eq("position_holdout"), "shared_position_count"
    ].item() == 0


def test_report_is_standalone_html(tmp_path: Path) -> None:
    rows = []
    for split, score, coverage in [
        ("random_variant", 0.8, 0.8),
        ("position_holdout", 0.4, 0.6),
        ("mutation_depth", 0.7, 0.7),
    ]:
        for model in ["mean", "biophysical_ridge", "additive_ridge"]:
            rows.append(
                {
                    "split": split,
                    "target": "log_ec50_prot_Sal10",
                    "model": model,
                    "spearman": score if model != "mean" else 0.0,
                    "rmse": 1.0,
                    "observed_coverage": coverage,
                    "test_rows": 100,
                }
            )
    output = render_report(pd.DataFrame(rows), tmp_path / "report.html", filtered_rows=300)
    contents = output.read_text()
    assert contents.startswith("<!doctype html>")
    assert "Random splits hide" in contents
    assert "0.40 Spearman" in contents
    assert "https://" not in contents

