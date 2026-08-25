"""Cross-condition transfer analysis for protein variant-effect models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .data import condition_columns
from .metrics import regression_metrics
from .models import AdditiveRidge, VariantRegressor
from .splits import VariantSplit, leakage_audit, position_holdout_split, random_variant_split


def default_transfer_splits(frame: pd.DataFrame, seed: int = 42) -> list[VariantSplit]:
    return [
        random_variant_split(frame, seed=seed),
        position_holdout_split(frame, seed=seed),
    ]


def run_condition_transfer(
    frame: pd.DataFrame,
    *,
    conditions: Sequence[str] | None = None,
    seed: int = 42,
    model_factory: Callable[[], VariantRegressor] = AdditiveRidge,
    splits: Sequence[VariantSplit] | None = None,
) -> pd.DataFrame:
    """Fit on one assay condition and rank held-out variants in every condition.

    Each source-condition model is trained once per split. Its held-out predictions are
    evaluated against every target condition, producing a transfer matrix without fitting
    on target-condition labels.
    """
    selected_conditions = tuple(conditions or condition_columns(frame))
    if len(selected_conditions) < 2:
        raise ValueError("Condition transfer requires at least two assay conditions")
    missing = sorted(set(selected_conditions).difference(frame.columns))
    if missing:
        raise ValueError(f"Condition columns not found: {', '.join(missing)}")

    numeric = frame.loc[:, selected_conditions].apply(pd.to_numeric, errors="coerce")
    finite_rows = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    selected_splits = list(splits or default_transfer_splits(frame, seed=seed))
    rows: list[dict[str, object]] = []

    for split in selected_splits:
        train_indices = split.train_indices[finite_rows[split.train_indices]]
        test_indices = split.test_indices[finite_rows[split.test_indices]]
        if len(train_indices) < 2 or len(test_indices) < 2:
            raise ValueError(f"Split {split.name} has insufficient finite condition rows")

        audit = leakage_audit(frame, split)
        train_codes = frame.iloc[train_indices]["mutation_codes"].astype(str).to_list()
        test_codes = frame.iloc[test_indices]["mutation_codes"].astype(str).to_list()
        test_values = numeric.iloc[test_indices]

        for source_condition in selected_conditions:
            model = model_factory()
            model.fit(
                train_codes,
                numeric.iloc[train_indices][source_condition].to_numpy(dtype=float),
            )
            prediction = model.predict(test_codes)
            source_observed = test_values[source_condition].to_numpy(dtype=float)
            source_spearman = regression_metrics(source_observed, prediction).spearman

            for target_condition in selected_conditions:
                target_observed = test_values[target_condition].to_numpy(dtype=float)
                transfer_spearman = regression_metrics(target_observed, prediction).spearman
                assay_spearman = regression_metrics(
                    source_observed, target_observed
                ).spearman
                rows.append(
                    {
                        "split": split.name,
                        "seed": seed,
                        "model": model.name,
                        "source_condition": source_condition,
                        "target_condition": target_condition,
                        "source_spearman": source_spearman,
                        "transfer_spearman": transfer_spearman,
                        "transfer_gap": source_spearman - transfer_spearman,
                        "assay_spearman": assay_spearman,
                        "train_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        "exact_variant_overlap": audit["exact_variant_overlap"],
                        "shared_position_count": audit["shared_position_count"],
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["split", "source_condition", "target_condition"]
    ).reset_index(drop=True)


def summarize_condition_transfer(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize in-condition accuracy and off-diagonal transfer by split."""
    required = {
        "split",
        "source_condition",
        "target_condition",
        "source_spearman",
        "transfer_spearman",
        "transfer_gap",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Transfer results are missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for split, group in results.groupby("split", sort=True):
        diagonal = group["source_condition"].eq(group["target_condition"])
        off_diagonal = group.loc[~diagonal]
        rows.append(
            {
                "split": split,
                "conditions": int(group["source_condition"].nunique()),
                "diagonal_spearman_mean": float(group.loc[diagonal, "transfer_spearman"].mean()),
                "off_diagonal_spearman_mean": float(
                    off_diagonal["transfer_spearman"].mean()
                ),
                "off_diagonal_spearman_min": float(
                    off_diagonal["transfer_spearman"].min()
                ),
                "off_diagonal_spearman_max": float(
                    off_diagonal["transfer_spearman"].max()
                ),
                "off_diagonal_gap_mean": float(off_diagonal["transfer_gap"].mean()),
                "off_diagonal_gap_max": float(off_diagonal["transfer_gap"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
