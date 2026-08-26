"""Repeated-split diagnostics for generalization and conformal coverage."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .evaluate import DEFAULT_TARGETS, run_benchmark
from .models import VariantRegressor

ROBUSTNESS_METRICS = ("spearman", "observed_coverage", "interval_width")
GAP_METRICS = ("spearman_gap", "coverage_gap")


def seed_schedule(start_seed: int = 42, repeats: int = 10) -> tuple[int, ...]:
    """Return a transparent deterministic schedule of consecutive split seeds."""
    if repeats < 2:
        raise ValueError("Robustness analysis requires at least two repeated splits")
    if start_seed < 0:
        raise ValueError("Start seed must be non-negative")
    return tuple(range(start_seed, start_seed + repeats))


def run_repeated_benchmark(
    frame: pd.DataFrame,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    start_seed: int = 42,
    repeats: int = 10,
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    model_factories: dict[str, Callable[[], VariantRegressor]] | None = None,
) -> pd.DataFrame:
    """Run the complete benchmark across a deterministic schedule of split seeds."""
    runs = [
        run_benchmark(
            frame,
            targets=targets,
            seed=seed,
            calibration_fraction=calibration_fraction,
            coverage=coverage,
            model_factories=model_factories,
        )
        for seed in seed_schedule(start_seed=start_seed, repeats=repeats)
    ]
    return pd.concat(runs, ignore_index=True).sort_values(
        ["seed", "target", "split", "model"]
    ).reset_index(drop=True)


def _distribution(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = values.to_numpy(dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(numeric)),
        f"{prefix}_std": float(np.std(numeric, ddof=1)),
        f"{prefix}_min": float(np.min(numeric)),
        f"{prefix}_p05": float(np.quantile(numeric, 0.05)),
        f"{prefix}_p95": float(np.quantile(numeric, 0.95)),
        f"{prefix}_max": float(np.max(numeric)),
    }


def summarize_robustness(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed sensitivity without treating repeated seeds as new data."""
    required = {"seed", "split", "target", "model", *ROBUSTNESS_METRICS}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Repeated benchmark is missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for (split, target, model), group in runs.groupby(
        ["split", "target", "model"], sort=True
    ):
        row: dict[str, object] = {
            "split": split,
            "target": target,
            "model": model,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in ROBUSTNESS_METRICS:
            row.update(_distribution(group[metric], metric))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "split", "model"]).reset_index(
        drop=True
    )


def generalization_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair random and unseen-position results within each seed."""
    required = {"seed", "split", "target", "model", "spearman", "observed_coverage"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Repeated benchmark is missing columns: {', '.join(sorted(missing))}")

    selected = runs.loc[
        runs["split"].isin(["random_variant", "position_holdout"]),
        ["seed", "target", "model", "split", "spearman", "observed_coverage"],
    ]
    pivot = selected.pivot(
        index=["seed", "target", "model"],
        columns="split",
        values=["spearman", "observed_coverage"],
    )
    needed = {
        ("spearman", "random_variant"),
        ("spearman", "position_holdout"),
        ("observed_coverage", "random_variant"),
        ("observed_coverage", "position_holdout"),
    }
    if not needed.issubset(set(pivot.columns)):
        raise ValueError("Repeated benchmark must include random and unseen-position results")

    output = pivot.index.to_frame(index=False)
    output["random_spearman"] = pivot[("spearman", "random_variant")].to_numpy()
    output["position_spearman"] = pivot[("spearman", "position_holdout")].to_numpy()
    output["spearman_gap"] = output["random_spearman"] - output["position_spearman"]
    output["random_coverage"] = pivot[
        ("observed_coverage", "random_variant")
    ].to_numpy()
    output["position_coverage"] = pivot[
        ("observed_coverage", "position_holdout")
    ].to_numpy()
    output["coverage_gap"] = output["random_coverage"] - output["position_coverage"]
    return output.sort_values(["target", "model", "seed"]).reset_index(drop=True)


def summarize_generalization_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """Summarize the paired generalization penalty across seeds."""
    required = {"seed", "target", "model", *GAP_METRICS}
    missing = required.difference(gaps.columns)
    if missing:
        raise ValueError(f"Gap table is missing columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for (target, model), group in gaps.groupby(["target", "model"], sort=True):
        row: dict[str, object] = {
            "target": target,
            "model": model,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in GAP_METRICS:
            row.update(_distribution(group[metric], metric))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "model"]).reset_index(drop=True)
