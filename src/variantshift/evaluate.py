"""End-to-end benchmarking across random and shifted evaluation regimes."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .metrics import conformal_radius, interval_coverage, regression_metrics
from .models import VariantRegressor, baseline_factories
from .splits import (
    VariantSplit,
    leakage_audit,
    mutation_depth_split,
    position_holdout_split,
    random_variant_split,
)


DEFAULT_TARGETS = ("log_ec50_prot_Sal10", "log_ec50_prot_Sal25")


def default_splits(frame: pd.DataFrame, seed: int = 42) -> list[VariantSplit]:
    return [
        random_variant_split(frame, seed=seed),
        position_holdout_split(frame, seed=seed),
        mutation_depth_split(frame),
    ]


def _finite_indices(frame: pd.DataFrame, target: str, indices: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(frame.iloc[indices][target], errors="coerce").to_numpy(dtype=float)
    return indices[np.isfinite(values)]


def run_benchmark(
    frame: pd.DataFrame,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    seed: int = 42,
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    model_factories: dict[str, Callable[[], VariantRegressor]] | None = None,
) -> pd.DataFrame:
    """Evaluate every baseline under each split with split-conformal intervals."""
    factories = model_factories or baseline_factories()
    results: list[dict[str, object]] = []

    for split in default_splits(frame, seed=seed):
        audit = leakage_audit(frame, split)
        for target in targets:
            if target not in frame:
                raise ValueError(f"Target column not found: {target}")
            train_indices = _finite_indices(frame, target, split.train_indices)
            test_indices = _finite_indices(frame, target, split.test_indices)
            fit_indices, calibration_indices = train_test_split(
                train_indices,
                test_size=calibration_fraction,
                random_state=seed,
            )

            fit_codes = frame.iloc[fit_indices]["mutation_codes"].astype(str).to_list()
            calibration_codes = (
                frame.iloc[calibration_indices]["mutation_codes"].astype(str).to_list()
            )
            test_codes = frame.iloc[test_indices]["mutation_codes"].astype(str).to_list()
            fit_target = frame.iloc[fit_indices][target].to_numpy(dtype=float)
            calibration_target = frame.iloc[calibration_indices][target].to_numpy(dtype=float)
            test_target = frame.iloc[test_indices][target].to_numpy(dtype=float)

            for model_name, factory in factories.items():
                model = factory()
                model.fit(fit_codes, fit_target)
                calibration_prediction = model.predict(calibration_codes)
                prediction = model.predict(test_codes)
                radius = conformal_radius(
                    calibration_target - calibration_prediction,
                    coverage=coverage,
                )
                metrics = regression_metrics(test_target, prediction)
                results.append(
                    {
                        "split": split.name,
                        "target": target,
                        "model": model_name,
                        **metrics.to_dict(),
                        "nominal_coverage": coverage,
                        "observed_coverage": interval_coverage(
                            test_target, prediction, radius
                        ),
                        "interval_width": 2 * radius,
                        "fit_rows": len(fit_indices),
                        "calibration_rows": len(calibration_indices),
                        "test_rows": len(test_indices),
                        "exact_variant_overlap": audit["exact_variant_overlap"],
                        "shared_position_count": audit["shared_position_count"],
                    }
                )
    return pd.DataFrame(results).sort_values(["target", "split", "model"]).reset_index(
        drop=True
    )

