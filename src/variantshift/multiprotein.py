"""Repeated multi-protein validation over eligible ProteinGym assays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .metrics import conformal_radius, interval_coverage, regression_metrics
from .models import VariantRegressor, baseline_factories
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    iter_eligible_assays,
    read_assay_member,
    read_reference_index,
)
from .robustness import seed_schedule
from .splits import leakage_audit, position_holdout_split, random_variant_split

PRIMARY_SPLITS = ("random_variant", "position_holdout")

_WORKER_ARCHIVE: ZipFile | None = None
_WORKER_MEMBERS: dict[str, str] | None = None
_WORKER_REFERENCE: pd.DataFrame | None = None


def _initialize_worker(archive_path: Path, reference_path: Path) -> None:
    global _WORKER_ARCHIVE, _WORKER_MEMBERS, _WORKER_REFERENCE
    _WORKER_ARCHIVE = ZipFile(archive_path)
    _WORKER_MEMBERS = _archive_members(_WORKER_ARCHIVE)
    _WORKER_REFERENCE = read_reference_index(reference_path).set_index("DMS_id", drop=False)


def _evaluate_assay_worker(arguments) -> pd.DataFrame:
    assay_id, seeds, calibration_fraction, coverage = arguments
    if _WORKER_ARCHIVE is None or _WORKER_MEMBERS is None or _WORKER_REFERENCE is None:
        raise RuntimeError("ProteinGym worker was not initialized")
    metadata = _WORKER_REFERENCE.loc[assay_id]
    filename = str(metadata["DMS_filename"])
    frame = canonicalize_assay(
        read_assay_member(_WORKER_ARCHIVE, _WORKER_MEMBERS[filename]), metadata
    )
    return evaluate_proteingym_assay(
        frame,
        seeds=seeds,
        calibration_fraction=calibration_fraction,
        coverage=coverage,
    )


def evaluate_proteingym_assay(
    frame: pd.DataFrame,
    *,
    seeds: Sequence[int],
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    model_factories: dict[str, Callable[[], VariantRegressor]] | None = None,
) -> pd.DataFrame:
    """Evaluate one canonical single-substitution assay across repeated splits."""
    required = {"mutation_codes", "DMS_score", "assay_id", "uniprot_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Canonical assay is missing columns: {', '.join(missing)}")
    if frame["assay_id"].nunique() != 1 or frame["uniprot_id"].nunique() != 1:
        raise ValueError("Each evaluation frame must contain exactly one assay and protein")

    factories = model_factories or baseline_factories()
    assay_id = str(frame["assay_id"].iat[0])
    uniprot_id = str(frame["uniprot_id"].iat[0])
    target = frame["DMS_score"].to_numpy(dtype=float)
    results: list[dict[str, object]] = []

    for seed in seeds:
        splits = (
            random_variant_split(frame, seed=seed),
            position_holdout_split(frame, seed=seed),
        )
        for split in splits:
            audit = leakage_audit(frame, split)
            fit_indices, calibration_indices = train_test_split(
                split.train_indices,
                test_size=calibration_fraction,
                random_state=seed,
            )
            fit_codes = frame.iloc[fit_indices]["mutation_codes"].astype(str).tolist()
            calibration_codes = (
                frame.iloc[calibration_indices]["mutation_codes"].astype(str).tolist()
            )
            test_codes = frame.iloc[split.test_indices]["mutation_codes"].astype(str).tolist()
            fit_target = target[fit_indices]
            calibration_target = target[calibration_indices]
            test_target = target[split.test_indices]
            target_scale = float(np.std(fit_target))

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
                        "assay_id": assay_id,
                        "uniprot_id": uniprot_id,
                        "taxon": str(frame["taxon"].iat[0]),
                        "coarse_selection_type": str(
                            frame["coarse_selection_type"].iat[0]
                        ),
                        "seed": int(seed),
                        "split": split.name,
                        "model": model_name,
                        **metrics.to_dict(),
                        "normalized_rmse": metrics.rmse / target_scale
                        if target_scale > 1e-12
                        else np.nan,
                        "nominal_coverage": coverage,
                        "observed_coverage": interval_coverage(
                            test_target, prediction, radius
                        ),
                        "normalized_interval_width": 2 * radius / target_scale
                        if target_scale > 1e-12
                        else np.nan,
                        "fit_rows": len(fit_indices),
                        "calibration_rows": len(calibration_indices),
                        "test_rows": len(split.test_indices),
                        "excluded_rows": audit["excluded_rows"],
                        "exact_variant_overlap": audit["exact_variant_overlap"],
                        "shared_position_count": audit["shared_position_count"],
                    }
                )
    return pd.DataFrame(results)


def run_multiprotein_benchmark(
    archive_path,
    reference_path,
    eligibility: pd.DataFrame,
    *,
    start_seed: int = 42,
    repeats: int = 10,
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    model_factories: dict[str, Callable[[], VariantRegressor]] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Run the fixed protocol over every assay marked eligible in the ledger."""
    if workers < 1:
        raise ValueError("Worker count must be at least one")
    if workers > 1 and model_factories is not None:
        raise ValueError("Custom model factories require single-process evaluation")
    seeds = seed_schedule(start_seed=start_seed, repeats=repeats)
    frames: list[pd.DataFrame]
    if workers == 1:
        frames = []
        for assay, _ in iter_eligible_assays(archive_path, reference_path, eligibility):
            frames.append(
                evaluate_proteingym_assay(
                    assay,
                    seeds=seeds,
                    calibration_fraction=calibration_fraction,
                    coverage=coverage,
                    model_factories=model_factories,
                )
            )
    else:
        assay_ids = eligibility.loc[
            eligibility["eligible"].astype(bool), "assay_id"
        ].astype(str)
        jobs = [
            (assay_id, seeds, calibration_fraction, coverage) for assay_id in assay_ids
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(Path(archive_path), Path(reference_path)),
        ) as executor:
            frames = list(executor.map(_evaluate_assay_worker, jobs, chunksize=1))
    if not frames:
        raise ValueError("Eligibility ledger does not contain any eligible assays")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["assay_id", "seed", "split", "model"]
    ).reset_index(drop=True)


def multiprotein_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair random and unseen-position metrics within assay, model, and seed."""
    metrics = ("spearman", "observed_coverage", "normalized_rmse")
    identifiers = (
        "assay_id",
        "uniprot_id",
        "taxon",
        "coarse_selection_type",
        "seed",
        "model",
    )
    required = {*identifiers, "split", *metrics}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"Multi-protein runs are missing columns: {', '.join(missing)}")
    selected = runs.loc[runs["split"].isin(PRIMARY_SPLITS), [*identifiers, "split", *metrics]]
    pivot = selected.pivot(index=list(identifiers), columns="split", values=list(metrics))
    for metric in metrics:
        for split in PRIMARY_SPLITS:
            if (metric, split) not in pivot.columns:
                raise ValueError(f"Missing {metric} results for split {split}")

    output = pivot.index.to_frame(index=False)
    for metric in metrics:
        random_name = f"random_{metric}"
        position_name = f"position_{metric}"
        output[random_name] = pivot[(metric, "random_variant")].to_numpy()
        output[position_name] = pivot[(metric, "position_holdout")].to_numpy()
        output[f"{metric}_gap"] = output[random_name] - output[position_name]
    return output.sort_values(["model", "uniprot_id", "assay_id", "seed"]).reset_index(
        drop=True
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_multiprotein_gaps(
    gaps: pd.DataFrame,
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 2026,
) -> pd.DataFrame:
    """Aggregate within assay and protein before bootstrapping across proteins."""
    required = {
        "assay_id",
        "uniprot_id",
        "seed",
        "model",
        "random_spearman",
        "position_spearman",
        "spearman_gap",
        "observed_coverage_gap",
        "normalized_rmse_gap",
    }
    missing = sorted(required.difference(gaps.columns))
    if missing:
        raise ValueError(f"Multi-protein gaps are missing columns: {', '.join(missing)}")
    if bootstrap_repeats < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")

    assay_level = (
        gaps.groupby(["model", "uniprot_id", "assay_id"], as_index=False)
        .agg(
            random_spearman=("random_spearman", "mean"),
            position_spearman=("position_spearman", "mean"),
            spearman_gap=("spearman_gap", "mean"),
            coverage_gap=("observed_coverage_gap", "mean"),
            normalized_rmse_gap=("normalized_rmse_gap", "mean"),
        )
    )
    protein_level = (
        assay_level.groupby(["model", "uniprot_id"], as_index=False)
        .agg(
            random_spearman=("random_spearman", "mean"),
            position_spearman=("position_spearman", "mean"),
            spearman_gap=("spearman_gap", "mean"),
            coverage_gap=("coverage_gap", "mean"),
            normalized_rmse_gap=("normalized_rmse_gap", "mean"),
        )
    )

    rows: list[dict[str, object]] = []
    for offset, (model, group) in enumerate(protein_level.groupby("model", sort=True)):
        rng = np.random.default_rng(bootstrap_seed + offset)
        gap_values = group["spearman_gap"].to_numpy(dtype=float)
        gap_low, gap_high = _bootstrap_mean_interval(
            gap_values,
            repeats=bootstrap_repeats,
            rng=rng,
        )
        rows.append(
            {
                "model": model,
                "n_assays": int(assay_level.loc[assay_level["model"].eq(model), "assay_id"].nunique()),
                "n_proteins": int(group["uniprot_id"].nunique()),
                "random_spearman_mean": float(group["random_spearman"].mean()),
                "position_spearman_mean": float(group["position_spearman"].mean()),
                "spearman_gap_mean": float(gap_values.mean()),
                "spearman_gap_median": float(np.median(gap_values)),
                "spearman_gap_ci_low": gap_low,
                "spearman_gap_ci_high": gap_high,
                "proteins_with_positive_gap_fraction": float(np.mean(gap_values > 0)),
                "coverage_gap_mean": float(group["coverage_gap"].mean()),
                "normalized_rmse_gap_mean": float(group["normalized_rmse_gap"].mean()),
                "bootstrap_unit": "UniProt_ID",
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def summarize_assays(gaps: pd.DataFrame) -> pd.DataFrame:
    """Produce one transparent repeated-split summary row per assay and model."""
    columns = [
        "random_spearman",
        "position_spearman",
        "spearman_gap",
        "random_observed_coverage",
        "position_observed_coverage",
        "observed_coverage_gap",
        "random_normalized_rmse",
        "position_normalized_rmse",
        "normalized_rmse_gap",
    ]
    aggregations = {column: (column, "mean") for column in columns}
    return (
        gaps.groupby(
            ["assay_id", "uniprot_id", "taxon", "coarse_selection_type", "model"],
            as_index=False,
        )
        .agg(n_seeds=("seed", "nunique"), **aggregations)
        .sort_values(["model", "spearman_gap"], ascending=[True, False])
        .reset_index(drop=True)
    )
