"""Development-only pilot of phenotype coverage in a published predictor panel.

The pilot deliberately uses ProteinGym development data only.  It does not read any
VariantShift confirmation registry, lock, prediction, or outcome artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NON_PREDICTOR_COLUMNS = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
SINGLE_MUTATION = re.compile(r"^[A-Z](\d+)[A-Z]$")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 3 or np.ptp(left) < 1e-12 or np.ptp(right) < 1e-12:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def selection_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    count = max(1, int(np.ceil(0.1 * len(observed))))
    predicted_top = np.argpartition(predicted, -count)[-count:]
    true_top = set(np.argpartition(observed, -count)[-count:])
    scale = float(np.std(observed))
    return {
        "spearman": safe_spearman(observed, predicted),
        "selection_gain_sd": (
            float(np.mean(observed[predicted_top]) - np.mean(observed)) / scale
            if scale > 1e-12
            else 0.0
        ),
        "top_recall": len(true_top.intersection(predicted_top)) / count,
    }


def predictor_geometry(matrix: np.ndarray) -> dict[str, float | int]:
    """Summarize redundancy after predictors are rank-normalized across variants."""
    standardized = StandardScaler().fit_transform(np.asarray(matrix, dtype=float))
    variance = PCA().fit(standardized).explained_variance_ratio_
    cumulative = np.cumsum(variance)
    normalized = variance / variance.sum()
    effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized + 1e-15))))
    return {
        "pcs_80pct": int(np.searchsorted(cumulative, 0.8) + 1),
        "pcs_90pct": int(np.searchsorted(cumulative, 0.9) + 1),
        "pcs_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "effective_rank": effective_rank,
    }


def _rank_matrix(frame: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    matrix = frame[predictors].rank(pct=True).to_numpy(dtype=float).copy()
    medians = np.nanmedian(matrix, axis=0)
    missing_rows, missing_columns = np.where(~np.isfinite(matrix))
    matrix[missing_rows, missing_columns] = medians[missing_columns]
    return matrix


def _best_single_predictions(
    matrix: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    folds: GroupKFold,
) -> np.ndarray:
    predictions = np.full(len(observed), np.nan)
    for train, test in folds.split(matrix, observed, groups):
        correlations = [safe_spearman(observed[train], matrix[train, j]) for j in range(matrix.shape[1])]
        predictions[test] = matrix[test, int(np.nanargmax(correlations))]
    return predictions


def _ridge_predictions(
    matrix: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    folds: GroupKFold,
    alpha_grid: list[float],
) -> np.ndarray:
    """Nested position-grouped ridge predictions; this is not called an upper bound."""
    predictions = np.full(len(observed), np.nan)
    for train, test in folds.split(matrix, observed, groups):
        inner_groups = np.unique(groups[train])
        inner = GroupKFold(n_splits=min(4, len(inner_groups)))
        best_alpha = alpha_grid[0]
        best_score = -np.inf
        for alpha in alpha_grid:
            scores = []
            for inner_train, inner_test in inner.split(
                matrix[train], observed[train], groups[train]
            ):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(matrix[train][inner_train], observed[train][inner_train])
                scores.append(
                    safe_spearman(
                        observed[train][inner_test], model.predict(matrix[train][inner_test])
                    )
                )
            score = float(np.nanmean(scores))
            if score > best_score:
                best_score = score
                best_alpha = alpha
        model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        model.fit(matrix[train], observed[train])
        predictions[test] = model.predict(matrix[test])
    return predictions


def evaluate_task(
    frame: pd.DataFrame,
    predictors: list[str],
    *,
    minimum_predictor_coverage: float,
    outer_folds: int,
    alpha_grid: list[float],
) -> tuple[dict[str, float | int], list[dict[str, float | str]]]:
    eligible = [
        column
        for column in predictors
        if frame[column].notna().mean() >= minimum_predictor_coverage
        and frame[column].nunique(dropna=True) > 2
    ]
    matrix = _rank_matrix(frame, eligible)
    observed = frame["DMS_score"].rank(pct=True).to_numpy(dtype=float)
    groups = frame["position"].to_numpy(dtype=int)
    folds = GroupKFold(n_splits=min(outer_folds, len(np.unique(groups))))
    best_single = _best_single_predictions(matrix, observed, groups, folds)
    ridge = _ridge_predictions(matrix, observed, groups, folds, alpha_grid)
    policies = {
        "uniform_panel": np.mean(matrix, axis=1),
        "crossfit_best_single": best_single,
        "crossfit_linear_span": ridge,
    }
    metrics = []
    for policy, prediction in policies.items():
        metrics.append({"policy": policy, **selection_metrics(observed, prediction)})
    return {"eligible_predictors": len(eligible), **predictor_geometry(matrix)}, metrics


def _read_assay(archive: ZipFile, assay_id: str) -> tuple[pd.DataFrame, list[str]]:
    filename = f"{assay_id}.csv"
    if filename not in archive.namelist():
        raise ValueError(f"ProteinGym archive is missing {filename}")
    with archive.open(filename) as handle:
        frame = pd.read_csv(handle, low_memory=False)
    matches = frame["mutant"].astype(str).str.extract(SINGLE_MUTATION, expand=False)
    frame = frame.loc[matches.notna()].copy()
    frame["position"] = matches.loc[matches.notna()].astype(int)
    frame["DMS_score"] = pd.to_numeric(frame["DMS_score"], errors="coerce")
    predictors = [column for column in frame.columns if column not in NON_PREDICTOR_COLUMNS | {"position"}]
    for column in predictors:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[np.isfinite(frame["DMS_score"])].reset_index(drop=True)
    return frame[["mutant", "position", "DMS_score", *predictors]], predictors


def _bootstrap_task_differences(
    task_metrics: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    wide = task_metrics.pivot(
        index=["protein_id", "assay_id"], columns="policy", values=["spearman", "selection_gain_sd", "top_recall"]
    )
    proteins = wide.index.get_level_values("protein_id").unique().to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ("spearman", "selection_gain_sd", "top_recall"):
        differences = []
        for _ in range(repeats):
            sampled = rng.choice(proteins, size=len(proteins), replace=True)
            values = []
            for protein in sampled:
                block = wide.loc[protein]
                values.extend(
                    (
                        block[(metric, "crossfit_linear_span")]
                        - block[(metric, "crossfit_best_single")]
                    ).tolist()
                )
            differences.append(float(np.mean(values)))
        observed = float(
            np.mean(
                wide[(metric, "crossfit_linear_span")]
                - wide[(metric, "crossfit_best_single")]
            )
        )
        rows.append(
            {
                "metric": metric,
                "linear_minus_best_single": observed,
                "bootstrap_ci_low": float(np.quantile(differences, 0.025)),
                "bootstrap_ci_high": float(np.quantile(differences, 0.975)),
                "bootstrap_unit": "protein_pair",
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    archive_path = root / config["score_archive"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    with ZipFile(archive_path) as archive:
        for pair in config["pairs"]:
            left, left_predictors = _read_assay(archive, pair["assay_a"])
            right, right_predictors = _read_assay(archive, pair["assay_b"])
            common_predictors = [column for column in left_predictors if column in right_predictors]
            shared = left.merge(right, on="mutant", suffixes=("_a", "_b"))
            predictor_agreements = [
                safe_spearman(
                    shared[f"{column}_a"].to_numpy(dtype=float),
                    shared[f"{column}_b"].to_numpy(dtype=float),
                )
                for column in common_predictors
            ]
            pair_rows.append(
                {
                    **pair,
                    "shared_variants": len(shared),
                    "outcome_spearman": safe_spearman(
                        shared["DMS_score_a"].to_numpy(dtype=float),
                        shared["DMS_score_b"].to_numpy(dtype=float),
                    ),
                    "median_predictor_agreement": float(np.nanmedian(predictor_agreements)),
                    "minimum_predictor_agreement": float(np.nanmin(predictor_agreements)),
                    "common_predictors": len(common_predictors),
                }
            )
            for suffix, assay_key, phenotype_key, frame in (
                ("a", "assay_a", "phenotype_a", left),
                ("b", "assay_b", "phenotype_b", right),
            ):
                geometry, metrics = evaluate_task(
                    frame,
                    common_predictors,
                    minimum_predictor_coverage=float(config["minimum_predictor_coverage"]),
                    outer_folds=int(config["outer_folds"]),
                    alpha_grid=[float(value) for value in config["alpha_grid"]],
                )
                for metric_row in metrics:
                    task_rows.append(
                        {
                            "protein_id": pair["protein_id"],
                            "assay_id": pair[assay_key],
                            "phenotype": pair[phenotype_key],
                            "pair_side": suffix,
                            "variants": len(frame),
                            **geometry,
                            **metric_row,
                        }
                    )

    pairs = pd.DataFrame(pair_rows).sort_values("protein_id").reset_index(drop=True)
    tasks = pd.DataFrame(task_rows).sort_values(["protein_id", "assay_id", "policy"])
    bootstrap = _bootstrap_task_differences(
        tasks, repeats=int(config["bootstrap_repeats"]), seed=int(config["seed"])
    )
    pairs.to_csv(output_dir / "paired-phenotype-audit.csv", index=False)
    tasks.to_csv(output_dir / "task-panel-recoverability.csv", index=False)
    bootstrap.to_csv(output_dir / "protein-pair-bootstrap.csv", index=False)

    linear = tasks.loc[tasks["policy"].eq("crossfit_linear_span")]
    best = tasks.loc[tasks["policy"].eq("crossfit_best_single")]
    summary = {
        "status": "development_only_exploratory",
        "protein_pairs": len(pairs),
        "tasks": tasks["assay_id"].nunique(),
        "predictors": int(pairs["common_predictors"].median()),
        "shared_variants": int(pairs["shared_variants"].sum()),
        "median_paired_outcome_spearman": float(pairs["outcome_spearman"].median()),
        "median_predictor_agreement": float(pairs["median_predictor_agreement"].median()),
        "median_effective_rank": float(linear["effective_rank"].median()),
        "median_pcs_90pct": float(linear["pcs_90pct"].median()),
        "mean_crossfit_linear_spearman": float(linear["spearman"].mean()),
        "mean_crossfit_best_single_spearman": float(best["spearman"].mean()),
        "mean_crossfit_linear_top_recall": float(linear["top_recall"].mean()),
        "mean_crossfit_best_single_top_recall": float(best["top_recall"].mean()),
        "confirmation_artifacts_accessed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "config": str(config_path.relative_to(root)),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "score_archive": str(archive_path.relative_to(root)),
        "score_archive_sha256": digest.hexdigest(),
        "protected_confirmation_inputs": [],
        "outputs": {},
    }
    for path in sorted(output_dir.glob("*")):
        if path.name == "manifest.json":
            continue
        manifest["outputs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
