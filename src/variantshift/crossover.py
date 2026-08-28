"""Predict when supervised mutation models will beat fixed zero-shot scores."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import regression_metrics, top_selection_metrics
from .mutations import parse_variant
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    read_assay_member,
    read_reference_index,
)
from .splits import position_holdout_split
from .zero_shot import _align_scores, _read_score_member


def _training_properties(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    protein_length: int,
    zero_score_column: str,
) -> dict[str, float]:
    train = frame.iloc[train_indices].copy()
    positions = np.asarray(
        [parse_variant(code)[0].position for code in train["mutation_codes"].astype(str)],
        dtype=int,
    )
    target = train["DMS_score"].to_numpy(dtype=float)
    zero_score = train[zero_score_column].to_numpy(dtype=float)
    grouped = pd.DataFrame({"position": positions, "target": target}).groupby("position")
    site_means = grouped["target"].mean().to_numpy(dtype=float)
    site_stds = grouped["target"].std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    variants_per_position = grouped.size().to_numpy(dtype=float)
    target_std = float(np.std(target))
    rvsm = float(np.std(site_means) / target_std) if target_std > 1e-12 else 0.0
    fhvs = (
        float(np.mean(site_stds / target_std > 0.7)) if target_std > 1e-12 else 0.0
    )
    zero_metrics = regression_metrics(target, zero_score)
    selection = top_selection_metrics(target, zero_score)
    return {
        "train_rows": float(len(train)),
        "train_positions": float(len(site_means)),
        "mutation_density": float(len(train) / max(1, len(site_means) * 19)),
        "variants_per_position_mean": float(np.mean(variants_per_position)),
        "variants_per_position_cv": float(
            np.std(variants_per_position) / max(np.mean(variants_per_position), 1e-12)
        ),
        "target_std": target_std,
        "target_iqr": float(np.quantile(target, 0.75) - np.quantile(target, 0.25)),
        "target_skew": float(skew(target, bias=False)) if len(target) > 2 else 0.0,
        "relative_variability_site_means": rvsm,
        "fraction_highly_variable_sites": fhvs,
        "mean_within_site_sd_scaled": float(np.mean(site_stds) / target_std)
        if target_std > 1e-12
        else 0.0,
        "zero_score_train_spearman": zero_metrics.spearman,
        "zero_score_train_top_recall": selection["top_recall"],
        "zero_score_std": float(np.std(zero_score)),
        "log_protein_length": float(np.log1p(protein_length)),
    }


def build_crossover_examples(
    source_archive_path: Path,
    score_archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    supervised_runs: pd.DataFrame,
    zero_shot_runs: pd.DataFrame,
    *,
    supervised_model: str,
    zero_shot_model: str = "ESM2_650M",
) -> pd.DataFrame:
    """Construct split-specific, training-only properties and win/loss outcomes."""
    supervised = supervised_runs.loc[
        supervised_runs["model"].eq(supervised_model)
        & supervised_runs["split"].eq("position_holdout")
    ].copy()
    if "calibration_method" in supervised.columns:
        supervised = supervised.loc[
            supervised["calibration_method"].eq("standard_split")
        ]
    supervised = supervised[
        ["assay_id", "uniprot_id", "seed", "spearman", "test_rows"]
    ].rename(columns={"spearman": "supervised_test_spearman"})
    zero = zero_shot_runs.loc[
        zero_shot_runs["model"].eq(zero_shot_model)
        & zero_shot_runs["split"].eq("position_holdout")
    ][["assay_id", "uniprot_id", "seed", "spearman", "test_rows"]].rename(
        columns={
            "spearman": "zero_shot_test_spearman",
            "test_rows": "zero_shot_test_rows",
        }
    )
    paired = supervised.merge(
        zero,
        on=["assay_id", "uniprot_id", "seed"],
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("No paired supervised and zero-shot position results were found")
    reference = read_reference_index(reference_path).set_index("DMS_id", drop=False)
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    paired = paired.loc[paired["assay_id"].astype(str).isin(eligible_ids)]
    rows: list[dict[str, object]] = []
    with ZipFile(source_archive_path) as source_archive, ZipFile(
        score_archive_path
    ) as score_archive:
        source_members = _archive_members(source_archive)
        score_members = _archive_members(score_archive)
        for assay_id, assay_pairs in paired.groupby("assay_id", sort=True):
            metadata = reference.loc[str(assay_id)]
            filename = str(metadata["DMS_filename"])
            source = canonicalize_assay(
                read_assay_member(source_archive, source_members[filename]), metadata
            )
            score_frame = _read_score_member(
                score_archive, score_members[filename], (zero_shot_model,)
            )
            frame, audit = _align_scores(source, score_frame, (zero_shot_model,))
            if audit["common_score_coverage"] < 1.0:
                raise ValueError(f"Incomplete zero-shot scores for {assay_id}")
            for pair in assay_pairs.itertuples(index=False):
                split = position_holdout_split(frame, seed=int(pair.seed))
                if len(split.test_indices) != int(pair.test_rows):
                    raise ValueError(f"Supervised split reconstruction failed for {assay_id}")
                properties = _training_properties(
                    frame,
                    split.train_indices,
                    protein_length=len(str(metadata["target_seq"])),
                    zero_score_column=zero_shot_model,
                )
                delta = float(pair.supervised_test_spearman - pair.zero_shot_test_spearman)
                rows.append(
                    {
                        "assay_id": str(assay_id),
                        "uniprot_id": str(pair.uniprot_id),
                        "seed": int(pair.seed),
                        "taxon": str(metadata["taxon"]),
                        "coarse_selection_type": str(
                            metadata["coarse_selection_type"]
                        ),
                        **properties,
                        "supervised_test_spearman": float(pair.supervised_test_spearman),
                        "zero_shot_test_spearman": float(pair.zero_shot_test_spearman),
                        "supervised_advantage": delta,
                        "supervised_wins": int(delta > 0),
                    }
                )
    return pd.DataFrame(rows).sort_values(["assay_id", "seed"]).reset_index(drop=True)


def _crossover_design(examples: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    excluded = {
        "assay_id",
        "uniprot_id",
        "seed",
        "supervised_test_spearman",
        "zero_shot_test_spearman",
        "supervised_advantage",
        "supervised_wins",
    }
    features = examples.drop(columns=sorted(excluded.intersection(examples.columns)))
    features = pd.get_dummies(
        features, columns=["taxon", "coarse_selection_type"], dtype=float
    )
    return features, features.to_numpy(dtype=float)


def evaluate_crossover_predictor(
    examples: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate predictions only from folds whose proteins are absent from training."""
    feature_frame, design = _crossover_design(examples)
    target = examples["supervised_wins"].to_numpy(dtype=int)
    groups = examples["uniprot_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    model_factories = {
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2_000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "histgb": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    for model_name, factory in model_factories.items():
        probabilities = np.zeros(len(examples), dtype=float)
        fold_ids = np.full(len(examples), -1, dtype=int)
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(design, target, groups)
        ):
            model = factory()
            model.fit(design[train_indices], target[train_indices])
            probabilities[test_indices] = model.predict_proba(design[test_indices])[:, 1]
            fold_ids[test_indices] = fold
        output = examples[
            [
                "assay_id",
                "uniprot_id",
                "seed",
                "supervised_advantage",
                "supervised_wins",
            ]
        ].copy()
        output["model"] = model_name
        output["fold"] = fold_ids
        output["predicted_probability"] = probabilities
        output["predicted_supervised_win"] = (probabilities >= 0.5).astype(int)
        prediction_rows.append(output)

        if model_name == "logistic":
            final_model = factory().fit(design, target)
            coefficients = final_model.named_steps["logisticregression"].coef_[0]
            for name, coefficient in zip(feature_frame.columns, coefficients, strict=True):
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "feature": name,
                        "standardized_coefficient": float(coefficient),
                    }
                )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    summaries = []
    prevalence = float(np.mean(target))
    majority_accuracy = max(prevalence, 1 - prevalence)
    for model_name, group in predictions.groupby("model", sort=True):
        observed = group["supervised_wins"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        predicted = group["predicted_supervised_win"].to_numpy(dtype=int)
        protein_accuracy = (
            group.assign(correct=observed == predicted)
            .groupby("uniprot_id")["correct"]
            .mean()
        )
        summaries.append(
            {
                "model": model_name,
                "validation": "five_fold_held_out_UniProt_ID",
                "n_examples": len(group),
                "n_assays": int(group["assay_id"].nunique()),
                "n_proteins": int(group["uniprot_id"].nunique()),
                "supervised_win_prevalence": prevalence,
                "roc_auc": float(roc_auc_score(observed, probability)),
                "brier_score": float(brier_score_loss(observed, probability)),
                "accuracy": float(accuracy_score(observed, predicted)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(observed, predicted)
                ),
                "mean_protein_accuracy": float(protein_accuracy.mean()),
                "majority_accuracy": majority_accuracy,
            }
        )
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        "standardized_coefficient", key=np.abs, ascending=False
    )
    return (
        predictions.sort_values(["model", "fold", "assay_id", "seed"]).reset_index(
            drop=True
        ),
        pd.DataFrame(summaries).sort_values("model").reset_index(drop=True),
        coefficients.reset_index(drop=True),
    )
