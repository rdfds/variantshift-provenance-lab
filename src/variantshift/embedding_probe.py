"""Local ESM-2 residue-embedding probe under four biological split protocols."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import interval_suite
from .esm_embeddings import load_cached_embedding
from .features import HYDROPATHY, biophysical_matrix
from .metrics import (
    interval_metrics,
    position_conditional_coverage,
    regression_metrics,
    risk_coverage_curve,
    top_selection_metrics,
)
from .proteingym import (
    _archive_members,
    canonicalize_assay,
    read_assay_member,
    read_reference_index,
)
from .splits import (
    contiguous_position_split,
    leakage_audit,
    modulo_position_split,
    position_holdout_split,
    random_variant_split,
)

AMINO_ACIDS = tuple(sorted(HYDROPATHY))
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}

_PROBE_ARCHIVE: ZipFile | None = None
_PROBE_MEMBERS: dict[str, str] | None = None
_PROBE_REFERENCE: pd.DataFrame | None = None


def embedding_probe_matrix(
    codes: Sequence[str],
    target_sequence: str,
    residue_embedding: np.ndarray,
) -> np.ndarray:
    """Combine an unsupervised WT position embedding with mutation descriptors."""
    from .mutations import parse_variant

    if residue_embedding.shape[0] != len(target_sequence):
        raise ValueError("Residue embedding and target sequence lengths differ")
    codes = list(map(str, codes))
    biochemical = biophysical_matrix(codes)
    representation = np.zeros(
        (len(codes), residue_embedding.shape[1] + 2 * len(AMINO_ACIDS) + 1),
        dtype=np.float32,
    )
    for row, code in enumerate(codes):
        mutations = parse_variant(code)
        if len(mutations) != 1:
            raise ValueError("The ESM-2 residue probe supports single substitutions only")
        mutation = mutations[0]
        position = mutation.position - 1
        if target_sequence[position] != mutation.reference:
            raise ValueError(f"Reference mismatch for {code}")
        representation[row, : residue_embedding.shape[1]] = residue_embedding[position]
        offset = residue_embedding.shape[1]
        representation[row, offset + AA_INDEX[mutation.reference]] = 1.0
        representation[
            row, offset + len(AMINO_ACIDS) + AA_INDEX[mutation.alternate]
        ] = 1.0
        representation[row, -1] = mutation.position / len(target_sequence)
    return np.concatenate([representation, biochemical.astype(np.float32)], axis=1)


def _split_suite(frame: pd.DataFrame, *, seed: int, fold: int):
    return (
        random_variant_split(frame, seed=seed),
        position_holdout_split(frame, seed=seed),
        modulo_position_split(frame, fold=fold),
        contiguous_position_split(frame, fold=fold),
    )


def evaluate_embedding_probe_assay(
    frame: pd.DataFrame,
    target_sequence: str,
    residue_embedding: np.ndarray,
    *,
    seeds: Sequence[int],
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    alpha: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one assay and return interval summaries plus risk-coverage curves."""
    codes = frame["mutation_codes"].astype(str).tolist()
    features = embedding_probe_matrix(codes, target_sequence, residue_embedding)
    target = frame["DMS_score"].to_numpy(dtype=float)
    metric_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    for replicate, seed in enumerate(seeds):
        fold = replicate % 5
        for split in _split_suite(frame, seed=int(seed), fold=fold):
            audit = leakage_audit(frame, split)
            fit_indices, calibration_indices = train_test_split(
                split.train_indices,
                test_size=calibration_fraction,
                random_state=int(seed),
            )
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=alpha, solver="lsqr"),
            )
            model.fit(features[fit_indices], target[fit_indices])
            calibration_prediction = np.asarray(
                model.predict(features[calibration_indices]), dtype=float
            )
            test_prediction = np.asarray(
                model.predict(features[split.test_indices]), dtype=float
            )
            test_target = target[split.test_indices]
            point_metrics = regression_metrics(test_target, test_prediction).to_dict()
            selection_metrics = top_selection_metrics(test_target, test_prediction)
            fit_codes = [codes[index] for index in fit_indices]
            calibration_codes = [codes[index] for index in calibration_indices]
            test_codes = [codes[index] for index in split.test_indices]
            target_scale = float(np.std(target[fit_indices]))
            intervals = interval_suite(
                fit_codes,
                calibration_codes,
                target[calibration_indices],
                calibration_prediction,
                test_codes,
                test_prediction,
                coverage=coverage,
            )
            for interval in intervals:
                interval_summary = interval_metrics(
                    test_target, interval.lower, interval.upper
                )
                position_summary = position_conditional_coverage(
                    test_codes,
                    test_target,
                    interval.lower,
                    interval.upper,
                )
                base = {
                    "assay_id": str(frame["assay_id"].iat[0]),
                    "uniprot_id": str(frame["uniprot_id"].iat[0]),
                    "taxon": str(frame["taxon"].iat[0]),
                    "coarse_selection_type": str(
                        frame["coarse_selection_type"].iat[0]
                    ),
                    "seed": int(seed),
                    "fold": fold,
                    "split": split.name,
                    "model": "esm2_residue_ridge_probe",
                    "calibration_method": interval.method,
                    "nominal_coverage": coverage,
                    **point_metrics,
                    **selection_metrics,
                    **interval_summary,
                    **position_summary,
                    "normalized_mean_interval_width": (
                        interval_summary["mean_interval_width"] / target_scale
                        if target_scale > 1e-12
                        else np.nan
                    ),
                    "fit_rows": len(fit_indices),
                    "calibration_rows": len(calibration_indices),
                    "test_rows": len(split.test_indices),
                    "excluded_rows": audit["excluded_rows"],
                    "exact_variant_overlap": audit["exact_variant_overlap"],
                    "shared_position_count": audit["shared_position_count"],
                }
                metric_rows.append(base)
                for risk in risk_coverage_curve(
                    test_target, test_prediction, interval.uncertainty
                ):
                    risk_rows.append(
                        {
                            key: base[key]
                            for key in (
                                "assay_id",
                                "uniprot_id",
                                "seed",
                                "fold",
                                "split",
                                "model",
                                "calibration_method",
                            )
                        }
                        | risk
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(risk_rows)


def _initialize_probe_worker(archive_path: Path, reference_path: Path) -> None:
    global _PROBE_ARCHIVE, _PROBE_MEMBERS, _PROBE_REFERENCE
    _PROBE_ARCHIVE = ZipFile(archive_path)
    _PROBE_MEMBERS = _archive_members(_PROBE_ARCHIVE)
    _PROBE_REFERENCE = read_reference_index(reference_path).set_index("DMS_id", drop=False)


def _probe_worker(arguments):
    assay_id, embedding_path, seeds, calibration_fraction, coverage, alpha = arguments
    if _PROBE_ARCHIVE is None or _PROBE_MEMBERS is None or _PROBE_REFERENCE is None:
        raise RuntimeError("Embedding-probe worker was not initialized")
    metadata = _PROBE_REFERENCE.loc[assay_id]
    frame = canonicalize_assay(
        read_assay_member(
            _PROBE_ARCHIVE,
            _PROBE_MEMBERS[str(metadata["DMS_filename"])],
        ),
        metadata,
    )
    sequence = str(metadata["target_seq"])
    embedding = load_cached_embedding(Path(embedding_path), sequence)
    return evaluate_embedding_probe_assay(
        frame,
        sequence,
        embedding,
        seeds=seeds,
        calibration_fraction=calibration_fraction,
        coverage=coverage,
        alpha=alpha,
    )


def run_embedding_probe_benchmark(
    archive_path: Path,
    reference_path: Path,
    eligibility: pd.DataFrame,
    embedding_index: pd.DataFrame,
    *,
    start_seed: int = 42,
    repeats: int = 5,
    calibration_fraction: float = 0.2,
    coverage: float = 0.8,
    alpha: float = 100.0,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the local ESM-2 ridge probe over every eligible assay."""
    if repeats < 1:
        raise ValueError("At least one repeat is required")
    if workers < 1:
        raise ValueError("Worker count must be at least one")
    eligible_ids = eligibility.loc[
        eligibility["eligible"].astype(bool), "assay_id"
    ].astype(str)
    index = embedding_index.set_index("assay_id", drop=False)
    missing = sorted(set(eligible_ids).difference(index.index.astype(str)))
    if missing:
        raise ValueError(f"Embedding index is missing {len(missing)} eligible assays")
    seeds = tuple(range(start_seed, start_seed + repeats))
    jobs = [
        (
            assay_id,
            str(index.loc[assay_id, "embedding_path"]),
            seeds,
            calibration_fraction,
            coverage,
            alpha,
        )
        for assay_id in eligible_ids
    ]
    if workers == 1:
        _initialize_probe_worker(Path(archive_path), Path(reference_path))
        evaluated = [_probe_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_probe_worker,
            initargs=(Path(archive_path), Path(reference_path)),
        ) as executor:
            evaluated = list(executor.map(_probe_worker, jobs, chunksize=1))
    metric_frames, risk_frames = zip(*evaluated, strict=True)
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["model", "calibration_method", "split", "assay_id", "seed"]
    ).reset_index(drop=True)
    risks = pd.concat(risk_frames, ignore_index=True).sort_values(
        [
            "model",
            "calibration_method",
            "split",
            "assay_id",
            "seed",
            "retained_fraction",
        ]
    ).reset_index(drop=True)
    return metrics, risks


def summarize_embedding_probe(metrics: pd.DataFrame) -> pd.DataFrame:
    """Give proteins equal weight when summarizing probe and calibration behavior."""
    value_columns = [
        "spearman",
        "top_recall",
        "selection_gain_sd",
        "observed_coverage",
        "position_coverage_mean",
        "position_coverage_p10",
        "normalized_mean_interval_width",
    ]
    assay_level = metrics.groupby(
        [
            "model",
            "calibration_method",
            "split",
            "uniprot_id",
            "assay_id",
        ],
        as_index=False,
    )[value_columns].mean()
    protein_level = assay_level.groupby(
        ["model", "calibration_method", "split", "uniprot_id"], as_index=False
    )[value_columns].mean()
    summary = protein_level.groupby(
        ["model", "calibration_method", "split"], as_index=False
    ).agg(
        n_proteins=("uniprot_id", "nunique"),
        **{f"mean_{column}": (column, "mean") for column in value_columns},
    )
    assay_counts = assay_level.groupby(
        ["model", "calibration_method", "split"]
    )["assay_id"].nunique()
    summary["n_assays"] = [
        int(assay_counts.loc[(row.model, row.calibration_method, row.split)])
        for row in summary.itertuples(index=False)
    ]
    return summary.sort_values(["split", "calibration_method"]).reset_index(drop=True)


def summarize_probe_risk_coverage(risks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selective risk without treating assays or seeds as independent proteins."""
    assay_level = risks.groupby(
        [
            "model",
            "calibration_method",
            "split",
            "retained_fraction",
            "uniprot_id",
            "assay_id",
        ],
        as_index=False,
    ).agg(normalized_mae=("normalized_mae", "mean"))
    protein_level = assay_level.groupby(
        [
            "model",
            "calibration_method",
            "split",
            "retained_fraction",
            "uniprot_id",
        ],
        as_index=False,
    ).agg(normalized_mae=("normalized_mae", "mean"))
    return (
        protein_level.groupby(
            ["model", "calibration_method", "split", "retained_fraction"],
            as_index=False,
        )
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_normalized_mae=("normalized_mae", "mean"),
        )
        .sort_values(["split", "calibration_method", "retained_fraction"])
        .reset_index(drop=True)
    )
