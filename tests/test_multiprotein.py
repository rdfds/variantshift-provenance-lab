import numpy as np
import pandas as pd

from variantshift.models import MeanBaseline
from variantshift.multiprotein import (
    evaluate_proteingym_assay,
    multiprotein_gaps,
    summarize_assays,
    summarize_multiprotein_gaps,
)


def assay_frame() -> pd.DataFrame:
    rows = []
    for position in range(1, 31):
        for offset, alternate in enumerate("CDE"):
            rows.append(
                {
                    "mutation_codes": f"A{position}{alternate}",
                    "DMS_score": position / 30 + offset * 0.1,
                    "assay_id": "ASSAY_1",
                    "uniprot_id": "PROTEIN_1",
                    "taxon": "Human",
                    "coarse_selection_type": "Activity",
                }
            )
    return pd.DataFrame(rows)


def test_assay_evaluation_pairs_leakage_safe_splits():
    runs = evaluate_proteingym_assay(
        assay_frame(),
        seeds=(7, 8),
        model_factories={"mean": MeanBaseline},
    )
    assert len(runs) == 4
    assert runs["exact_variant_overlap"].eq(0).all()
    assert runs.loc[
        runs["split"].eq("position_holdout"), "shared_position_count"
    ].eq(0).all()
    assert np.isfinite(runs["normalized_rmse"]).all()

    gaps = multiprotein_gaps(runs)
    assert len(gaps) == 2
    assert gaps["spearman_gap"].eq(
        gaps["random_spearman"] - gaps["position_spearman"]
    ).all()
    assays = summarize_assays(gaps)
    assert assays["n_seeds"].item() == 2


def test_summary_bootstraps_proteins_after_assays_and_seeds():
    rows = []
    for protein, assay, gap in [
        ("P1", "A1", 0.1),
        ("P1", "A2", 0.3),
        ("P2", "A3", 0.5),
    ]:
        for seed in (42, 43):
            rows.append(
                {
                    "assay_id": assay,
                    "uniprot_id": protein,
                    "seed": seed,
                    "model": "ridge",
                    "random_spearman": 0.7,
                    "position_spearman": 0.7 - gap,
                    "spearman_gap": gap,
                    "observed_coverage_gap": 0.05,
                    "normalized_rmse_gap": -0.2,
                }
            )
    summary = summarize_multiprotein_gaps(
        pd.DataFrame(rows), bootstrap_repeats=1_000, bootstrap_seed=3
    )
    assert summary["n_assays"].item() == 3
    assert summary["n_proteins"].item() == 2
    assert summary["spearman_gap_mean"].item() == 0.35
    assert summary["bootstrap_unit"].item() == "UniProt_ID"
