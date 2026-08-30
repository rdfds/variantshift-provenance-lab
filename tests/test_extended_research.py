import numpy as np
import pandas as pd

from variantshift.cross_protein import (
    CrossProteinDataset,
    compare_holdout_protocols,
    evaluate_held_out_families,
    evaluate_held_out_proteins,
    summarize_grouped_repeat_estimates,
    summarize_held_out_proteins,
)
from variantshift.crossover import evaluate_crossover_predictor
from variantshift.esm_embeddings import _window_ranges


def test_long_sequence_windows_cover_every_residue():
    ranges = _window_ranges(2_500, window_size=1_022, overlap=128)
    covered = np.zeros(2_500, dtype=int)
    for start, end in ranges:
        covered[start:end] += 1
    assert covered.min() >= 1
    assert ranges[-1][1] == 2_500


def test_held_out_protein_evaluation_never_mixes_protein_groups():
    rng = np.random.default_rng(4)
    rows = []
    features = []
    targets = []
    for protein in range(8):
        for position in range(1, 21):
            feature = rng.normal(size=6)
            target = 0.3 * feature[0] - 0.1 * feature[1] + protein * 0.01
            rows.append(
                {
                    "assay_id": f"A{protein}",
                    "uniprot_id": f"P{protein}",
                    "mutation_codes": f"A{position}C",
                    "taxon": "Synthetic",
                    "coarse_selection_type": "Activity",
                    "experimental_score": target,
                }
            )
            features.append(feature)
            targets.append(target)
    dataset = CrossProteinDataset(
        metadata=pd.DataFrame(rows),
        features=np.asarray(features),
        targets=np.asarray(targets),
        feature_names=tuple(f"x{i}" for i in range(6)),
    )
    assays, risks, predictions = evaluate_held_out_proteins(dataset, folds=3, repeats=2)
    assert set(assays["model"]) == {"cross_protein_ridge", "cross_protein_histgb"}
    assert assays["test_proteins"].min() >= 2
    assert predictions.groupby(["model", "repeat", "uniprot_id"])["fold"].nunique().max() == 1
    assert assays["repeat"].nunique() == 2
    repeat_estimates = summarize_grouped_repeat_estimates(assays)
    assert repeat_estimates["repeat"].nunique() == 2
    assert repeat_estimates["n_proteins"].eq(8).all()
    assert not risks.empty
    assert summarize_held_out_proteins(assays)["n_proteins"].eq(8).all()

    families = pd.DataFrame(
        {
            "uniprot_id": [f"P{protein}" for protein in range(8)],
            "family_id": ["F01", "F01", "F23", "F23", "F4", "F5", "F6", "F7"],
        }
    )
    family_assays, _, family_predictions = evaluate_held_out_families(
        dataset, families, folds=3, repeats=2
    )
    assert family_assays["evaluation_type"].eq("held_out_sequence_family").all()
    assert family_assays["test_families"].min() >= 1
    assert family_predictions.groupby(["model", "repeat", "family_id"])["fold"].nunique().max() == 1
    protein_folds = family_predictions.groupby(["model", "repeat", "uniprot_id"])["fold"].first()
    assert (
        protein_folds.loc[("cross_protein_ridge", 0, "P0")]
        == protein_folds.loc[("cross_protein_ridge", 0, "P1")]
    )
    comparison = compare_holdout_protocols(assays, family_assays, bootstrap_repeats=100)
    assert len(comparison) == 20
    assert comparison["n_proteins"].eq(8).all()
    assert comparison["n_bootstrap_groups"].eq(6).all()
    assert comparison["bootstrap_unit"].eq("heldout_family:family_id").all()
    assert comparison["delta_ci_low"].notna().all()
    structure_assays, _, _ = evaluate_held_out_families(
        dataset,
        families,
        folds=3,
        repeats=2,
        group_name="sequence_structure_family",
        evaluation_type="held_out_sequence_structure_family",
    )
    assert structure_assays["evaluation_type"].eq("held_out_sequence_structure_family").all()
    structure_comparison = compare_holdout_protocols(
        family_assays,
        structure_assays,
        bootstrap_repeats=100,
        baseline_label="heldout_sequence_family",
        alternative_label="heldout_structure_family",
    )
    assert "heldout_structure_family_minus_heldout_sequence_family" in (
        structure_comparison.columns
    )

    ablation_dataset = CrossProteinDataset(
        metadata=dataset.metadata,
        features=dataset.features,
        targets=dataset.targets,
        feature_names=dataset.feature_names,
        score_feature_count=2,
    )
    ablation_assays, _, _ = evaluate_held_out_families(
        ablation_dataset,
        families,
        folds=3,
        include_feature_ablation=True,
    )
    assert set(ablation_assays["feature_set"]) == {
        "mutation_descriptors_only",
        "mutation_descriptors_plus_fixed_esm_scores",
    }
    assert set(ablation_assays["model"]) == {
        "cross_protein_ridge",
        "cross_protein_histgb",
        "cross_protein_ridge_mutation_only",
        "cross_protein_histgb_mutation_only",
    }


def test_crossover_validation_holds_out_complete_proteins():
    rows = []
    for protein in range(12):
        for seed in (42, 43):
            rows.append(
                {
                    "assay_id": f"A{protein}",
                    "uniprot_id": f"P{protein}",
                    "seed": seed,
                    "taxon": "Human" if protein % 2 else "Prokaryote",
                    "coarse_selection_type": "Activity",
                    "train_rows": 500 + protein,
                    "target_std": 0.5 + protein / 20,
                    "zero_score_train_spearman": protein / 12,
                    "supervised_test_spearman": 0.4,
                    "zero_shot_test_spearman": 0.5,
                    "supervised_advantage": -0.1 if protein % 2 else 0.1,
                    "supervised_wins": int(protein % 2 == 0),
                }
            )
    predictions, summary, coefficients = evaluate_crossover_predictor(pd.DataFrame(rows), folds=4)
    assert predictions.groupby(["model", "uniprot_id"])["fold"].nunique().max() == 1
    assert summary["n_proteins"].eq(12).all()
    assert set(summary["model"]) == {"histgb", "logistic"}
    assert not coefficients.empty
