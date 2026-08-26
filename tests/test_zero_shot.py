from zipfile import ZipFile

import numpy as np
import pandas as pd

from variantshift.zero_shot import (
    run_zero_shot_benchmark,
    summarize_zero_shot,
    zero_shot_subset_differences,
)


def _write_archives(tmp_path):
    assay_id = "TEST_HUMAN_Author_2024"
    filename = f"{assay_id}.csv"
    metadata = pd.DataFrame(
        [
            {
                "DMS_id": assay_id,
                "DMS_filename": filename,
                "UniProt_ID": "TEST_HUMAN",
                "target_seq": "A" * 30,
                "taxon": "Human",
                "selection_type": "Binding",
                "coarse_selection_type": "Activity",
                "ProteinGym_version": 1.3,
            }
        ]
    )
    reference_path = tmp_path / "reference.csv"
    metadata.to_csv(reference_path, index=False)

    rows = []
    for position in range(1, 31):
        for alternate, offset in (("C", 0.0), ("D", 0.2)):
            target = position / 30 + offset
            sequence = list("A" * 30)
            sequence[position - 1] = alternate
            rows.append(
                {
                    "mutant": f"A{position}{alternate}",
                    "mutated_sequence": "".join(sequence),
                    "DMS_score": target,
                    "ESM2_8M": target + (position % 3) * 0.01,
                    "ESM2_650M": target + (position % 2) * 0.005,
                }
            )
    scores = pd.DataFrame(rows)
    source_path = tmp_path / "source.csv"
    scores[["mutant", "mutated_sequence", "DMS_score"]].to_csv(source_path, index=False)
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path, index=False)

    source_archive = tmp_path / "source.zip"
    with ZipFile(source_archive, "w") as archive:
        archive.write(source_path, f"DMS_ProteinGym_substitutions/{filename}")
    score_archive = tmp_path / "scores.zip"
    with ZipFile(score_archive, "w") as archive:
        archive.write(score_path, filename)
    eligibility = pd.DataFrame([{"assay_id": assay_id, "eligible": True}])
    return source_archive, score_archive, reference_path, eligibility


def test_zero_shot_scores_are_joined_audited_and_labeled_as_fixed(tmp_path):
    source, scores, reference, eligibility = _write_archives(tmp_path)
    runs, audit = run_zero_shot_benchmark(
        source,
        scores,
        reference,
        eligibility,
        model_columns=("ESM2_8M", "ESM2_650M"),
        repeats=2,
    )

    assert audit["eligible_for_zero_shot"].item()
    assert audit["source_single_variants"].item() == 60
    assert audit["common_score_coverage"].item() == 1.0
    assert audit["dms_score_max_abs_difference"].item() < 1e-12
    assert runs["evaluation_type"].eq("zero_shot_fixed_scores").all()
    assert runs["exact_variant_overlap"].eq(0).all()
    assert runs.loc[
        runs["split"].eq("position_holdout"), "shared_position_count"
    ].eq(0).all()

    differences = zero_shot_subset_differences(runs)
    assert np.allclose(
        differences["subset_spearman_difference"],
        differences["random_spearman"] - differences["position_spearman"],
    )
    summary = summarize_zero_shot(differences, bootstrap_repeats=100)
    assert set(summary["model"]) == {"ESM2_8M", "ESM2_650M"}
    assert summary["evaluation_type"].eq("zero_shot_fixed_scores").all()
