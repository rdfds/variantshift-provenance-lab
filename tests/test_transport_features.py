from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from variantshift.transport_features import build_proteingym_transport_features


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_proteingym_transport_builder_uses_scores_but_not_outcomes(tmp_path: Path) -> None:
    assays = ["A1", "A2"]
    models = [
        ("esm2_650m", "ESM2_650M", "single_sequence"),
        ("gemme", "GEMME", "msa"),
    ]
    runs = []
    for assay, protein in zip(assays, ["P1", "P2"], strict=True):
        for model, _column, modality in models:
            runs.append(
                {
                    "assay_id": assay,
                    "uniprot_id": protein,
                    "model": model,
                    "modality": modality,
                    "selection_gain_sd": 0.5,
                    "taxon": "Human",
                    "coarse_selection_type": "Activity",
                }
            )
    runs_path = _write_csv(tmp_path / "runs.csv", runs)
    eligibility = _write_csv(
        tmp_path / "eligibility.csv",
        [
            {
                "assay_id": assay,
                "protein_length": 4,
                "mutated_positions": 4,
                "single_variants": 8,
            }
            for assay in assays
        ],
    )
    reference = _write_csv(
        tmp_path / "reference.csv",
        [
            {
                "DMS_id": assay,
                "DMS_filename": f"{assay}.csv",
                "MSA_num_seqs": 100,
                "MSA_N_eff": 10,
                "MSA_perc_cov": 0.9,
                "pdb_file": "x.pdb",
            }
            for assay in assays
        ],
    )
    families = _write_csv(
        tmp_path / "families.csv",
        [
            {"uniprot_id": "P1", "family_id": "F1", "family_size": 1},
            {"uniprot_id": "P2", "family_id": "F2", "family_size": 1},
        ],
    )
    sequence = _write_csv(
        tmp_path / "sequence.csv",
        [
            {
                "query_uniprot_id": "P1",
                "target_uniprot_id": "P2",
                "sequence_identity": 0.25,
            }
        ],
    )
    structure = _write_csv(
        tmp_path / "structure.csv",
        [
            {
                "protein_a": "P1",
                "protein_b": "P2",
                "reciprocal_minimum_tm_score": 0.4,
            }
        ],
    )
    domains = _write_csv(
        tmp_path / "domains.csv",
        [
            {
                "assay_id": assay,
                "overlap_fraction_of_shorter": 0.8,
                "qualifies_curated_domain": True,
            }
            for assay in assays
        ],
    )
    score_archive = tmp_path / "scores.zip"
    with ZipFile(score_archive, "w") as archive:
        for assay in assays:
            score_path = tmp_path / f"{assay}.csv"
            pd.DataFrame(
                {
                    "mutant": [f"A{i}V" for i in range(1, 61)],
                    "DMS_score": [999.0] * 60,
                    "ESM2_650M": list(range(60)),
                    "GEMME": list(reversed(range(60))),
                }
            ).to_csv(score_path, index=False)
            archive.write(score_path, arcname=f"{assay}.csv")
    output = tmp_path / "features.csv"
    summary = build_proteingym_transport_features(
        runs_path,
        eligibility,
        reference,
        families,
        sequence,
        structure,
        domains,
        score_archive,
        output,
    )
    frame = pd.read_csv(output)
    assert summary["outcome_columns_read_from_prediction_archive"] == []
    assert len(frame) == 4
    assert frame["score_dispersion"].gt(0).all()
    assert frame.loc[
        frame["protein_id"].eq("P1"), "sequence_identity_to_development"
    ].eq(0.25).all()
    assert "DMS_score" not in frame
