import pandas as pd

from variantshift.splits import (
    contiguous_position_split,
    leakage_audit,
    modulo_position_split,
    mutation_depth_split,
    position_holdout_split,
    random_variant_split,
)


def split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mutation_codes": [
                "A1C", "C2D", "D3E", "E4F", "F5G",
                "A1C/C2D", "D3E/E4F", "C2D/F5G",
            ],
            "goi_amino_mutations": [1, 1, 1, 1, 1, 2, 2, 2],
        }
    )


def test_random_split_keeps_identical_variants_together() -> None:
    frame = pd.concat([split_frame(), split_frame().iloc[[0]]], ignore_index=True)
    split = random_variant_split(frame, test_size=0.3)
    audit = leakage_audit(frame, split)
    assert audit["exact_variant_overlap"] == 0


def test_position_holdout_has_no_residue_overlap() -> None:
    frame = split_frame()
    split = position_holdout_split(frame, position_fraction=0.4)
    audit = leakage_audit(frame, split)
    assert audit["shared_position_count"] == 0


def test_proteingym_style_position_folds_are_leakage_free() -> None:
    frame = pd.DataFrame(
        {
            "mutation_codes": [
                f"A{position}{alternate}"
                for position in range(1, 31)
                for alternate in "CD"
            ]
        }
    )
    modulo = modulo_position_split(frame, fold=2, n_folds=5)
    contiguous = contiguous_position_split(frame, fold=2, n_folds=5)
    assert leakage_audit(frame, modulo)["shared_position_count"] == 0
    assert leakage_audit(frame, contiguous)["shared_position_count"] == 0
    assert modulo.metadata["held_out_positions"] == [2, 7, 12, 17, 22, 27]
    assert contiguous.metadata["held_out_positions"] == list(range(13, 19))


def test_depth_split_extrapolates_from_singles_to_multiples() -> None:
    frame = split_frame()
    split = mutation_depth_split(frame)
    assert set(frame.iloc[split.train_indices]["goi_amino_mutations"]) == {1}
    assert set(frame.iloc[split.test_indices]["goi_amino_mutations"]) == {2}
