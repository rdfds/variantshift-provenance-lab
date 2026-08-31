import pandas as pd
import pytest

from variantshift.schemas import (
    all_single_substitutions,
    sequence_sha256,
    stable_frame_sha256,
    validate_targets,
)


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_id": ["panel"],
            "target_id": ["T1"],
            "protein_id": ["P1"],
            "sequence": ["AC"],
            "sequence_sha256": [sequence_sha256("AC")],
            "sequence_length": [2],
        }
    )


def test_complete_substitution_universe_has_19_per_position() -> None:
    variants = all_single_substitutions(_targets())
    assert len(variants) == 38
    assert variants.groupby("position").size().eq(19).all()
    assert not variants["reference"].eq(variants["alternate"]).any()


def test_target_digest_and_table_digest_are_enforced() -> None:
    targets = _targets()
    validate_targets(targets)
    altered = targets.assign(sequence_sha256="incorrect")
    with pytest.raises(ValueError, match="sequence_sha256"):
        validate_targets(altered)
    assert stable_frame_sha256(targets) == stable_frame_sha256(
        targets.loc[::-1, list(reversed(targets.columns))]
    )
