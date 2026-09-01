from __future__ import annotations

from pathlib import Path

import pandas as pd

from variantshift.pilot import _canonicalize_venus, _stratified_venus_targets


def test_venus_pilot_selection_is_deterministic_and_stratified() -> None:
    targets = pd.DataFrame(
        {
            "target_id": [f"target-{index:02d}" for index in range(40)],
            "sequence_length": list(range(100, 140)),
        }
    )
    first = _stratified_venus_targets(targets, 12)
    second = _stratified_venus_targets(targets.sample(frac=1, random_state=9), 12)
    assert first == second
    assert len(first) == 12
    selected = targets.loc[targets["target_id"].isin(first)].sort_values("sequence_length")
    strata = ((selected["sequence_length"] - 100) // 10).value_counts().to_dict()
    assert strata == {0: 3, 1: 3, 2: 3, 3: 3}


def test_venus_parser_uses_frozen_direction_and_reference_sequence() -> None:
    frame = pd.DataFrame(
        {
            "mutation": ["A1V", "C2D", "X3A", "A1V"],
            "activity": [2.0, 4.0, 100.0, 6.0],
            "position": [1, 2, 3, 1],
        }
    )
    canonical, audit = _canonicalize_venus(
        frame, assay_id="example_activity", sequence="ACD", direction=-1
    )
    assert canonical.set_index("variant_id")["effect"].to_dict() == {
        "A1V": -4.0,
        "C2D": -4.0,
    }
    assert audit["mutation_column"] == "mutation"
    assert audit["outcome_column"] == "activity"


def test_pilot_module_does_not_name_domainome_outcome_sources() -> None:
    source = (Path(__file__).parents[1] / "src/variantshift/pilot.py").read_text()
    assert "domainome" in source.lower()
    assert "domainome-outcome" not in source.lower()
