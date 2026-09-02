import pandas as pd

from variantshift.model_adapters import (
    ModelAdapter,
    ModelSpecification,
    prediction_cache_key,
    score_panel,
)
from variantshift.schemas import all_single_substitutions, sequence_sha256


class DeterministicAdapter(ModelAdapter):
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        return variants.assign(score=variants["position"].astype(float)).loc[
            :, ["variant_id", "score"]
        ]


def test_content_addressed_panel_scoring_reuses_target_cache(tmp_path) -> None:
    targets = pd.DataFrame(
        {
            "panel_id": ["panel"],
            "target_id": ["T1"],
            "protein_id": ["P1"],
            "sequence": ["AC"],
            "sequence_sha256": [sequence_sha256("AC")],
            "sequence_length": [2],
        }
    )
    variants = all_single_substitutions(targets)
    specification = ModelSpecification(
        model_id="fixture",
        model_version="1",
        family="fixture",
        modalities=("sequence",),
        adapter="fixture",
        source_url="https://example.org/model",
        license_name="MIT",
        license_status="permitted",
    )
    adapter = DeterministicAdapter(specification)
    first, first_audit = score_panel(
        adapter, targets, variants, protocol_id="p1", cache_dir=tmp_path
    )
    second, second_audit = score_panel(
        adapter, targets, variants, protocol_id="p1", cache_dir=tmp_path
    )
    assert len(first) == len(second) == 38
    assert not first_audit["cache_hit"].any()
    assert second_audit["cache_hit"].all()
    assert prediction_cache_key(specification, targets.iloc[0], variants) in str(
        next((tmp_path / "fixture").iterdir())
    )
