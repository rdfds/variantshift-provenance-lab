import pandas as pd
import pytest

from variantshift.model_adapters import (
    CARPAdapter,
    ModelAdapter,
    ModelSpecification,
    ProSSTAdapter,
    VespaGAdapter,
    adapter_from_specification,
    prediction_cache_key,
    score_panel,
)
from variantshift.schemas import all_single_substitutions, sequence_sha256


class DeterministicAdapter(ModelAdapter):
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        return variants.assign(score=variants["position"].astype(float)).loc[
            :, ["variant_id", "score"]
        ]


class FailingAdapter(ModelAdapter):
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        raise ModuleNotFoundError("missing-runtime")


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


def test_total_panel_failure_preserves_first_target_error(tmp_path) -> None:
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
        model_id="failing",
        model_version="1",
        family="fixture",
        modalities=("sequence",),
        adapter="fixture",
        source_url="https://example.org/model",
        license_name="MIT",
        license_status="permitted",
    )
    with pytest.raises(RuntimeError, match="T1: ModuleNotFoundError: missing-runtime"):
        score_panel(
            FailingAdapter(specification),
            targets,
            variants,
            protocol_id="p1",
            cache_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("adapter_name", "expected_type"),
    [("prosst", ProSSTAdapter), ("vespag", VespaGAdapter), ("carp", CARPAdapter)],
)
def test_extension_adapters_are_explicitly_routed(adapter_name, expected_type) -> None:
    specification = ModelSpecification(
        model_id=adapter_name,
        model_version="1",
        family="fixture",
        modalities=("sequence",),
        adapter=adapter_name,
        source_url="https://example.org/model",
        license_name="test",
        license_status="test",
    )
    assert isinstance(adapter_from_specification(specification), expected_type)


def test_vespag_rejects_unvalidated_long_sequence_before_inference() -> None:
    specification = ModelSpecification(
        model_id="vespag",
        model_version="1",
        family="fixture",
        modalities=("sequence",),
        adapter="vespag",
        source_url="https://example.org/model",
        license_name="test",
        license_status="test",
        strategy="raw-esm2-3b-fnn-v2",
    )
    adapter = VespaGAdapter(specification)
    adapter._runtime = (object(), object(), object(), object(), "cpu")
    target = pd.Series({"target_id": "long", "sequence": "A" * 1023})
    variants = pd.DataFrame(
        {"variant_id": ["A1C"], "position": [1], "reference": ["A"], "alternate": ["C"]}
    )
    with pytest.raises(ValueError, match="longer than 1,022"):
        adapter.score_target(target, variants)
