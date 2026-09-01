import hashlib
import json

import pandas as pd
import pytest

from variantshift.model_adapters import ModelAdapter, ModelSpecification
from variantshift.provenance import sha256_file
from variantshift.qualification_audit import _load_execution, _parity, _repeatability
from variantshift.qualification_panel import _pdb_chain_a_sequence


class FixtureAdapter(ModelAdapter):
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        return variants.loc[:, ["variant_id"]].assign(score=0.0)


def _prediction_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol_id": ["q"] * len(scores),
            "panel_id": ["p"] * len(scores),
            "target_id": ["T1"] * len(scores),
            "variant_id": [f"A1{x}" for x in "CDE"[: len(scores)]],
            "model_id": ["m"] * len(scores),
            "model_version": ["1"] * len(scores),
            "score": scores,
            "status": ["ok"] * len(scores),
        }
    )


def test_repeatability_requires_matching_variant_keys() -> None:
    first = _prediction_frame([1.0, 2.0, 3.0])
    second = _prediction_frame([10.0, 20.0, 30.0]).iloc[::-1]
    correlation, rows, exact = _repeatability(first, second)
    assert correlation == pytest.approx(1.0)
    assert rows == 3
    assert exact == 0


def test_parity_is_computed_per_proteingym_target() -> None:
    predictions = _prediction_frame([1.0, 2.0, 3.0])
    official = pd.DataFrame(
        {
            "dms_id": ["D1"] * 3,
            "target_id": ["T1"] * 3,
            "variant_id": ["A1C", "A1D", "A1E"],
            "model_id": ["m"] * 3,
            "official_score": [2.0, 4.0, 6.0],
        }
    )
    result = _parity(predictions, official, "m")
    assert len(result) == 1
    assert result.iloc[0]["parity_spearman"] == pytest.approx(1.0)
    assert result.iloc[0]["shared_variants"] == 3


def test_execution_loader_rejects_any_cache_hit(tmp_path) -> None:
    predictions = _prediction_frame([1.0, 2.0, 3.0])
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    audit = pd.DataFrame(
        {
            "target_id": ["T1"],
            "cache_hit": [True],
            "coverage": [1.0],
            "status": ["ok"],
            "error": [""],
        }
    )
    audit.to_csv(tmp_path / "prediction-audit.csv", index=False)
    (tmp_path / "model-provenance.json").write_text(
        json.dumps({"m": {"checkpoint_sha256": "x"}})
    )
    artifacts = {
        name: sha256_file(tmp_path / name)
        for name in ("predictions.csv", "prediction-audit.csv", "model-provenance.json")
    }
    (tmp_path / "execution-manifest.json").write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "cache_hit_count": 1,
                "cache_namespace_fresh": False,
                "confirmation_outcomes_accessed": False,
            }
        )
    )
    with pytest.raises(ValueError, match="cache hits"):
        _load_execution(tmp_path, "m")


def test_checkpoint_hash_uses_torch_home(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "fixture.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    specification = ModelSpecification(
        model_id="m",
        model_version="1",
        family="fixture",
        modalities=("sequence",),
        adapter="fixture",
        source_url="https://example.org",
        license_name="MIT",
        license_status="permitted",
        checkpoint="fixture",
    )
    provenance = FixtureAdapter(specification).provenance()
    expected = hashlib.sha256(
        b"fixture.pt\0" + sha256_file(checkpoint).encode("ascii") + b"\n"
    ).hexdigest()
    assert provenance["checkpoint_sha256"] == expected


def test_pdb_sequence_parser_uses_chain_a_residue_order() -> None:
    payload = (
        "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 90.00           C  \n"
        "ATOM      2  CA  CYS A   2      1.000   0.000   0.000  1.00 90.00           C  \n"
        "ATOM      3  CA  GLY B   1      2.000   0.000   0.000  1.00 90.00           C  \n"
    ).encode("ascii")
    assert _pdb_chain_a_sequence(payload) == "AC"
