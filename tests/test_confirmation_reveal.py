import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from variantshift.confirmation_panels import (
    _DOMAINOME_PREDICTOR_HEADER,
    DomainomePredictorSource,
)
from variantshift.confirmation_reveal import retrieve_registered_confirmation_outcomes
from variantshift.outcome_lock import create_outcome_lock, freeze_predictions, register_confirmation


def _domainome_fixture() -> bytes:
    values = {
        "domain_ID": "P1_PF00001_10",
        "uniprot_ID": "P1",
        "uniprot_ID_mutation": "P1_A10C",
        "aa_seq": "CC",
        "fitness": "1.5",
        "fitness_sigma": "0.1",
        "scaled_fitness": "2.0",
        "scaled_fitness_sigma": "0.2",
        "Organism": "human",
        "Gene Names (primary)": "GENE",
        "Gene Names (synonym)": "",
    }
    row = [values.get(column, "0.5") for column in _DOMAINOME_PREDICTOR_HEADER]
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr(
            "Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt",
            "\t".join(_DOMAINOME_PREDICTOR_HEADER) + "\n" + "\t".join(row) + "\n",
        )
    return output.getvalue()


def test_retrieve_registered_confirmation_outcomes_is_panel_limited_and_hashed(tmp_path) -> None:
    tasks = tmp_path / "tasks.csv"
    pd.DataFrame(
        [
            {"panel_id": "human-domainome-v1", "assay_id": "P1_PF00001_10",
             "target_id": "P1_PF00001_10", "direction": 1, "included": True,
             "outcomes_accessed": False},
            {"panel_id": "venusmuthub-v1", "assay_id": "VENUS_activity",
             "target_id": "venus-target", "direction": 1, "included": True,
             "outcomes_accessed": False},
            {"panel_id": "mavedb-complement-v1", "assay_id": "DO_NOT_FETCH",
             "target_id": "mave-target", "direction": 1, "included": True,
             "outcomes_accessed": False},
        ]
    ).to_csv(tasks, index=False)
    domainome_targets = tmp_path / "domainome-targets.csv"
    pd.DataFrame({"target_id": ["P1_PF00001_10"], "sequence": ["AC"]}).to_csv(
        domainome_targets, index=False
    )
    domainome_variants = tmp_path / "domainome-variants.csv"
    pd.DataFrame({"target_id": ["P1_PF00001_10"], "variant_id": ["A1C"]}).to_csv(
        domainome_variants, index=False
    )
    venus_targets = tmp_path / "venus-targets.csv"
    pd.DataFrame({"target_id": ["venus-target"], "sequence": ["ACD"]}).to_csv(
        venus_targets, index=False
    )
    venus_audit = tmp_path / "venus-audit.csv"
    pd.DataFrame(
        {"dataset_id": ["VENUS_activity"], "source_path": ["single_mutant/activity/a.csv"]}
    ).to_csv(venus_audit, index=False)
    venus_protocol = tmp_path / "venus-protocol.json"
    venus_protocol.write_text(
        json.dumps({"protocol_id": "variantshift-venusmuthub-confirmation-v1",
                    "source_git_commit": "frozen-revision"})
    )
    prediction, method = tmp_path / "prediction", tmp_path / "method"
    prediction.write_text("prediction")
    method.write_text("method")
    lock = tmp_path / "lock.json"
    create_outcome_lock(
        lock,
        protocol_id="variantshift-confirmation-freeze-v2",
        target_artifacts=[tasks, domainome_targets, domainome_variants, venus_targets],
    )
    freeze_predictions(lock, prediction_artifacts=[prediction], method_artifacts=[method])
    with pytest.raises(PermissionError):
        retrieve_registered_confirmation_outcomes(
            lock, tasks, domainome_targets, domainome_variants, venus_targets, venus_audit,
            venus_protocol, tmp_path / "blocked"
        )
    register_confirmation(lock, registration_uri="https://osf.io/example")
    domainome_payload = _domainome_fixture()
    urls = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        if url == "https://example.test/domainome.zip":
            return domainome_payload
        if "VenusMutHub" in url:
            return b"mutation,activity\nA1C,3.0\nC2A,2.0\nD3A,1.0\n"
        raise AssertionError(f"Unexpected URL: {url}")

    source = DomainomePredictorSource(
        url="https://example.test/domainome.zip",
        expected_md5=hashlib.md5(domainome_payload).hexdigest(),
    )
    outputs = retrieve_registered_confirmation_outcomes(
        lock, tasks, domainome_targets, domainome_variants, venus_targets, venus_audit,
        venus_protocol, tmp_path / "reveal", domainome_source=source, fetch_bytes=fetch
    )
    outcomes = pd.read_csv(outputs["outcomes"])
    assert set(outcomes["panel_id"]) == {"human-domainome-v1", "venusmuthub-v1"}
    assert outcomes.set_index("panel_id").loc["human-domainome-v1", "effect"] == 2.0
    assert "DO_NOT_FETCH" not in " ".join(urls)
    ledger = json.loads(outputs["ledger"].read_text())
    assert ledger["mavedb_outcomes_requested"] is False
    assert len(ledger["requests"]) == 2
    assert all(item["sha256"] and item["accessed_at_utc"] for item in ledger["requests"])
