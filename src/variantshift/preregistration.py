"""Generate an immutable, outcome-blind registration package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .outcome_lock import read_outcome_lock
from .provenance import git_revision, sha256_file

PRIMARY_QUESTIONS = (
    (
        "How much do model rankings and apparent performance deteriorate from random variants to "
        "unseen positions, proteins, families, assay modalities, and external datasets?"
    ),
    (
        "Can label-free properties of a new task predict whether each model will provide useful "
        "variant selection?"
    ),
    (
        "Can an outcome-free confidence rule reduce failed deployments by abstaining while "
        "otherwise using the best fixed development model?"
    ),
)


def build_preregistration_model_audit(
    qualification_audit_path: Path,
    qualification_summary_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Convert the frozen qualification evidence into the preregistration contract."""
    audit = pd.read_csv(qualification_audit_path)
    summary = json.loads(qualification_summary_path.read_text(encoding="utf-8"))
    required = {"model_id", "family", "qualification_status"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Qualification audit is missing columns: {missing}")
    shared = int(summary["shared_confirmation_targets"])
    summary_gates = dict(summary.get("gates", {}))
    feasibility = bool(summary_gates) and all(map(bool, summary_gates.values()))
    audit = audit.copy()
    audit["primary_eligible"] = audit["qualification_status"].astype(str).eq("passed")
    audit["exclusion_reason"] = np.where(
        audit["primary_eligible"], "", audit["qualification_status"].astype(str)
    )
    audit["primary_shared_target_count"] = shared
    audit["feasibility_gate_passed"] = feasibility
    columns = [
        "model_id",
        "family",
        "primary_eligible",
        "exclusion_reason",
        "primary_shared_target_count",
        "feasibility_gate_passed",
        *[
            column
            for column in audit.columns
            if column
            not in {
                "model_id",
                "family",
                "primary_eligible",
                "exclusion_reason",
                "primary_shared_target_count",
                "feasibility_gate_passed",
            }
        ],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.loc[:, columns].to_csv(output_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "eligible_models": int(audit["primary_eligible"].sum()),
        "eligible_families": int(
            audit.loc[audit["primary_eligible"], "family"].nunique()
        ),
        "shared_confirmation_targets": shared,
        "feasibility_gate_passed": feasibility,
        "inputs": {
            str(qualification_audit_path): sha256_file(qualification_audit_path),
            str(qualification_summary_path): sha256_file(qualification_summary_path),
        },
        "artifact": {str(output_path): sha256_file(output_path)},
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_locked_artifacts(lock: dict[str, object]) -> None:
    for section in ("target_artifacts", "prediction_artifacts", "method_artifacts"):
        records = lock.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"Outcome lock has no {section}")
        for name, expected in records.items():
            path = Path(name)
            if not path.is_file():
                raise ValueError(f"Locked artifact is unavailable: {path}")
            if sha256_file(path) != expected:
                raise ValueError(f"Locked artifact changed after freezing: {path}")


def build_preregistration_bundle(
    protocol_path: Path,
    outcome_lock_path: Path,
    model_audit_path: Path,
    method_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Build files suitable for a public OSF registration before outcome access."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    lock = read_outcome_lock(outcome_lock_path)
    if lock["state"] != "predictions_frozen":
        raise ValueError("Preregistration must be built while outcomes remain predictions_frozen")
    _verify_locked_artifacts(lock)
    method = json.loads(Path(method_path).read_text(encoding="utf-8"))
    model_audit = pd.read_csv(model_audit_path)
    required_model_columns = {
        "model_id",
        "family",
        "primary_eligible",
        "exclusion_reason",
        "primary_shared_target_count",
        "feasibility_gate_passed",
    }
    missing = sorted(required_model_columns.difference(model_audit.columns))
    if missing:
        raise ValueError(f"Model audit is missing columns: {missing}")
    eligible = sorted(
        model_audit.loc[model_audit["primary_eligible"].astype(bool), "model_id"].astype(str)
    )
    excluded = {
        str(row.model_id): str(row.exclusion_reason)
        for row in model_audit.loc[~model_audit["primary_eligible"].astype(bool)].itertuples()
    }
    eligible_family_count = int(
        model_audit.loc[model_audit["primary_eligible"].astype(bool), "family"].nunique()
    )
    shared_target_counts = set(
        model_audit["primary_shared_target_count"].dropna().astype(int)
    )
    if len(shared_target_counts) != 1:
        raise ValueError("Model audit has inconsistent shared-target counts")
    shared_target_count = next(iter(shared_target_counts))
    feasibility_gate = bool(model_audit["feasibility_gate_passed"].astype(bool).all())
    if not (
        feasibility_gate
        and len(eligible) >= 8
        and eligible_family_count >= 4
        and shared_target_count >= 300
    ):
        raise ValueError(
            "Model-panel feasibility gate failed: require at least 8 executable models, "
            "4 model/input families, and 300 shared targets at >=95% coverage"
        )
    registration = {
        "schema_version": 1,
        "title": "VariantShift: outcome-blind evaluation of variant-effect model transport",
        "protocol_id": protocol["protocol_id"],
        "panel_id": protocol["panel_id"],
        "source_git_commit": git_revision(Path.cwd()),
        "outcome_state": lock["state"],
        "primary_questions": list(PRIMARY_QUESTIONS),
        "primary_utility": "task-level standardized top-decile selection gain",
        "primary_failure": "selection_gain_sd <= 0",
        "primary_endpoint": "task-level selection-regret coverage AUC",
        "primary_comparator": "always deploy VespaG with analytical random abstention",
        "secondary_endpoints": [
            "failure risk-coverage AUC",
            "Spearman correlation",
            "top-decile recall",
            "normalized discounted cumulative gain",
            "best-variant regret",
            "model-rank stability",
            "marginal and position-conditional coverage",
        ],
        "resampling_unit": "family, then protein, then assay; variants are not independent units",
        "confirmation_characterization": "outcome-blind retrospective confirmation",
        "eligible_models": eligible,
        "eligible_model_family_count": eligible_family_count,
        "shared_confirmation_targets": shared_target_count,
        "excluded_models": excluded,
        "transport_method": method,
        "inclusion": protocol.get("inclusion", {}),
        "exclusion": protocol.get("exclusion", {}),
        "subgroups": [
            "all eligible tasks",
            "exact-sequence unseen",
            "MMseqs2 sequence-family unseen",
            "Foldseek structure-family unseen",
            "Pfam-clan unseen",
            "model-specific temporal-clean",
        ],
        "multiplicity": "Holm adjustment across preregistered primary policy comparisons",
        "bootstrap_repeats": 10_000,
        "locked_artifacts": {
            section: lock[section]
            for section in ("target_artifacts", "prediction_artifacts", "method_artifacts")
        },
        "stopping_rule": "One reveal after public registration; no method or threshold refitting.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "registration": output_dir / "registration.json",
        "narrative": output_dir / "PREREGISTRATION.md",
        "model_audit": output_dir / "model-audit.csv",
        "checksums": output_dir / "checksums.sha256",
    }
    outputs["registration"].write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_audit.to_csv(outputs["model_audit"], index=False, lineterminator="\n")
    question_text = "\n".join(
        f"{index}. {question}" for index, question in enumerate(PRIMARY_QUESTIONS, start=1)
    )
    eligible_text = "\n".join(f"- {model}" for model in eligible) or "- None passed"
    excluded_text = (
        "\n".join(f"- {model}: {reason}" for model, reason in sorted(excluded.items()))
        or "- None"
    )
    outputs["narrative"].write_text(
        f"""# VariantShift confirmation preregistration

## Status

This record was produced before confirmation outcomes were accessed. The study is an
outcome-blind retrospective confirmation, not prospective or independently blinded validation.

## Primary questions

{question_text}

## Estimands and analysis

The primary utility is task-level standardized top-decile selection gain. Selection regret is the
difference between the best available model's gain and the selected model's gain. The primary
reliability endpoint is task-level selection-regret coverage AUC. Failure is a gain less than or
equal to zero, and failure risk-coverage AUC is secondary.
Families, proteins, and assays are the resampling hierarchy; individual variants are not treated
as independent replicates. Primary policy comparisons use 10,000 hierarchical bootstrap repeats
and Holm multiplicity adjustment. The primary comparator always deploys VespaG; VariantShift v2
may only deploy VespaG or abstain and may never switch to another model.

## Frozen primary model panel

{eligible_text}

## Recorded exclusions

{excluded_text}

## Reveal rule

Confirmation effects may be retrieved once after this bundle receives a public OSF or Zenodo
timestamp. The auditor model, feature list, confidence rankings, model predictions, inclusion
rules, and subgroup definitions will not be changed after reveal. The development leave-one-panel-
out failure is disclosed in advance, and a failed confirmation will be reported as a negative
result rather than followed by method refitting. Any additional analysis will be labeled
exploratory.
""",
        encoding="utf-8",
    )
    checksum_files = [
        outputs["registration"],
        outputs["narrative"],
        outputs["model_audit"],
        Path(protocol_path),
        Path(outcome_lock_path),
        Path(method_path),
    ]
    outputs["checksums"].write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_files),
        encoding="utf-8",
    )
    return outputs
