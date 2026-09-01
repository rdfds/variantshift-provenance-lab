"""Audit published, outcome-blind prediction panels without calling them executions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .provenance import sha256_file


def audit_official_prediction_panel(
    config_path: Path,
    predictions_path: Path,
    receipt_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Measure per-target score coverage and a distinct published-score feasibility gate."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("outcomes_accessed") is not False:
        raise ValueError("Official prediction receipt does not attest outcome-blind extraction")
    if sha256_file(predictions_path) != receipt.get("prediction_sha256"):
        raise ValueError("Official prediction artifact differs from its extraction receipt")
    models = config["models"]
    identifiers = [str(model["model_id"]) for model in models]
    columns = [str(model["source_column"]) for model in models]
    if len(identifiers) != len(set(identifiers)) or len(columns) != len(set(columns)):
        raise ValueError("Official model identifiers and source columns must be unique")
    counts: dict[str, dict[str, list[int]]] = {
        column: defaultdict(lambda: [0, 0]) for column in columns
    }
    for chunk in pd.read_csv(
        predictions_path,
        usecols=["target_id", *columns],
        chunksize=50_000,
    ):
        target_ids = chunk["target_id"].astype(str)
        for column in columns:
            present = pd.to_numeric(chunk[column], errors="coerce").notna()
            grouped = pd.DataFrame({"target_id": target_ids, "present": present}).groupby(
                "target_id", sort=False
            )["present"]
            for target_id, values in grouped.agg(["sum", "count"]).iterrows():
                counts[column][str(target_id)][0] += int(values["sum"])
                counts[column][str(target_id)][1] += int(values["count"])
    rows = []
    passing_targets_by_model: dict[str, set[str]] = {}
    for model in models:
        model_id = str(model["model_id"])
        column = str(model["source_column"])
        target_counts = counts[column]
        observed = sum(item[0] for item in target_counts.values())
        total = sum(item[1] for item in target_counts.values())
        passing = {
            target_id
            for target_id, (present, target_total) in target_counts.items()
            if target_total and present / target_total >= 0.95
        }
        eligible = bool(
            str(model["license_status"]) == "permitted" and total and observed / total >= 0.95
        )
        if eligible:
            passing_targets_by_model[model_id] = passing
        rows.append(
            {
                "model_id": model_id,
                "source_column": column,
                "family": str(model["family"]),
                "modalities": ";".join(map(str, model["modalities"])),
                "source_url": str(model["source_url"]),
                "license_name": str(model["license_name"]),
                "license_status": str(model["license_status"]),
                "evidence_type": "official_precomputed_scores",
                "execution_status": "not_executed_by_variantshift",
                "mutation_coverage": observed / total if total else 0.0,
                "targets_at_95pct_coverage": len(passing),
                "official_score_eligible": eligible,
                "exclusion_reason": "" if eligible else "sub_95pct_or_license_gate",
            }
        )
    audit = pd.DataFrame(rows)
    eligible_ids = audit.loc[audit["official_score_eligible"], "model_id"].astype(str)
    eligible_sets = [passing_targets_by_model[model_id] for model_id in eligible_ids]
    shared = set.intersection(*eligible_sets) if eligible_sets else set()
    family_count = int(audit.loc[audit["official_score_eligible"], "family"].nunique())
    gate = bool(len(eligible_ids) >= 8 and family_count >= 4 and len(shared) >= 300)
    audit["official_eligible_model_count"] = len(eligible_ids)
    audit["official_family_count"] = family_count
    audit["official_shared_target_count"] = len(shared)
    audit["official_shared_targets_sha256"] = (
        hashlib.sha256("\n".join(sorted(shared)).encode("utf-8")).hexdigest() if shared else ""
    )
    audit["official_score_gate_passed"] = gate
    audit["executable_model_gate_passed"] = False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False, lineterminator="\n")
    return audit
