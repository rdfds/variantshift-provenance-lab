"""Build the outcome-free task-by-model feature table used at confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_adapters import load_model_specifications
from .outcome_lock import OUTCOME_COLUMN_MARKERS, assert_target_only
from .provenance import sha256_file
from .schemas import TRANSPORT_FEATURE_SCHEMA, validate_targets, write_table


def _interval_coverage(frame: pd.DataFrame, length: int) -> float:
    intervals = sorted(
        (max(1, int(row.domain_start)), min(length, int(row.domain_end)))
        for row in frame.itertuples(index=False)
        if pd.notna(row.domain_start) and pd.notna(row.domain_end)
    )
    if not intervals:
        return float("nan")
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end + 1:
            end = max(end, next_end)
        else:
            covered += end - start + 1
            start, end = next_start, next_end
    covered += end - start + 1
    return covered / length


def _sequence_similarity(alignments: pd.DataFrame) -> dict[tuple[str, str], float]:
    development = alignments.loc[alignments["target_source"].eq("development")]
    return (
        development.groupby(["query_panel_id", "query_target_id"])["sequence_identity"]
        .max()
        .to_dict()
    )


def _structure_similarity(
    structure_inputs: pd.DataFrame, edges: pd.DataFrame
) -> dict[tuple[str, str], float]:
    lookup = structure_inputs.set_index("structure_id").to_dict(orient="index")
    values: dict[tuple[str, str], float] = {}
    for row in edges.itertuples(index=False):
        first = lookup[str(row.structure_a)]
        second = lookup[str(row.structure_b)]
        if first["source"] == second["source"]:
            continue
        confirmation = first if first["source"] == "confirmation" else second
        key = (str(confirmation["panel_id"]), str(confirmation["target_id"]))
        values[key] = max(values.get(key, 0.0), float(row.reciprocal_minimum_tm_score))
    return values


def _score_shape(wide: pd.DataFrame, expected: int) -> dict[str, dict[str, float]]:
    correlations = wide.corr(method="spearman", min_periods=50)
    output = {}
    for model_id in wide.columns:
        values = pd.to_numeric(wide[model_id], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        dispersion = float(np.std(finite, ddof=0)) if len(finite) else np.nan
        if len(finite) and dispersion > 0:
            threshold = float(np.quantile(finite, 0.9))
            separation = float(
                (finite[finite >= threshold].mean() - np.median(finite)) / dispersion
            )
        else:
            separation = np.nan
        peer = correlations.loc[model_id].drop(index=model_id, errors="ignore")
        peer = peer[np.isfinite(peer)]
        median = float(peer.median()) if len(peer) else np.nan
        agreement = (median + 1.0) / 2.0 if np.isfinite(median) else np.nan
        output[str(model_id)] = {
            "score_dispersion": dispersion,
            "score_tail_separation": separation,
            "missing_fraction": 1.0 - len(finite) / max(expected, 1),
            "ensemble_agreement": agreement,
            "ensemble_disagreement": 1.0 - agreement if np.isfinite(agreement) else np.nan,
        }
    return output


def build_confirmation_transport_features(
    prediction_registry_path: Path,
    task_registry_path: Path,
    model_config_path: Path,
    target_paths: list[Path],
    variant_paths: list[Path],
    overlap_audit_path: Path,
    exposure_path: Path,
    pfam_annotations_path: Path,
    structure_inputs_path: Path,
    structure_edges_path: Path,
    structure_audit_path: Path,
    output_path: Path,
    *,
    protocol_id: str = "variantshift-confirmation-freeze-v1",
    model_ids: set[str] | None = None,
    minimum_model_coverage: float = 0.95,
) -> dict[str, object]:
    """Build descriptors from frozen targets, metadata, structures, and predictions only."""
    targets = pd.concat(
        [validate_targets(pd.read_csv(path)) for path in target_paths], ignore_index=True
    )
    assert_target_only(targets)
    variants = pd.concat([pd.read_csv(path) for path in variant_paths], ignore_index=True)
    assert_target_only(variants)
    tasks = pd.read_csv(task_registry_path)
    tasks = tasks.loc[tasks["included"].astype(bool)].copy()
    assert_target_only(tasks)
    blocked = OUTCOME_COLUMN_MARKERS.intersection(map(str.lower, tasks.columns))
    if blocked:
        raise ValueError(f"Task registry contains outcome columns: {sorted(blocked)}")
    registry = pd.read_csv(prediction_registry_path)
    overlap = pd.read_csv(overlap_audit_path)
    exposure = pd.read_csv(exposure_path)
    pfam = pd.read_csv(pfam_annotations_path)
    structure_inputs = pd.read_csv(structure_inputs_path)
    structure_edges = pd.read_csv(structure_edges_path)
    structure_audit = pd.read_csv(structure_audit_path)
    if model_ids is not None:
        available = set(registry["model_id"].astype(str))
        missing_models = sorted(model_ids.difference(available))
        if missing_models:
            raise ValueError(
                f"Prediction registry omits requested auditor models: {missing_models}"
            )
        registry = registry.loc[registry["model_id"].astype(str).isin(model_ids)].copy()
    all_specifications = {
        item.model_id: item for item in load_model_specifications(model_config_path)
    }
    selected_model_ids = set(registry["model_id"].astype(str))
    unknown_models = sorted(selected_model_ids.difference(all_specifications))
    if unknown_models:
        raise ValueError(
            "Prediction registry contains models absent from the frozen model config: "
            f"{unknown_models}"
        )
    specifications = {
        model_id: all_specifications[model_id]
        for model_id in sorted(selected_model_ids)
    }

    sequence_similarity = _sequence_similarity(
        pd.read_csv(Path(overlap_audit_path).with_name("confirmation-sequence-alignments.csv.gz"))
    )
    structure_similarity = _structure_similarity(structure_inputs, structure_edges)
    overlap_by_target = overlap.set_index(["panel_id", "target_id"])
    exposure_by_target = exposure.set_index(["panel_id", "target_id", "model_id"])
    structure_available = {
        (str(row.panel_id), str(row.target_id)): str(row.status) == "audited"
        for row in structure_audit.itertuples(index=False)
    }
    pfam_groups = (
        pfam.groupby(["panel_id", "target_id"])["pfam_clan_family_id"]
        .agg(lambda values: sorted(set(map(str, values))))
        .to_dict()
    )
    domain_coverage = {}
    target_lengths = targets.set_index(["panel_id", "target_id"])[
        "sequence_length"
    ].to_dict()
    for key, group in pfam.groupby(["panel_id", "target_id"]):
        length = int(target_lengths[(str(key[0]), str(key[1]))])
        if str(key[0]) == "human-domainome-v1":
            domain_coverage[(str(key[0]), str(key[1]))] = 1.0
        else:
            domain_coverage[(str(key[0]), str(key[1]))] = _interval_coverage(
                group, length
            )

    expected_counts = variants.groupby(["panel_id", "target_id"])["variant_id"].nunique()
    positions = variants.groupby(["panel_id", "target_id"])["position"].nunique()
    predictions: dict[tuple[str, str], dict[str, pd.Series]] = {}
    for record in registry.itertuples(index=False):
        frame = pd.read_csv(record.prediction_path)
        for target_id, group in frame.groupby("target_id"):
            predictions.setdefault((str(record.panel_id), str(target_id)), {})[
                str(record.model_id)
            ] = group.set_index("variant_id")["score"]

    rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    tasks_by_target = {
        key: group for key, group in tasks.groupby(["panel_id", "target_id"], sort=True)
    }
    targets_by_key = targets.set_index(["panel_id", "target_id"])
    for key, task_group in tasks_by_target.items():
        panel_id, target_id = map(str, key)
        if key not in predictions:
            for task in task_group.itertuples(index=False):
                exclusion_rows.append(
                    {
                        "panel_id": panel_id,
                        "target_id": target_id,
                        "assay_id": str(task.assay_id),
                        "reason": "no_selected_predictions",
                        "models": "",
                    }
                )
            continue
        wide = pd.concat(predictions[key], axis=1)
        expected = int(expected_counts.get(key, 0))
        shapes = _score_shape(wide, expected)
        missing_models = sorted(set(specifications).difference(shapes))
        low_coverage_models = sorted(
            model_id
            for model_id, values in shapes.items()
            if float(values["missing_fraction"]) > 1.0 - minimum_model_coverage + 1e-12
        )
        if missing_models or low_coverage_models:
            reasons = []
            if missing_models:
                reasons.append("missing_model_predictions")
            if low_coverage_models:
                reasons.append("below_minimum_model_coverage")
            for task in task_group.itertuples(index=False):
                exclusion_rows.append(
                    {
                        "panel_id": panel_id,
                        "target_id": target_id,
                        "assay_id": str(task.assay_id),
                        "reason": ";".join(reasons),
                        "models": ";".join(
                            sorted(set(missing_models + low_coverage_models))
                        ),
                    }
                )
            continue
        target = targets_by_key.loc[key]
        overlap_row = overlap_by_target.loc[key]
        families = pfam_groups.get(key, [])
        family_id = families[0] if len(families) == 1 else (
            "multi:" + hashlib.sha256("|".join(families).encode("utf-8")).hexdigest()[:12]
            if families
            else str(overlap_row.family_id)
        )
        for task in task_group.itertuples(index=False):
            for model_id, specification in specifications.items():
                exposure_row = exposure_by_target.loc[(panel_id, target_id, model_id)]
                complete_assay = panel_id == "human-domainome-v1"
                rows.append(
                    {
                        "protocol_id": protocol_id,
                        "panel_id": panel_id,
                        "dataset_id": panel_id,
                        "assay_id": str(task.assay_id),
                        "task_id": f"{panel_id}::{task.assay_id}",
                        "target_id": target_id,
                        "protein_id": str(target.protein_id),
                        "family_id": family_id,
                        "model_id": model_id,
                        "protein_length": int(target.sequence_length),
                        "assayed_fraction": (
                            float(positions.get(key, 0)) / int(target.sequence_length)
                            if complete_assay
                            else np.nan
                        ),
                        "mutated_positions": int(positions.get(key, 0))
                        if complete_assay
                        else np.nan,
                        "msa_depth": np.nan,
                        "msa_neff": np.nan,
                        "alignment_coverage": np.nan,
                        "sequence_identity_to_development": sequence_similarity.get(
                            key, 0.0
                        ),
                        "structure_similarity_to_development": structure_similarity.get(
                            key, np.nan
                        ),
                        "domain_coverage": domain_coverage.get(key, np.nan),
                        **shapes[model_id],
                        "taxon": "human"
                        if panel_id in {"human-domainome-v1", "mavedb-complement-v1"}
                        else "mixed",
                        "assay_modality": str(task.assay_modality),
                        "model_family": specification.family,
                        "model_modalities": "+".join(specification.modalities),
                        "exposure_status": str(exposure_row.exposure_category),
                        "structure_available": (
                            "yes" if structure_available.get(key, False) else "no"
                        ),
                        "exact_sequence_unseen": bool(overlap_row.exact_sequence_unseen),
                        "mmseqs_family_unseen": bool(overlap_row.mmseqs_family_unseen),
                        "pfam_clan_unseen": overlap_row.pfam_clan_unseen,
                        "foldseek_structure_family_unseen": (
                            overlap_row.foldseek_structure_family_unseen
                        ),
                    }
                )
    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No confirmation task passed the complete-model coverage gate")
    TRANSPORT_FEATURE_SCHEMA.validate(features)
    if features.columns.str.lower().isin(OUTCOME_COLUMN_MARKERS).any():
        raise RuntimeError("Outcome columns entered the confirmation feature table")
    write_table(features, output_path)
    exclusions = pd.DataFrame(
        exclusion_rows,
        columns=["panel_id", "target_id", "assay_id", "reason", "models"],
    )
    exclusion_path = output_path.with_suffix(".exclusions.csv")
    write_table(exclusions, exclusion_path)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "outcomes_accessed": False,
        "rows": len(features),
        "tasks": int(features["task_id"].nunique()),
        "models": int(features["model_id"].nunique()),
        "selected_model_ids": sorted(features["model_id"].astype(str).unique()),
        "minimum_model_coverage": minimum_model_coverage,
        "excluded_tasks": len(exclusions),
        "excluded_tasks_by_panel": {
            str(panel): int(count)
            for panel, count in exclusions.groupby("panel_id").size().items()
        },
        "missing_msa_policy": (
            "MSA depth, Neff, and MSA alignment coverage remain missing and use the frozen "
            "development-time imputer; they are not reverse-engineered from outcomes."
        ),
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                prediction_registry_path,
                task_registry_path,
                model_config_path,
                *target_paths,
                *variant_paths,
                overlap_audit_path,
                exposure_path,
                pfam_annotations_path,
                structure_inputs_path,
                structure_edges_path,
                structure_audit_path,
            ]
        },
        "artifacts": {
            str(output_path): sha256_file(output_path),
            str(exclusion_path): sha256_file(exclusion_path),
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
