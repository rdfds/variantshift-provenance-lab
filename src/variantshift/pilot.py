"""Outcome-blind development pilot for deciding whether full confirmation is warranted.

The pilot is deliberately not confirmatory evidence. It consumes a deterministic subset of
external outcomes only after its targets, model executions, selector, and analysis code are
frozen. Human Domainome and the remaining VenusMutHub targets are never read here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .external_validation import MAVEDB_API_BASE, canonicalize_mavedb_scores
from .metrics import regression_metrics, top_selection_metrics
from .outcome_lock import read_outcome_lock
from .provenance import sha256_file
from .schemas import OUTCOME_SCHEMA, TARGET_SCHEMA, TASK_METRIC_SCHEMA, write_table

PILOT_PROTOCOL_ID = "variantshift-external-development-pilot-v1"
PILOT_SEED = "variantshift-external-development-pilot-v1|2026-09-01"
VENUS_SOURCE_BASE = "https://huggingface.co/datasets/AI4Protein/VenusMutHub/resolve"
_ONE_LETTER_SUBSTITUTION = re.compile(
    r"^(?:p\.)?(?P<reference>[ACDEFGHIKLMNPQRSTVWY])(?P<position>[1-9][0-9]*)"
    r"(?P<alternate>[ACDEFGHIKLMNPQRSTVWY])$",
    re.IGNORECASE,
)
_THREE_LETTER_SUBSTITUTION = re.compile(
    r"^(?:p\.)?(?P<reference>[A-Z][a-z]{2})(?P<position>[1-9][0-9]*)"
    r"(?P<alternate>[A-Z][a-z]{2})$"
)
_AA3_TO_AA1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_order(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _stratified_venus_targets(targets: pd.DataFrame, count: int) -> set[str]:
    if count < 4 or count > len(targets):
        raise ValueError("Venus pilot target count must be between four and panel size")
    ordered = targets.sort_values(["sequence_length", "target_id"]).reset_index(drop=True)
    ordered["length_stratum"] = np.floor(np.arange(len(ordered)) * 4 / len(ordered)).astype(int)
    base, remainder = divmod(count, 4)
    selected: list[str] = []
    for stratum in range(4):
        take = base + int(stratum < remainder)
        candidates = ordered.loc[ordered["length_stratum"].eq(stratum), "target_id"].astype(str)
        ranked = sorted(candidates, key=lambda value: _hash_order(PILOT_SEED, value))
        if len(ranked) < take:
            raise RuntimeError(f"Length stratum {stratum} cannot supply {take} targets")
        selected.extend(ranked[:take])
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("Deterministic Venus pilot selection is not one-to-one")
    return set(selected)


def freeze_development_pilot(
    task_registry_path: Path,
    mavedb_targets_path: Path,
    venus_targets_path: Path,
    mavedb_audit_path: Path,
    venus_audit_path: Path,
    output_dir: Path,
    *,
    model_ids: list[str],
    venus_target_count: int = 24,
    shared_intersection_path: Path,
) -> dict[str, Path]:
    """Freeze a deterministic external pilot while preserving a disjoint holdout."""
    tasks = pd.read_csv(task_registry_path)
    tasks = tasks.loc[
        tasks["included"].astype(bool)
        & tasks["panel_id"].isin(["mavedb-complement-v1", "venusmuthub-v1"])
    ].copy()
    mavedb_targets = pd.read_csv(mavedb_targets_path)
    venus_targets = pd.read_csv(venus_targets_path)
    shared = pd.read_csv(shared_intersection_path)
    shared = shared.loc[shared["shared_all_models"].astype(bool)].copy()
    shared_by_panel = {
        str(panel): set(group["target_id"].astype(str))
        for panel, group in shared.groupby("panel_id")
    }
    TARGET_SCHEMA.validate(mavedb_targets)
    TARGET_SCHEMA.validate(venus_targets)
    eligible_venus = set(
        tasks.loc[tasks["panel_id"].eq("venusmuthub-v1"), "target_id"].astype(str)
    ).intersection(shared_by_panel.get("venusmuthub-v1", set()))
    venus_candidates = venus_targets.loc[
        venus_targets["target_id"].astype(str).isin(eligible_venus)
    ].copy()
    selected_venus = _stratified_venus_targets(venus_candidates, venus_target_count)
    pilot_mask = (
        tasks["panel_id"].eq("mavedb-complement-v1")
        & tasks["target_id"].astype(str).isin(
            shared_by_panel.get("mavedb-complement-v1", set())
        )
    ) | (
        tasks["panel_id"].eq("venusmuthub-v1")
        & tasks["target_id"].astype(str).isin(selected_venus)
    )
    tasks["partition"] = np.where(pilot_mask, "development_pilot", "untouched_holdout")
    pilot_tasks = tasks.loc[pilot_mask].drop(columns="partition").copy()
    holdout_tasks = tasks.loc[~pilot_mask].drop(columns="partition").copy()
    pilot_targets = pd.concat(
        [
            mavedb_targets.loc[
                mavedb_targets["target_id"].astype(str).isin(
                    set(pilot_tasks["target_id"].astype(str))
                )
            ],
            venus_targets.loc[venus_targets["target_id"].astype(str).isin(selected_venus)],
        ],
        ignore_index=True,
    )
    pilot_targets = pilot_targets.loc[:, list(TARGET_SCHEMA.required)]
    TARGET_SCHEMA.validate(pilot_targets)
    if set(pilot_tasks["target_id"]).intersection(holdout_tasks["target_id"]):
        raise RuntimeError("A target crossed the development-pilot/holdout boundary")

    mavedb_audit = pd.read_csv(mavedb_audit_path)
    mavedb_source = mavedb_audit.loc[
        mavedb_audit["selected"].astype(bool)
        & mavedb_audit["urn"].astype(str).isin(pilot_tasks["assay_id"].astype(str))
    ].copy()
    if set(mavedb_source["urn"].astype(str)) != set(
        pilot_tasks.loc[pilot_tasks["panel_id"].eq("mavedb-complement-v1"), "assay_id"].astype(str)
    ):
        raise ValueError("MaveDB pilot tasks and frozen selected score sets differ")
    venus_audit = pd.read_csv(venus_audit_path)
    venus_source = venus_audit.loc[
        venus_audit["dataset_id"].astype(str).isin(
            pilot_tasks.loc[pilot_tasks["panel_id"].eq("venusmuthub-v1"), "assay_id"].astype(str)
        )
    ].copy()
    if set(venus_source["dataset_id"].astype(str)) != set(
        pilot_tasks.loc[pilot_tasks["panel_id"].eq("venusmuthub-v1"), "assay_id"].astype(str)
    ):
        raise ValueError("Venus pilot tasks and frozen source paths differ")
    source_tree_path = Path(venus_audit_path).with_name("source-tree.json")
    source_tree = json.loads(source_tree_path.read_text(encoding="utf-8"))
    oid_by_path = {str(item["path"]): str(item["oid"]) for item in source_tree if item["type"] == "file"}
    source_rows = [
        {
            "panel_id": "mavedb-complement-v1",
            "assay_id": str(row.urn),
            "target_id": str(row.target_id),
            "source_locator": f"{MAVEDB_API_BASE}/score-sets/{row.urn}/scores",
            "source_object_id": str(row.detail_metadata_sha256),
        }
        for row in mavedb_source.itertuples(index=False)
    ]
    source_rows.extend(
        {
            "panel_id": "venusmuthub-v1",
            "assay_id": str(row.dataset_id),
            "target_id": str(row.target_id),
            "source_locator": str(row.source_path),
            "source_object_id": oid_by_path[str(row.source_path)],
        }
        for row in venus_source.itertuples(index=False)
    )
    source_manifest = pd.DataFrame(source_rows).sort_values(["panel_id", "assay_id"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "split": output_dir / "task-partitions.csv",
        "pilot_tasks": output_dir / "pilot-task-registry.csv",
        "holdout_tasks": output_dir / "untouched-holdout-task-registry.csv",
        "pilot_targets": output_dir / "pilot-targets.csv",
        "sources": output_dir / "pilot-source-manifest.csv",
        "protocol": output_dir / "pilot-protocol.json",
    }
    write_table(tasks.sort_values(["panel_id", "target_id", "assay_id"]), outputs["split"])
    write_table(pilot_tasks.sort_values(["panel_id", "target_id", "assay_id"]), outputs["pilot_tasks"])
    write_table(holdout_tasks.sort_values(["panel_id", "target_id", "assay_id"]), outputs["holdout_tasks"])
    write_table(pilot_targets.sort_values(["panel_id", "target_id"]), outputs["pilot_targets"])
    write_table(source_manifest, outputs["sources"])
    source_revision = json.loads(
        Path(venus_audit_path).with_name("target-freeze-protocol.json").read_text(encoding="utf-8")
    )["source"]["revision"]
    protocol = {
        "schema_version": 1,
        "protocol_id": PILOT_PROTOCOL_ID,
        "status": "outcome_blind_development_pilot_frozen",
        "created_at_utc": _now(),
        "outcomes_accessed": False,
        "claim_boundary": (
            "Development evidence for a go/no-go decision only; never confirmation evidence."
        ),
        "model_ids": sorted(set(map(str, model_ids))),
        "selection": {
            "mavedb": (
                "all direction-resolved tasks whose targets are in the frozen six-model "
                "95%-coverage intersection"
            ),
            "venus": (
                f"{venus_target_count} target-level selections, six per sequence-length quartile, "
                "ranked by SHA-256 of a frozen seed and target id"
            ),
            "seed": PILOT_SEED,
            "target_level_partition": True,
        },
        "outcome_parsing": {
            "mavedb": "frozen metadata direction; finite single substitutions; duplicates median",
            "venus": (
                "single-substitution column chosen by maximum valid-sequence match rate; outcome "
                "column chosen from predeclared metric-name tokens then finite-row count; frozen "
                "direction; duplicates median; no manual repair"
            ),
            "minimum_aligned_variants_per_task_model": 10,
        },
        "primary_endpoints": [
            "selection-regret risk-coverage AUC versus frozen best label-free comparator",
            "failed-deployment rate at 50% task coverage",
            "mean top-decile selection gain",
        ],
        "venus_source_revision": source_revision,
        "counts": {
            "pilot_tasks": int(pilot_tasks["assay_id"].nunique()),
            "pilot_targets": int(pilot_tasks["target_id"].nunique()),
            "untouched_venus_tasks": int(holdout_tasks["assay_id"].nunique()),
            "untouched_venus_targets": int(holdout_tasks["target_id"].nunique()),
            "domainome_targets_untouched": 426,
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                task_registry_path,
                mavedb_targets_path,
                venus_targets_path,
                mavedb_audit_path,
                venus_audit_path,
                source_tree_path,
                shared_intersection_path,
            ]
        },
        "artifacts": {
            str(path): sha256_file(path)
            for name, path in outputs.items()
            if name != "protocol"
        },
    }
    outputs["protocol"].write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return outputs


def freeze_development_feature_subset(
    features_path: Path,
    output_path: Path,
    *,
    model_ids: list[str],
) -> dict[str, object]:
    frame = pd.read_csv(features_path)
    selected = sorted(set(map(str, model_ids)))
    subset = frame.loc[frame["model_id"].astype(str).isin(selected)].copy()
    missing = sorted(set(selected).difference(subset["model_id"].astype(str)))
    if missing:
        raise ValueError(f"Development features omit pilot models: {missing}")
    task_counts = subset.groupby("task_id")["model_id"].nunique()
    if not task_counts.eq(len(selected)).all():
        raise ValueError("The pilot development feature matrix is incomplete")
    TASK_METRIC_SCHEMA.validate(subset)
    write_table(subset, output_path)
    return {
        "outcomes_accessed": False,
        "models": selected,
        "tasks": int(subset["task_id"].nunique()),
        "rows": len(subset),
        "source_sha256": sha256_file(features_path),
        "artifact_sha256": sha256_file(output_path),
    }


def _verify_predictions_frozen(lock_path: Path) -> dict[str, object]:
    payload = read_outcome_lock(lock_path)
    if payload["state"] != "predictions_frozen":
        raise PermissionError("Pilot outcomes require a complete predictions_frozen lock")
    for section in ("target_artifacts", "prediction_artifacts", "method_artifacts"):
        for name, expected in dict(payload.get(section, {})).items():
            path = Path(name)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Frozen pilot artifact changed or disappeared: {path}")
    return payload


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "VariantShift/1.0 development-pilot client"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _parse_substitution(value: object, sequence: str) -> str | None:
    text = re.sub(r"[\s_-]", "", str(value).strip())
    match = _ONE_LETTER_SUBSTITUTION.fullmatch(text)
    if match:
        reference = match.group("reference").upper()
        alternate = match.group("alternate").upper()
        position = int(match.group("position"))
    else:
        match3 = _THREE_LETTER_SUBSTITUTION.fullmatch(text)
        if not match3:
            return None
        reference = _AA3_TO_AA1.get(match3.group("reference").title(), "")
        alternate = _AA3_TO_AA1.get(match3.group("alternate").title(), "")
        position = int(match3.group("position"))
    if not reference or not alternate or reference == alternate:
        return None
    if position > len(sequence) or sequence[position - 1] != reference:
        return None
    return f"{reference}{position}{alternate}"


def _venus_outcome_column(frame: pd.DataFrame, mutation_column: str, assay_id: str) -> str:
    normalized = {
        column: re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        for column in frame.columns
    }
    assay_tokens = set(re.sub(r"[^a-z0-9]+", "_", assay_id.lower()).split("_"))
    preferred_tokens = assay_tokens | {
        "activity", "kcat", "kcatkm", "km", "score", "fitness", "value", "effect"
    }
    candidates: list[tuple[int, int, str]] = []
    for column in frame.columns:
        if column == mutation_column:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = int(np.isfinite(values).sum())
        if finite < 3 or values.nunique(dropna=True) < 2:
            continue
        tokens = set(normalized[column].split("_"))
        priority = int(bool(tokens.intersection(preferred_tokens)))
        if tokens.intersection({"position", "site", "index", "residue", "rank"}):
            priority -= 2
        candidates.append((priority, finite, str(column)))
    if not candidates:
        raise ValueError("No finite non-constant Venus outcome column")
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][2]


def _canonicalize_venus(
    frame: pd.DataFrame,
    *,
    assay_id: str,
    sequence: str,
    direction: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    mutation_candidates = []
    for column in frame.columns:
        parsed = frame[column].map(lambda value: _parse_substitution(value, sequence))
        valid = int(parsed.notna().sum())
        if valid:
            name = re.sub(r"[^a-z0-9]+", "_", str(column).lower())
            priority = int(any(token in name for token in ("mut", "variant", "substitution")))
            mutation_candidates.append((priority, valid, str(column), parsed))
    if not mutation_candidates:
        raise ValueError("No Venus column contains valid single substitutions")
    mutation_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    _, _, mutation_column, parsed = mutation_candidates[0]
    outcome_column = _venus_outcome_column(frame, mutation_column, assay_id)
    effect = pd.to_numeric(frame[outcome_column], errors="coerce")
    canonical = pd.DataFrame({"variant_id": parsed, "effect": effect * direction}).dropna()
    canonical = (
        canonical.groupby("variant_id", as_index=False)
        .agg(effect=("effect", "median"), source_rows=("effect", "size"))
        .sort_values("variant_id")
    )
    return canonical, {
        "input_rows": len(frame),
        "valid_single_substitution_rows": int(parsed.notna().sum()),
        "retained_unique_variants": len(canonical),
        "mutation_column": mutation_column,
        "outcome_column": outcome_column,
        "direction": direction,
    }


def download_frozen_pilot_outcomes(
    protocol_path: Path,
    outcome_lock_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Access only the frozen development-pilot outcomes after method freeze."""
    _verify_predictions_frozen(outcome_lock_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PILOT_PROTOCOL_ID or protocol.get("outcomes_accessed"):
        raise ValueError("Pilot protocol is not an unopened frozen development pilot")
    protocol_dir = Path(protocol_path).parent
    tasks = pd.read_csv(protocol_dir / "pilot-task-registry.csv")
    targets = pd.read_csv(protocol_dir / "pilot-targets.csv").set_index(
        ["panel_id", "target_id"]
    )
    sources = pd.read_csv(protocol_dir / "pilot-source-manifest.csv").set_index(
        ["panel_id", "assay_id"]
    )
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outcome_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for task in tasks.sort_values(["panel_id", "assay_id"]).itertuples(index=False):
        panel_id, assay_id, target_id = map(
            str, (task.panel_id, task.assay_id, task.target_id)
        )
        source = sources.loc[(panel_id, assay_id)]
        target = targets.loc[(panel_id, target_id)]
        sequence = str(target.sequence)
        if panel_id == "mavedb-complement-v1":
            url = str(source.source_locator)
        else:
            revision = str(protocol["venus_source_revision"])
            path = quote(str(source.source_locator), safe="/")
            url = f"{VENUS_SOURCE_BASE}/{revision}/{path}?download=true"
        raw_path = raw_dir / panel_id / f"{hashlib.sha256(assay_id.encode()).hexdigest()[:20]}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        resumed = raw_path.is_file()
        payload = raw_path.read_bytes() if resumed else _fetch_bytes(url)
        if not resumed:
            raw_path.write_bytes(payload)
        frame = pd.read_csv(BytesIO(payload), low_memory=False)
        try:
            if panel_id == "mavedb-complement-v1":
                canonical, audit = canonicalize_mavedb_scores(
                    frame, sequence=sequence, orientation=int(task.direction)
                )
                canonical = canonical.rename(
                    columns={"mutation_codes": "variant_id", "DMS_score": "effect"}
                )
            else:
                canonical, audit = _canonicalize_venus(
                    frame,
                    assay_id=assay_id,
                    sequence=sequence,
                    direction=int(task.direction),
                )
        except (KeyError, TypeError, ValueError) as error:
            audit_rows.append(
                {
                    "panel_id": panel_id,
                    "assay_id": assay_id,
                    "target_id": target_id,
                    "status": "excluded_by_frozen_automatic_parser",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            ledger_rows.append(
                {
                    "panel_id": panel_id,
                    "assay_id": assay_id,
                    "source_locator": url,
                    "raw_path": str(raw_path),
                    "bytes": len(payload),
                    "sha256": sha256_file(raw_path),
                    "accessed_at_utc": _now(),
                    "access_mode": "resumed_local_copy" if resumed else "network",
                    "parse_status": "excluded",
                }
            )
            continue
        for row in canonical[["variant_id", "effect"]].itertuples(index=False):
            outcome_rows.append(
                {
                    "protocol_id": PILOT_PROTOCOL_ID,
                    "panel_id": panel_id,
                    "dataset_id": panel_id,
                    "assay_id": assay_id,
                    "target_id": target_id,
                    "variant_id": str(row.variant_id),
                    "effect": float(row.effect),
                    "direction": int(task.direction),
                }
            )
        audit_rows.append(
            {
                "panel_id": panel_id,
                "assay_id": assay_id,
                "target_id": target_id,
                "status": "parsed",
                **audit,
            }
        )
        ledger_rows.append(
            {
                "panel_id": panel_id,
                "assay_id": assay_id,
                "source_locator": url,
                "raw_path": str(raw_path),
                "bytes": len(payload),
                "sha256": sha256_file(raw_path),
                "accessed_at_utc": _now(),
                "access_mode": "resumed_local_copy" if resumed else "network",
                "parse_status": "parsed",
            }
        )
    outcomes = pd.DataFrame(outcome_rows)
    OUTCOME_SCHEMA.validate(outcomes)
    outputs = {
        "outcomes": output_dir / "pilot-outcomes.csv.gz",
        "audit": output_dir / "pilot-outcome-parsing-audit.csv",
        "ledger": output_dir / "pilot-outcome-access-ledger.json",
    }
    outcomes.to_csv(outputs["outcomes"], index=False, compression="gzip")
    write_table(pd.DataFrame(audit_rows), outputs["audit"])
    ledger = {
        "schema_version": 1,
        "protocol_id": PILOT_PROTOCOL_ID,
        "classification": "revealed development pilot; not confirmation",
        "parser_revision_status": (
            "Automatic per-assay exclusion was added after the first Venus file disclosed a "
            "target-coordinate mismatch. No outcome values or model performance were used; "
            "the entire pilot remains development evidence."
        ),
        "revealed_at_utc": _now(),
        "domainome_accessed": False,
        "nonpilot_venus_accessed": False,
        "lock_sha256_before_reveal": sha256_file(outcome_lock_path),
        "requests": ledger_rows,
        "artifacts": {
            str(outputs["outcomes"]): sha256_file(outputs["outcomes"]),
            str(outputs["audit"]): sha256_file(outputs["audit"]),
        },
    }
    outputs["ledger"].write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return outputs


def build_pilot_task_metrics(
    protocol_path: Path,
    confirmation_features_path: Path,
    prediction_registry_path: Path,
    outcomes_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Compute predeclared task-model metrics without changing the frozen selector."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PILOT_PROTOCOL_ID:
        raise ValueError("Task metrics require the frozen development-pilot protocol")
    features = pd.read_csv(confirmation_features_path)
    outcomes = pd.read_csv(outcomes_path)
    OUTCOME_SCHEMA.validate(outcomes)
    registry = pd.read_csv(prediction_registry_path)
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for row in registry.itertuples(index=False):
        frame = pd.read_csv(row.prediction_path)
        predictions[(str(row.panel_id), str(row.model_id))] = frame.loc[
            frame["status"].astype(str).eq("ok"), ["target_id", "variant_id", "score"]
        ]
    identity_columns = [
        "protocol_id", "panel_id", "dataset_id", "assay_id", "target_id",
        "protein_id", "family_id", "model_id",
    ]
    identities = features.loc[:, identity_columns].drop_duplicates()
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for identity in identities.itertuples(index=False):
        task_outcomes = outcomes.loc[
            outcomes["panel_id"].astype(str).eq(str(identity.panel_id))
            & outcomes["assay_id"].astype(str).eq(str(identity.assay_id)),
            ["target_id", "variant_id", "effect"],
        ]
        model_predictions = predictions[(str(identity.panel_id), str(identity.model_id))]
        aligned = task_outcomes.merge(
            model_predictions.loc[
                model_predictions["target_id"].astype(str).eq(str(identity.target_id)),
                ["variant_id", "score"],
            ],
            on="variant_id",
            how="inner",
            validate="one_to_one",
        ).dropna()
        status = "included" if len(aligned) >= 10 else "too_few_aligned_variants"
        audit_rows.append(
            {
                **{column: getattr(identity, column) for column in identity_columns},
                "aligned_variants": len(aligned),
                "status": status,
            }
        )
        if status != "included":
            continue
        observed = aligned["effect"].to_numpy(dtype=float)
        predicted = aligned["score"].to_numpy(dtype=float)
        metrics = top_selection_metrics(observed, predicted, fraction=0.1)
        rows.append(
            {
                **{column: getattr(identity, column) for column in identity_columns},
                "aligned_variants": len(aligned),
                "spearman": regression_metrics(observed, predicted).spearman,
                **metrics,
            }
        )
    metrics = pd.DataFrame(rows)
    TASK_METRIC_SCHEMA.validate(metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics": output_dir / "pilot-task-metrics.csv",
        "audit": output_dir / "pilot-task-metric-audit.csv",
        "manifest": output_dir / "pilot-metric-manifest.json",
    }
    write_table(metrics, outputs["metrics"])
    write_table(pd.DataFrame(audit_rows), outputs["audit"])
    manifest = {
        "schema_version": 1,
        "classification": "development pilot; not confirmation",
        "tasks": int(metrics["assay_id"].nunique()),
        "models": int(metrics["model_id"].nunique()),
        "rows": len(metrics),
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                protocol_path, confirmation_features_path,
                prediction_registry_path, outcomes_path,
            ]
        },
        "artifacts": {
            str(path): sha256_file(path)
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return outputs


def evaluate_pilot_signal(
    bundle_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Evaluate only the two decision-critical pilot contrasts with full resampling."""
    import joblib

    from .transportability import (
        TransportConfig,
        bootstrap_policy_difference,
        selective_policy_curve,
        summarize_policy_curves,
    )

    bundle = joblib.load(bundle_path)
    config = TransportConfig.from_dict(dict(bundle["config"]))
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)
    join_columns = [
        "protocol_id", "panel_id", "dataset_id", config.assay_column, "target_id",
        config.model_column, config.group_column, config.protein_column,
    ]
    outcome_columns = [column for column in metrics.columns if column not in predictions.columns]
    merged = predictions.merge(
        metrics.loc[:, [*join_columns, *outcome_columns]],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Pilot predictions and task metrics have no aligned rows")
    comparator = str(bundle["best_label_free_comparator"])
    best_model = str(bundle["best_average_model"])
    policies = ["variantshift", comparator, "always_best", "oracle", "random"]
    policies = list(dict.fromkeys(policies))
    curves = pd.concat(
        [
            selective_policy_curve(
                merged,
                config,
                policy=policy,
                best_average_model=best_model,
            )
            for policy in policies
        ],
        ignore_index=True,
    )
    summary = summarize_policy_curves(curves)
    bootstrap_frames = []
    bootstrap_rows = []
    for baseline in dict.fromkeys([comparator, "always_best"]):
        replicates, row = bootstrap_policy_difference(
            merged,
            config,
            comparator=baseline,
            best_average_model=best_model,
        )
        row["one_sided_p_value"] = float(
            (1 + np.sum(replicates["regret_coverage_auc_improvement"] <= 0))
            / (len(replicates) + 1)
        )
        bootstrap_frames.append(replicates)
        bootstrap_rows.append(row)
    bootstrap_summary = pd.DataFrame(bootstrap_rows)
    task_model_summary = (
        merged.groupby(["panel_id", config.model_column], as_index=False)
        .agg(
            tasks=(config.task_column, "nunique"),
            mean_selection_gain_sd=(config.target_column, "mean"),
            failure_rate=(config.target_column, lambda values: float((values <= 0).mean())),
            mean_spearman=("spearman", "mean"),
        )
        .sort_values(["panel_id", "mean_selection_gain_sd"], ascending=[True, False])
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "merged": output_dir / "pilot-task-predictions.csv",
        "curves": output_dir / "pilot-risk-coverage.csv",
        "summary": output_dir / "pilot-policy-summary.csv",
        "bootstrap": output_dir / "pilot-primary-bootstrap.csv.gz",
        "bootstrap_summary": output_dir / "pilot-primary-bootstrap-summary.csv",
        "model_summary": output_dir / "pilot-model-summary.csv",
        "manifest": output_dir / "pilot-signal-manifest.json",
    }
    write_table(merged, outputs["merged"])
    write_table(curves, outputs["curves"])
    write_table(summary, outputs["summary"])
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        outputs["bootstrap"], index=False, compression="gzip"
    )
    write_table(bootstrap_summary, outputs["bootstrap_summary"])
    write_table(task_model_summary, outputs["model_summary"])
    manifest = {
        "schema_version": 1,
        "classification": "development pilot; not confirmation",
        "frozen_comparator": comparator,
        "best_fixed_model": best_model,
        "tasks": int(merged[config.task_column].nunique()),
        "models": int(merged[config.model_column].nunique()),
        "bootstrap_repeats": config.bootstrap_repeats,
        "inputs": {
            str(path): sha256_file(path)
            for path in [bundle_path, predictions_path, metrics_path]
        },
        "artifacts": {
            str(path): sha256_file(path)
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return outputs
