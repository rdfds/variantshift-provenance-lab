"""Freeze outcome-blind confirmation tasks, directions, and exclusions."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .outcome_lock import assert_target_only
from .provenance import sha256_file
from .schemas import validate_targets, write_table

_KM_TOKEN = re.compile(r"(?:^|_)km(?:_|$)", flags=re.IGNORECASE)


def _normalize_doi(value: str) -> str:
    text = str(value).strip().lstrip("?")
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi/", "", text, flags=re.IGNORECASE)
    return text.strip()


def venus_direction(dataset_id: str, category: str) -> tuple[int | None, str]:
    """Set a direction from assay semantics, never from observed measurements."""
    if str(category) == "selectivity":
        return None, "signed ee/de selectivity is not universally ordered"
    if str(category) != "activity":
        return None, "unsupported assay category"
    if _KM_TOKEN.search(str(dataset_id)):
        return -1, "lower Michaelis constant is favorable"
    return 1, "higher activity, turnover, or catalytic efficiency is favorable"


def freeze_confirmation_task_registry(
    domainome_targets_path: Path,
    mavedb_score_audit_path: Path,
    venus_assay_audit_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Freeze task inclusion and direction rules before any outcome reveal."""
    domainome = validate_targets(pd.read_csv(domainome_targets_path))
    assert_target_only(domainome)
    mavedb = pd.read_csv(mavedb_score_audit_path)
    venus = pd.read_csv(venus_assay_audit_path)
    rows: list[dict[str, object]] = []
    for target in domainome.sort_values("target_id").itertuples(index=False):
        rows.append(
            {
                "panel_id": str(target.panel_id),
                "assay_id": str(target.target_id),
                "target_id": str(target.target_id),
                "assay_modality": "abundance/stability",
                "direction": 1,
                "direction_source": "Domainome abundance-score definition",
                "included": True,
                "exclusion_reason": "",
                "publication_ids": "",
                "outcomes_accessed": False,
            }
        )
    for assay in mavedb.sort_values("urn").itertuples(index=False):
        selected = bool(assay.selected)
        direction = int(assay.metadata_orientation) if selected else None
        reasons = str(assay.exclusion_reasons or "") if not pd.isna(assay.exclusion_reasons) else ""
        rows.append(
            {
                "panel_id": "mavedb-complement-v1",
                "assay_id": str(assay.urn),
                "target_id": str(assay.target_id) if selected else "",
                "assay_modality": "multiplexed variant effect",
                "direction": direction,
                "direction_source": "frozen MaveDB metadata orientation" if selected else "",
                "included": selected,
                "exclusion_reason": reasons,
                "publication_ids": str(assay.publication_dois or "")
                if not pd.isna(assay.publication_dois)
                else "",
                "outcomes_accessed": False,
            }
        )
    for assay in venus.sort_values("dataset_id").itertuples(index=False):
        selected = bool(assay.selected)
        direction, rule = venus_direction(str(assay.dataset_id), str(assay.category))
        reasons = []
        if not selected and isinstance(assay.exclusion_reasons, str):
            reasons.extend(filter(None, assay.exclusion_reasons.split(";")))
        if selected and direction is None:
            reasons.append("ambiguous_score_direction")
        included = selected and direction is not None
        rows.append(
            {
                "panel_id": "venusmuthub-v1",
                "assay_id": str(assay.dataset_id),
                "target_id": str(assay.target_id) if included else "",
                "assay_modality": str(assay.category),
                "direction": direction if included else None,
                "direction_source": rule if included else "",
                "included": included,
                "exclusion_reason": ";".join(sorted(set(reasons))),
                "publication_ids": str(assay.doi_normalized or "")
                if not pd.isna(assay.doi_normalized)
                else "",
                "outcomes_accessed": False,
            }
        )
    registry = pd.DataFrame(rows)
    included = registry.loc[registry["included"].astype(bool)]
    if included["direction"].isna().any():
        raise RuntimeError("Every included confirmation task must have a frozen direction")
    if included[["panel_id", "assay_id"]].duplicated().any():
        raise ValueError("Confirmation task identifiers are not unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "registry": output_dir / "confirmation-task-registry.csv",
        "summary": output_dir / "confirmation-task-summary.csv",
        "manifest": output_dir / "confirmation-task-manifest.json",
    }
    write_table(registry, outputs["registry"])
    summary = (
        registry.groupby(["panel_id", "included"], dropna=False)
        .size()
        .rename("task_count")
        .reset_index()
    )
    write_table(summary, outputs["summary"])
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "included_tasks": int(registry["included"].sum()),
        "included_tasks_by_panel": {
            str(panel): int(count)
            for panel, count in included.groupby("panel_id").size().items()
        },
        "direction_policy": {
            "domainome": "higher abundance score is favorable",
            "mavedb": "registry metadata orientation frozen during target acquisition",
            "venus_activity": "higher is favorable except Km, where lower is favorable",
            "venus_selectivity": "excluded because signed ee/de values are not universally ordered",
        },
        "inputs": {
            str(domainome_targets_path): sha256_file(domainome_targets_path),
            str(mavedb_score_audit_path): sha256_file(mavedb_score_audit_path),
            str(venus_assay_audit_path): sha256_file(venus_assay_audit_path),
        },
        "artifacts": {
            outputs["registry"].name: sha256_file(outputs["registry"]),
            outputs["summary"].name: sha256_file(outputs["summary"]),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def freeze_untouched_confirmation_registry(
    full_registry_path: Path,
    untouched_external_registry_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Freeze Domainome plus only the explicitly partitioned untouched external tasks."""
    full = pd.read_csv(full_registry_path)
    untouched = pd.read_csv(untouched_external_registry_path)
    domainome = full.loc[
        full["panel_id"].astype(str).eq("human-domainome-v1")
        & full["included"].astype(bool)
    ].copy()
    external = untouched.loc[untouched["included"].astype(bool)].copy()
    registry = pd.concat([domainome, external], ignore_index=True, sort=False)
    if registry.empty:
        raise ValueError("Untouched confirmation registry is empty")
    if registry["outcomes_accessed"].astype(bool).any():
        raise ValueError("Untouched confirmation registry contains an accessed task")
    if registry[["panel_id", "assay_id"]].duplicated().any():
        raise ValueError("Untouched confirmation task identifiers are not unique")
    expected_panels = {
        "human-domainome-v1",
        "mavedb-complement-v1",
        "venusmuthub-v1",
    }
    observed_panels = set(registry["panel_id"].astype(str))
    if observed_panels != expected_panels:
        raise ValueError(
            f"Untouched confirmation registry has unexpected panels: {sorted(observed_panels)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "registry": output_dir / "untouched-confirmation-task-registry.csv",
        "summary": output_dir / "untouched-confirmation-task-summary.csv",
        "manifest": output_dir / "untouched-confirmation-task-manifest.json",
    }
    registry = registry.sort_values(["panel_id", "assay_id"]).reset_index(drop=True)
    write_table(registry, outputs["registry"])
    summary = (
        registry.groupby("panel_id")
        .agg(tasks=("assay_id", "size"), targets=("target_id", "nunique"))
        .reset_index()
    )
    write_table(summary, outputs["summary"])
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "partition_policy": (
            "All frozen Domainome tasks plus only tasks explicitly assigned to the untouched "
            "external holdout; development-pilot tasks are excluded."
        ),
        "tasks": len(registry),
        "tasks_by_panel": {
            str(row.panel_id): int(row.tasks) for row in summary.itertuples(index=False)
        },
        "targets_by_panel": {
            str(row.panel_id): int(row.targets) for row in summary.itertuples(index=False)
        },
        "inputs": {
            str(full_registry_path): sha256_file(full_registry_path),
            str(untouched_external_registry_path): sha256_file(
                untouched_external_registry_path
            ),
        },
        "artifacts": {
            outputs["registry"].name: sha256_file(outputs["registry"]),
            outputs["summary"].name: sha256_file(outputs["summary"]),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def freeze_confirmation_publication_dates(
    mavedb_score_audit_path: Path,
    venus_assay_audit_path: Path,
    cache_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Freeze earliest public dates from registry metadata and Crossref only."""
    rows = [
        {
            "panel_id": "human-domainome-v1",
            "target_id": "*",
            "publication_date": "2024-12-10",
            "date_source": "Zenodo record 14356805 publication_date",
            "metadata_sha256": "",
        }
    ]
    mavedb = pd.read_csv(mavedb_score_audit_path)
    selected_mavedb = mavedb.loc[mavedb["selected"].astype(bool)].copy()
    for target_id, group in selected_mavedb.groupby("target_id"):
        rows.append(
            {
                "panel_id": "mavedb-complement-v1",
                "target_id": str(target_id),
                "publication_date": str(pd.to_datetime(group["published_date"]).min().date()),
                "date_source": "MaveDB frozen registry metadata",
                "metadata_sha256": sha256(
                    "|".join(sorted(group["detail_metadata_sha256"].astype(str))).encode()
                ).hexdigest(),
            }
        )
    venus = pd.read_csv(venus_assay_audit_path)
    venus = venus.loc[venus["selected"].astype(bool)].copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates: dict[str, tuple[str, str]] = {}
    errors: dict[str, str] = {}
    doi_values = sorted(
        {_normalize_doi(value) for value in venus["doi_normalized"].dropna().astype(str)}
    )
    for doi in doi_values:
        cache_path = cache_dir / f"{sha256(doi.encode()).hexdigest()[:20]}.json"
        try:
            if cache_path.is_file():
                payload = cache_path.read_bytes()
            else:
                time.sleep(2)
                request = urllib.request.Request(
                    f"https://api.crossref.org/works/{quote(doi, safe='')}",
                    headers={"User-Agent": "VariantShift/1.0 publication-date audit"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                cache_path.write_bytes(payload)
            message = json.loads(payload)["message"]
            date_parts = None
            source = ""
            for field in ("published-print", "published-online", "published", "issued"):
                parts = (message.get(field) or {}).get("date-parts") or []
                if parts:
                    date_parts = parts[0]
                    source = f"Crossref {field}"
                    break
            if not date_parts:
                raise ValueError("Crossref record has no publication date")
            year, month, day = (*date_parts, 1, 1)[:3]
            dates[doi] = (f"{int(year):04d}-{int(month):02d}-{int(day):02d}", source)
        except Exception as exception:  # noqa: BLE001 - audit missing Crossref metadata
            errors[doi] = f"{type(exception).__name__}: {exception}"
    for target_id, group in venus.groupby("target_id"):
        dois = [_normalize_doi(doi) for doi in group["doi_normalized"]]
        target_dates = [dates[doi][0] for doi in dois if doi in dates]
        target_sources = sorted(
            {dates[doi][1] for doi in dois if doi in dates}
        )
        target_errors = sorted({errors[doi] for doi in dois if doi in errors})
        rows.append(
            {
                "panel_id": "venusmuthub-v1",
                "target_id": str(target_id),
                "publication_date": min(target_dates) if target_dates else "",
                "date_source": ";".join(target_sources),
                "metadata_sha256": sha256(
                    "|".join(sorted(dois)).encode()
                ).hexdigest(),
                "error": ";".join(target_errors),
            }
        )
    frame = pd.DataFrame(rows)
    write_table(frame, output_path)
    manifest = {
        "schema_version": 1,
        "outcomes_accessed": False,
        "targets": len(frame) - 1 + 426,
        "targets_with_date": 426
        + int(
            frame.loc[frame["target_id"].ne("*"), "publication_date"]
            .fillna("")
            .ne("")
            .sum()
        ),
        "crossref_errors": len(errors),
        "inputs": {
            str(mavedb_score_audit_path): sha256_file(mavedb_score_audit_path),
            str(venus_assay_audit_path): sha256_file(venus_assay_audit_path),
        },
        "artifact": {str(output_path): sha256_file(output_path)},
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
