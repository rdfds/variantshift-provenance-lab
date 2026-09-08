"""Auditable external-compute budget guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256_file


def check_compute_budget(
    ledger_path: Path,
    *,
    planned_cost_usd: float = 0.0,
    hard_cap_usd: float = 2_000.0,
) -> dict[str, object]:
    ledger_path = Path(ledger_path)
    ledger = pd.read_csv(ledger_path)
    required = {"job_id", "provider", "actual_cost_usd", "status"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Compute ledger is missing columns: {missing}")
    if ledger["job_id"].astype(str).duplicated().any():
        raise ValueError("Compute ledger job_id values must be unique")
    costs = pd.to_numeric(ledger["actual_cost_usd"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Compute costs must be finite and nonnegative")
    if not np.isfinite(planned_cost_usd) or planned_cost_usd < 0:
        raise ValueError("Planned cost must be finite and nonnegative")
    if not np.isfinite(hard_cap_usd) or hard_cap_usd <= 0:
        raise ValueError("Hard cap must be finite and positive")
    spent = float(costs.sum())
    projected = spent + float(planned_cost_usd)
    fraction = projected / hard_cap_usd
    alert = "none"
    for threshold, label in ((0.90, "90%"), (0.75, "75%"), (0.50, "50%")):
        if fraction >= threshold:
            alert = label
            break
    permitted = projected <= hard_cap_usd
    return {
        "schema_version": 1,
        "ledger_sha256": sha256_file(ledger_path),
        "jobs_recorded": len(ledger),
        "spent_usd": spent,
        "planned_cost_usd": float(planned_cost_usd),
        "projected_total_usd": projected,
        "hard_cap_usd": float(hard_cap_usd),
        "alert_threshold_reached": alert,
        "permitted": permitted,
        "status": "within_budget" if permitted else "blocked_hard_cap",
    }


def write_budget_report(report: dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
