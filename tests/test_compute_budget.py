import pandas as pd

from variantshift.compute_budget import check_compute_budget


def test_compute_budget_enforces_hard_cap_and_alerts(tmp_path) -> None:
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame(
        {
            "job_id": ["job-1", "job-2"],
            "provider": ["cloud", "cloud"],
            "actual_cost_usd": [700.0, 500.0],
            "status": ["complete", "complete"],
        }
    ).to_csv(ledger, index=False)
    allowed = check_compute_budget(ledger, planned_cost_usd=300.0)
    assert allowed["permitted"]
    assert allowed["alert_threshold_reached"] == "75%"
    blocked = check_compute_budget(ledger, planned_cost_usd=801.0)
    assert not blocked["permitted"]
    assert blocked["status"] == "blocked_hard_cap"
