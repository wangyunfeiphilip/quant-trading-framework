from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.refresh_public_project_data import RefreshValidationError, validate_refresh_outputs


EXPECTED = {"AAPL", "MSFT"}
EXPECTED_LATEST = pd.Timestamp("2026-09-11")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for ticker in sorted(EXPECTED):
        for day in pd.date_range("2026-09-10", "2026-09-11", freq="B"):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "close": 100.0,
                    "adjusted_close": 100.0,
                }
            )
    feature = pd.DataFrame(rows)
    clean = feature.copy()
    quality = pd.DataFrame(
        {
            "ticker": sorted(EXPECTED),
            "rows": [2, 2],
            "end_date": ["2026-09-11", "2026-09-11"],
            "duplicate_records": [0, 0],
            "invalid_dates": [0, 0],
        }
    )
    return feature, clean, quality


def test_valid_refresh_is_accepted() -> None:
    feature, clean, quality = _frames()
    metrics = validate_refresh_outputs(feature, clean, quality, EXPECTED, EXPECTED_LATEST)
    assert metrics.latest_date.isoformat() == "2026-09-11"
    assert metrics.ticker_count == 2


def test_missing_ticker_is_rejected_before_replacement() -> None:
    feature, clean, quality = _frames()
    feature = feature[feature["ticker"] != "MSFT"]
    with pytest.raises(RefreshValidationError, match="ticker coverage invalid"):
        validate_refresh_outputs(feature, clean, quality, EXPECTED, EXPECTED_LATEST)


def test_stale_or_duplicate_candidate_is_rejected() -> None:
    feature, clean, quality = _frames()
    feature.loc[feature.index[-1], "date"] = pd.Timestamp("2026-09-10")
    with pytest.raises(RefreshValidationError):
        validate_refresh_outputs(feature, clean, quality, EXPECTED, EXPECTED_LATEST)

    feature, clean, quality = _frames()
    feature = pd.concat([feature, feature.iloc[[0]]], ignore_index=True)
    with pytest.raises(RefreshValidationError, match="duplicate"):
        validate_refresh_outputs(feature, clean, quality, EXPECTED, EXPECTED_LATEST)


def test_suspicious_row_loss_is_rejected() -> None:
    feature, clean, quality = _frames()
    previous = pd.concat([feature] * 10, ignore_index=True)
    with pytest.raises(RefreshValidationError, match="suspiciously low"):
        validate_refresh_outputs(feature, clean, quality, EXPECTED, EXPECTED_LATEST, previous)


def test_public_refresh_workflow_is_schedule_and_dispatch_only() -> None:
    workflow = Path(__file__).parents[1] / ".github/workflows/daily-public-project-refresh.yml"
    text = workflow.read_text(encoding="utf-8")
    assert 'cron: "0 23 * * 1-5"' in text
    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "contents: write" in text
    assert "ref: math-finance-upgrade" in text
    assert "git push origin \"HEAD:$TARGET_BRANCH\"" in text
    assert "--force" not in text
