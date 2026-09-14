from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.refresh_public_project_data as refresh_cli
from scripts.refresh_public_project_data import (
    RefreshValidationError,
    publish_csv_if_changed,
    preserve_published_fundamentals,
    public_csv_changed,
    validate_refresh_outputs,
)


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


def test_identical_candidate_is_not_rewritten_and_reports_no_change(tmp_path: Path) -> None:
    feature, _, _ = _frames()
    existing_path = tmp_path / "existing.csv"
    candidate_path = tmp_path / "candidate.csv"
    feature.to_csv(existing_path, index=False, lineterminator="\n")
    feature.iloc[::-1][list(reversed(feature.columns))].to_csv(candidate_path, index=False, lineterminator="\r\n")
    before = existing_path.read_bytes()

    assert public_csv_changed(candidate_path, existing_path, ["date", "ticker"]) is False
    assert publish_csv_if_changed(candidate_path, existing_path, ["date", "ticker"]) is False
    assert existing_path.read_bytes() == before


def test_real_numeric_change_and_new_date_are_detected(tmp_path: Path) -> None:
    feature, _, _ = _frames()
    existing_path = tmp_path / "existing.csv"
    candidate_path = tmp_path / "candidate.csv"
    feature.to_csv(existing_path, index=False)

    changed = feature.copy()
    changed.loc[0, "close"] = 100.5
    changed.to_csv(candidate_path, index=False)
    assert public_csv_changed(candidate_path, existing_path, ["date", "ticker"]) is True

    new_date = pd.concat(
        [feature, feature.iloc[[0]].assign(date=pd.Timestamp("2026-09-12"))],
        ignore_index=True,
    )
    new_date.to_csv(candidate_path, index=False)
    assert public_csv_changed(candidate_path, existing_path, ["date", "ticker"]) is True


def test_published_fundamentals_stay_stable_and_new_rows_use_current_values() -> None:
    previous, _, _ = _frames()
    previous = previous.assign(
        pe_ratio=10.0,
        pb_ratio=2.0,
        market_cap=1_000_000.0,
        roe=0.10,
        revenue_growth=0.05,
    )
    candidate = previous.copy()
    for column, value in {
        "pe_ratio": 20.0,
        "pb_ratio": 4.0,
        "market_cap": 2_000_000.0,
        "roe": 0.20,
        "revenue_growth": 0.15,
    }.items():
        candidate[column] = value
    new_rows = candidate.iloc[[0]].assign(date=pd.Timestamp("2026-09-12"))
    stabilized = preserve_published_fundamentals(
        pd.concat([candidate, new_rows], ignore_index=True),
        previous,
    )

    existing_rows = stabilized[stabilized["date"] < pd.Timestamp("2026-09-12")]
    new_row = stabilized[stabilized["date"] == pd.Timestamp("2026-09-12")].iloc[0]
    assert existing_rows["pe_ratio"].eq(10.0).all()
    assert existing_rows["pb_ratio"].eq(2.0).all()
    assert existing_rows["market_cap"].eq(1_000_000.0).all()
    assert existing_rows["roe"].eq(0.10).all()
    assert existing_rows["revenue_growth"].eq(0.05).all()
    assert new_row["pe_ratio"] == 20.0
    assert new_row["pb_ratio"] == 4.0
    assert new_row["market_cap"] == 2_000_000.0
    assert new_row["roe"] == 0.20
    assert new_row["revenue_growth"] == 0.15


def test_changed_fundamentals_alone_do_not_change_public_snapshot(tmp_path: Path) -> None:
    previous, clean, quality = _frames()
    previous = previous.assign(
        pe_ratio=10.0,
        pb_ratio=2.0,
        market_cap=1_000_000.0,
        roe=0.10,
        revenue_growth=0.05,
    )
    candidate = previous.copy()
    candidate[["pe_ratio", "pb_ratio", "market_cap", "roe", "revenue_growth"]] = [
        20.0,
        4.0,
        2_000_000.0,
        0.20,
        0.15,
    ]
    existing_path = tmp_path / "existing.csv"
    candidate_path = tmp_path / "candidate.csv"
    previous.to_csv(existing_path, index=False)
    stabilized = preserve_published_fundamentals(candidate, previous)
    stabilized.to_csv(candidate_path, index=False)

    assert public_csv_changed(candidate_path, existing_path, ["date", "ticker"]) is False
    assert stabilized["close"].equals(clean["close"])
    assert len(quality) == len(EXPECTED)


def test_changed_csv_uses_canonical_columns_floats_and_line_endings(tmp_path: Path) -> None:
    feature, _, _ = _frames()
    existing_path = tmp_path / "existing.csv"
    candidate_path = tmp_path / "candidate.csv"
    feature.to_csv(existing_path, index=False)
    changed = feature.iloc[::-1][list(reversed(feature.columns))].copy()
    changed.loc[0, "close"] = 100.25
    changed.to_csv(candidate_path, index=False, lineterminator="\r\n")

    assert publish_csv_if_changed(candidate_path, existing_path, ["date", "ticker"]) is True
    output = existing_path.read_bytes()
    assert b"\r\n" not in output
    assert output.splitlines()[0].decode() == ",".join(feature.columns)


def test_identical_cli_refresh_reports_no_change_and_preserves_bytes(tmp_path: Path, monkeypatch, capsys) -> None:
    feature, clean, quality = _frames()
    feature = feature.assign(
        pe_ratio=10.0,
        pb_ratio=2.0,
        market_cap=1_000_000.0,
        roe=0.10,
        revenue_growth=0.05,
    )
    refreshed_feature = feature.copy()
    refreshed_feature[["pe_ratio", "pb_ratio", "market_cap", "roe", "revenue_growth"]] = [
        20.0,
        4.0,
        2_000_000.0,
        0.20,
        0.15,
    ]
    (tmp_path / "config.yaml").write_text(
        "universe:\n  - AAPL\n  - MSFT\ndata:\n  start: '2023-01-01'\n",
        encoding="utf-8",
    )
    public_root = tmp_path / "demo_data"
    (public_root / "data/processed").mkdir(parents=True)
    (public_root / "README.md").write_text(
        "The equity feature snapshot covers 2023-01-03 through 2026-09-11;\n",
        encoding="utf-8",
    )
    existing_feature = public_root / "data/processed/feature_dataset.csv"
    existing_quality = public_root / "data/processed/data_quality_report.csv"
    feature.to_csv(existing_feature, index=False, lineterminator="\n")
    quality.to_csv(existing_quality, index=False, lineterminator="\n")
    before_feature = existing_feature.read_bytes()
    before_quality = existing_quality.read_bytes()

    def fake_refresh(temporary_root: Path, _tickers, _start) -> None:
        processed = temporary_root / "data/processed"
        processed.mkdir(parents=True)
        refreshed_feature.iloc[::-1][list(reversed(refreshed_feature.columns))].to_csv(
            processed / "feature_dataset.csv", index=False, lineterminator="\r\n"
        )
        clean.to_csv(processed / "clean_stock_data.csv", index=False)
        quality.iloc[::-1].to_csv(processed / "data_quality_report.csv", index=False, lineterminator="\r\n")

    monkeypatch.setattr(refresh_cli, "refresh_project_market_data", fake_refresh)
    monkeypatch.setattr(refresh_cli, "expected_latest_market_date", lambda: EXPECTED_LATEST)

    metrics = refresh_cli.update_public_snapshot(tmp_path)

    assert metrics.data_changed is False
    assert existing_feature.read_bytes() == before_feature
    assert existing_quality.read_bytes() == before_quality

    second_metrics = refresh_cli.update_public_snapshot(tmp_path)

    assert second_metrics.data_changed is False
    assert existing_feature.read_bytes() == before_feature
    assert existing_quality.read_bytes() == before_quality
    assert capsys.readouterr().out.count("Data changed: no") == 2
