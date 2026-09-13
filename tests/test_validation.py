import json

import pandas as pd

from research.validation import validate_research_outputs, write_validation_report


def _write_valid_run(root) -> None:
    results = root / "results"
    results.mkdir()
    (results / "backtest_manifest.json").write_text(
        json.dumps(
            {
                "data": {"end_date": "2026-01-05"},
                "validation": {
                    "chronological_signal_lag": True,
                    "future_price_backfill": False,
                    "warnings": [],
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "total_value": [100000.0, 100100.0],
            "daily_return": [0.0, 0.001],
        }
    ).to_csv(results / "portfolio_value.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2026-01-05"],
            "ticker": ["AAPL"],
            "side": ["BUY"],
            "quantity": [10],
            "executed_price": [100.0],
            "notional": [1000.0],
        }
    ).to_csv(results / "trade_history.csv", index=False)
    pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "duplicate_records": [0],
            "missing_price_values": [0],
            "invalid_dates": [0],
            "non_positive_prices": [0],
        }
    ).to_csv(results / "data_quality_report.csv", index=False)
    pd.DataFrame({"metric": ["sharpe_ratio"], "value": [1.0]}).to_csv(
        results / "performance_summary.csv", index=False
    )


def test_validation_passes_and_writes_a_report(tmp_path) -> None:
    _write_valid_run(tmp_path)

    report = validate_research_outputs(tmp_path)
    output = write_validation_report(report, tmp_path / "results" / "research_validation.json")

    assert report["status"] == "passed"
    assert output.exists()


def test_validation_fails_on_unsorted_portfolio_dates(tmp_path) -> None:
    _write_valid_run(tmp_path)
    portfolio_path = tmp_path / "results" / "portfolio_value.csv"
    frame = pd.read_csv(portfolio_path).iloc[::-1]
    frame.to_csv(portfolio_path, index=False)

    report = validate_research_outputs(tmp_path)

    assert report["status"] == "failed"
    assert any("dates must be increasing" in error for error in report["errors"])
