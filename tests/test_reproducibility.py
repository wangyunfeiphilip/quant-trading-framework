import json

import pandas as pd

from research.reproducibility import build_backtest_manifest, sha256_file, write_backtest_manifest


def test_backtest_manifest_records_data_cutoff_and_assumptions(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "ticker": ["AAPL", "MSFT"],
            "adjusted_close": [100.0, 200.0],
        }
    )
    artifact = tmp_path / "feature_dataset.csv"
    artifact.write_text("date,ticker,adjusted_close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    config = {
        "data": {"start": "2026-01-01"},
        "strategy": {"name": "momentum", "top_n": 3},
        "backtest": {"signal_lag": 1, "transaction_cost_bps": 5, "slippage_bps": 2},
    }

    manifest = build_backtest_manifest(
        config=config,
        features=features,
        artifact_paths=[artifact],
        project_root=tmp_path,
    )
    output = write_backtest_manifest(manifest, tmp_path / "backtest_manifest.json")

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["data"]["end_date"] == "2026-01-05"
    assert loaded["data"]["ticker_count"] == 2
    assert loaded["backtest"]["signal_lag_sessions"] == 1
    assert loaded["validation"]["future_price_backfill"] is False
    assert loaded["artifacts"]["feature_dataset.csv"]["sha256"] == sha256_file(artifact)
    assert "python" in loaded["runtime"]
