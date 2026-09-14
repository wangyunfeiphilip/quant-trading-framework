from __future__ import annotations

import pandas as pd

from investment_platform.trades import normalize_trade_frame, reconstruct_from_trades


def test_normalize_trade_frame_accepts_common_ibkr_aliases() -> None:
    raw = pd.DataFrame(
        {
            "Trade Date": ["2026-01-02", "2026-01-05"],
            "Symbol": ["US.NVDA", "NVDA"],
            "Action": ["BUY", "SELL"],
            "Qty": [10, 2],
            "Exec Price": [100, 110],
            "Commission": [1, 1],
        }
    )

    trades = normalize_trade_frame(raw)

    assert trades["ticker"].tolist() == ["NVDA", "NVDA"]
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert trades["quantity"].tolist() == [10, 2]
    assert trades["price"].tolist() == [100, 110]


def test_reconstruct_from_trades_builds_value_and_pnl_contribution() -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"] * 2),
            "ticker": ["NVDA", "NVDA", "NVDA", "SPY", "SPY", "SPY"],
            "close": [100.0, 110.0, 120.0, 500.0, 505.0, 510.0],
            "adjusted_close": [100.0, 110.0, 120.0, 500.0, 505.0, 510.0],
            "daily_return": [0.0, 0.10, 0.0909, 0.0, 0.01, 0.0099],
        }
    )
    trades = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "ticker": ["NVDA", "NVDA"],
            "side": ["BUY", "SELL"],
            "quantity": [10, 2],
            "price": [100.0, 110.0],
        }
    )

    result = reconstruct_from_trades(features, trades, initial_cash=2_000.0)

    assert result.portfolio_value["total_value"].iloc[-1] == 2_180.0
    assert result.contribution.iloc[0]["ticker"] == "NVDA"
    assert result.contribution.iloc[0]["quantity"] == 8
    assert result.contribution.iloc[0]["avg_cost"] == 100.0
    assert result.contribution.iloc[0]["unrealized_pnl"] == 160.0
    assert result.contribution.iloc[0]["realized_pnl"] == 20.0


def test_normalize_trade_frame_accepts_ibkr_activity_statement_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "DataDiscriminator": "Order",
                "Asset Category": "Stocks",
                "Currency": "USD",
                "Symbol": "US.AVGO",
                "Date/Time": "2026-08-21, 09:31:04",
                "Buy/Sell": "BUY",
                "Quantity": "2",
                "T. Price": "310.50",
                "Comm/Fee": "-1.00",
            },
            {
                "DataDiscriminator": "Summary",
                "Asset Category": "Forex",
                "Currency": "USD",
                "Symbol": "USD",
                "Date/Time": "2026-08-21",
                "Buy/Sell": "",
                "Quantity": "0",
                "T. Price": "1",
                "Comm/Fee": "0",
            },
        ]
    )

    normalized = normalize_trade_frame(frame)

    assert len(normalized) == 1
    assert normalized.loc[0, "ticker"] == "AVGO"
    assert normalized.loc[0, "side"] == "BUY"
    assert normalized.loc[0, "price"] == 310.5
    assert normalized.loc[0, "commission"] == 1.0
