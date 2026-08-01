import numpy as np
import pandas as pd

from research.market_monitor import analyze_ticker


def test_analyze_ticker_returns_snapshot() -> None:
    dates = pd.date_range("2025-01-01", periods=150, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "NVDA",
            "adjusted_close": np.linspace(100, 160, len(dates)),
        }
    )

    snapshot = analyze_ticker(prices, "NVDA")
    assert snapshot.ticker == "NVDA"
    assert snapshot.last_price == 160.0
    assert snapshot.annualized_volatility >= 0
