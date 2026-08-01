import numpy as np
import pandas as pd

from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=160, freq="B")
    rows = []
    for ticker, start, end in [("AAPL", 100, 150), ("MSFT", 100, 120), ("NVDA", 100, 180)]:
        for date, price in zip(dates, np.linspace(start, end, len(dates))):
            rows.append({"date": date, "ticker": ticker, "adjusted_close": price})
    return pd.DataFrame(rows)


def test_momentum_signal_generation() -> None:
    weights = generate_momentum_weights(_prices(), top_n=2)
    assert not weights.empty
    assert weights["target_weight"].between(0, 1).all()


def test_mean_reversion_buy_sell_logic_columns() -> None:
    weights = generate_mean_reversion_weights(_prices(), window=20)
    assert {"BUY", "SELL", "HOLD"}.intersection(set(weights["signal"]))
