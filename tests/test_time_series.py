import pandas as pd

from models.time_series import rolling_volatility, trend_signal


def test_rolling_volatility_length_matches_input() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01])
    result = rolling_volatility(returns, window=3)
    assert len(result) == len(returns)
    assert result.iloc[-1] >= 0


def test_trend_signal_is_binary() -> None:
    prices = pd.Series(range(1, 100))
    signal = trend_signal(prices, short_window=5, long_window=20)
    assert set(signal.dropna().unique()).issubset({0, 1})
