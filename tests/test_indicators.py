import numpy as np
import pandas as pd

from indicators.technical_indicators import bollinger_bands, ema, macd, rsi, sma


def test_sma_matches_rolling_mean() -> None:
    prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = sma(prices, 3)
    assert result.iloc[-1] == 4.0


def test_ema_preserves_index() -> None:
    prices = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2026-01-01", periods=5))
    result = ema(prices, 3)
    assert result.index.equals(prices.index)


def test_rsi_bounds() -> None:
    prices = pd.Series(np.linspace(100, 120, 40))
    result = rsi(prices, 14)
    assert result.dropna().between(0, 100).all()


def test_macd_schema() -> None:
    prices = pd.Series(np.linspace(100, 130, 80))
    result = macd(prices)
    assert set(result.columns) == {"macd", "signal", "histogram"}


def test_bollinger_bands_schema() -> None:
    prices = pd.Series(np.linspace(100, 130, 80))
    result = bollinger_bands(prices, window=20)
    assert {"middle", "upper", "lower", "rolling_std", "zscore"}.issubset(result.columns)
