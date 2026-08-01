"""Technical indicators used by the research strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(values: pd.Series | np.ndarray | list[float], name: str = "value") -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values, name=name)
    return pd.to_numeric(series, errors="coerce")


def sma(prices: pd.Series | np.ndarray | list[float], window: int) -> pd.Series:
    """Simple moving average: SMA_t = (1 / n) * sum(P_{t-i})."""

    if window <= 0:
        raise ValueError("window must be positive")
    return _as_series(prices, "price").rolling(window=window, min_periods=window).mean()


def ema(prices: pd.Series | np.ndarray | list[float], span: int) -> pd.Series:
    """Exponential moving average: EMA_t = alpha * P_t + (1-alpha) * EMA_{t-1}."""

    if span <= 0:
        raise ValueError("span must be positive")
    return _as_series(prices, "price").ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(prices: pd.Series | np.ndarray | list[float], window: int = 14) -> pd.Series:
    """Relative Strength Index: RSI = 100 - 100 / (1 + average gain / average loss)."""

    if window <= 0:
        raise ValueError("window must be positive")

    price = _as_series(prices, "price")
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = avg_gain.div(avg_loss.replace(0, np.nan))
    value = 100 - (100 / (1 + relative_strength))
    return value.fillna(50.0)


def macd(
    prices: pd.Series | np.ndarray | list[float],
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence: MACD = EMA_fast - EMA_slow."""

    if not (0 < fast_span < slow_span):
        raise ValueError("fast_span must be positive and smaller than slow_span")
    if signal_span <= 0:
        raise ValueError("signal_span must be positive")

    price = _as_series(prices, "price")
    line = ema(price, fast_span) - ema(price, slow_span)
    signal = line.ewm(span=signal_span, adjust=False, min_periods=signal_span).mean()
    return pd.DataFrame(
        {
            "macd": line,
            "signal": signal,
            "histogram": line - signal,
        },
        index=price.index,
    )


def bollinger_bands(
    prices: pd.Series | np.ndarray | list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: middle = SMA, upper/lower = SMA +/- k * rolling sigma."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if num_std <= 0:
        raise ValueError("num_std must be positive")

    price = _as_series(prices, "price")
    middle = sma(price, window)
    rolling_std = price.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = price.sub(middle).div(rolling_std.replace(0, np.nan))
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + num_std * rolling_std,
            "lower": middle - num_std * rolling_std,
            "rolling_std": rolling_std,
            "zscore": zscore,
        },
        index=price.index,
    )
