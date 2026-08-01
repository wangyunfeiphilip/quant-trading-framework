"""Time-series analytics for equity returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Compute annualized rolling volatility."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    return pd.to_numeric(returns, errors="coerce").rolling(window).std() * np.sqrt(periods_per_year)


def trend_signal(prices: pd.Series, short_window: int = 20, long_window: int = 60) -> pd.Series:
    """Return a moving-average trend signal."""

    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        raise ValueError("require 0 < short_window < long_window")
    price = pd.to_numeric(prices, errors="coerce")
    short_ma = price.rolling(short_window).mean()
    long_ma = price.rolling(long_window).mean()
    return (short_ma > long_ma).astype(int)


def arima_forecast(returns: pd.Series, steps: int = 5, order: tuple[int, int, int] = (1, 0, 1)) -> pd.Series:
    """Forecast returns with a basic ARIMA model."""

    from statsmodels.tsa.arima.model import ARIMA

    if steps <= 0:
        raise ValueError("steps must be positive")
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean_returns) < 30:
        raise ValueError("ARIMA forecasting requires at least 30 return observations")
    forecast = ARIMA(clean_returns, order=order).fit().forecast(steps=steps)
    forecast.name = "forecast_return"
    return forecast
