"""Single-name quantitative market monitor."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from indicators.technical_indicators import macd, rsi, sma
from risk.risk_metrics import annualized_return, annualized_volatility, maximum_drawdown, sharpe_ratio


@dataclass(frozen=True)
class StockResearchSnapshot:
    """Compact quantitative snapshot for one ticker."""

    ticker: str
    last_price: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    sma_20: float
    sma_60: float
    rsi_14: float
    macd: float
    momentum_126d: float

    def to_dict(self) -> dict[str, float | str]:
        """Convert the snapshot to a serializable dictionary."""

        return asdict(self)


def analyze_ticker(price_data: pd.DataFrame, ticker: str, price_col: str = "adjusted_close") -> StockResearchSnapshot:
    """Create a quantitative research snapshot from historical price data."""

    frame = price_data[price_data["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()
    if frame.empty:
        raise ValueError(f"ticker not found in price_data: {ticker}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    close = pd.to_numeric(frame[price_col], errors="coerce").dropna()
    returns = close.pct_change().dropna()
    max_dd, _ = maximum_drawdown(returns)
    macd_frame = macd(close)

    return StockResearchSnapshot(
        ticker=ticker.upper(),
        last_price=float(close.iloc[-1]),
        annualized_return=annualized_return(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns),
        maximum_drawdown=max_dd,
        sma_20=float(sma(close, 20).iloc[-1]),
        sma_60=float(sma(close, 60).iloc[-1]),
        rsi_14=float(rsi(close, 14).iloc[-1]),
        macd=float(macd_frame["macd"].iloc[-1]),
        momentum_126d=float(close.pct_change(126).iloc[-1]) if len(close) > 126 else np.nan,
    )
