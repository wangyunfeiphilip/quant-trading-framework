"""Cross-sectional momentum strategy with monthly rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_scores(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (21, 63, 126),
    weights: tuple[float, ...] | None = None,
    price_col: str = "adjusted_close",
) -> pd.DataFrame:
    """Compute weighted momentum score from multiple lookback returns."""

    if weights is None:
        weights = tuple(1.0 / len(horizons) for _ in horizons)
    if len(horizons) != len(weights):
        raise ValueError("horizons and weights must have the same length")

    frame = data[["date", "ticker", price_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])

    score = pd.Series(0.0, index=frame.index)
    for horizon, weight in zip(horizons, weights):
        col = f"return_{horizon}d"
        frame[col] = frame.groupby("ticker")[price_col].pct_change(horizon)
        score = score.add(frame[col].mul(weight), fill_value=0.0)

    frame["momentum_score"] = score
    return frame


def _rebalance_dates(dates: pd.Series, frequency: str = "M") -> pd.DatetimeIndex:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates).sort_values().unique())
    if date_index.empty:
        return date_index
    return pd.DatetimeIndex(pd.Series(date_index).groupby(date_index.to_period(frequency)).max())


def generate_momentum_weights(
    data: pd.DataFrame,
    top_n: int = 3,
    horizons: tuple[int, ...] = (21, 63, 126),
    weights: tuple[float, ...] | None = None,
    rebalance_frequency: str = "M",
    long_only: bool = True,
) -> pd.DataFrame:
    """Rank stocks by momentum score and produce target weights on rebalance dates."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    scores = compute_momentum_scores(data, horizons=horizons, weights=weights)
    signal_dates = set(_rebalance_dates(scores["date"], rebalance_frequency))
    tickers = sorted(scores["ticker"].unique())
    rows: list[dict[str, object]] = []

    for date, group in scores[scores["date"].isin(signal_dates)].groupby("date"):
        ranked = group.dropna(subset=["momentum_score"]).sort_values("momentum_score", ascending=False)
        if long_only:
            ranked = ranked[ranked["momentum_score"] > 0]
        selected = set(ranked.head(top_n)["ticker"])
        selected_count = len(selected)

        for ticker in tickers:
            score_value = group.loc[group["ticker"].eq(ticker), "momentum_score"]
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "target_weight": 1.0 / selected_count if ticker in selected and selected_count else 0.0,
                    "signal": "LONG" if ticker in selected else "FLAT",
                    "momentum_score": float(score_value.iloc[0]) if not score_value.empty else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
