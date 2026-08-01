"""Multi-factor stock selection strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
    def zscore(group: pd.Series) -> pd.Series:
        std = group.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return values.groupby(dates).transform(zscore)


def calculate_factor_scores(
    data: pd.DataFrame,
    value_weight: float = 0.30,
    momentum_weight: float = 0.30,
    quality_weight: float = 0.25,
    growth_weight: float = 0.15,
) -> pd.DataFrame:
    """Compute composite value, momentum, quality, and growth factor scores."""

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("pe_ratio", "pb_ratio", "return_126d", "roe", "revenue_growth"):
        if column not in frame.columns:
            frame[column] = np.nan

    frame["value_score"] = (
        _zscore_by_date(-frame["pe_ratio"], frame["date"])
        + _zscore_by_date(-frame["pb_ratio"], frame["date"])
    ) / 2.0
    frame["momentum_factor_score"] = _zscore_by_date(frame["return_126d"], frame["date"])
    frame["quality_score"] = _zscore_by_date(frame["roe"], frame["date"])
    frame["growth_score"] = _zscore_by_date(frame["revenue_growth"], frame["date"])

    score_cols = ["value_score", "momentum_factor_score", "quality_score", "growth_score"]
    frame[score_cols] = frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["factor_score"] = (
        value_weight * frame["value_score"]
        + momentum_weight * frame["momentum_factor_score"]
        + quality_weight * frame["quality_score"]
        + growth_weight * frame["growth_score"]
    )
    return frame


def _rebalance_dates(dates: pd.Series, frequency: str = "M") -> pd.DatetimeIndex:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates).sort_values().unique())
    if date_index.empty:
        return date_index
    return pd.DatetimeIndex(pd.Series(date_index).groupby(date_index.to_period(frequency)).max())


def generate_factor_weights(
    data: pd.DataFrame,
    top_n: int = 3,
    rebalance_frequency: str = "M",
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Select the highest composite factor score names on each rebalance date."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")

    scored = calculate_factor_scores(data)
    signal_dates = set(_rebalance_dates(scored["date"], rebalance_frequency))
    tickers = sorted(scored["ticker"].unique())
    rows: list[dict[str, object]] = []

    for date, group in scored[scored["date"].isin(signal_dates)].groupby("date"):
        ranked = group.sort_values("factor_score", ascending=False)
        selected = set(ranked.head(top_n)["ticker"])
        selected_count = len(selected)

        for ticker in tickers:
            score_value = group.loc[group["ticker"].eq(ticker), "factor_score"]
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "target_weight": gross_exposure / selected_count if ticker in selected and selected_count else 0.0,
                    "signal": "LONG" if ticker in selected else "FLAT",
                    "factor_score": float(score_value.iloc[0]) if not score_value.empty else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
