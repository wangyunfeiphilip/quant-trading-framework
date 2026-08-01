"""Fama-French style factor exposure estimation."""

from __future__ import annotations

import pandas as pd


def build_proxy_factors(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Build market, size, and value factor proxies from the project universe."""

    frame = feature_data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "daily_return" not in frame.columns:
        frame["daily_return"] = frame.groupby("ticker")["adjusted_close"].pct_change()

    rows = []
    for date, group in frame.dropna(subset=["daily_return"]).groupby("date"):
        market = group.loc[group["ticker"].eq("SPY"), "daily_return"]
        market_return = float(market.iloc[0]) if not market.empty else float(group["daily_return"].mean())

        size_factor = 0.0
        if "market_cap" in group.columns and group["market_cap"].notna().sum() >= 4:
            ranked_size = group.dropna(subset=["market_cap"]).sort_values("market_cap")
            small = ranked_size.head(max(1, len(ranked_size) // 3))["daily_return"].mean()
            big = ranked_size.tail(max(1, len(ranked_size) // 3))["daily_return"].mean()
            size_factor = float(small - big)

        value_factor = 0.0
        if "pb_ratio" in group.columns and group["pb_ratio"].notna().sum() >= 4:
            ranked_value = group.dropna(subset=["pb_ratio"]).sort_values("pb_ratio")
            value = ranked_value.head(max(1, len(ranked_value) // 3))["daily_return"].mean()
            growth = ranked_value.tail(max(1, len(ranked_value) // 3))["daily_return"].mean()
            value_factor = float(value - growth)

        rows.append(
            {
                "date": date,
                "mkt_rf": market_return,
                "smb": size_factor,
                "hml": value_factor,
                "rf": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fama_french_regression(strategy_returns: pd.Series, factors: pd.DataFrame):
    """Fit OLS: R_p - R_f = alpha + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + eps."""

    import statsmodels.api as sm

    y = pd.Series(strategy_returns).rename("strategy_return")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("strategy_returns must have a DatetimeIndex")

    factor_frame = factors.copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"])
    factor_frame = factor_frame.set_index("date")
    required = ["mkt_rf", "smb", "hml"]
    missing = [column for column in required if column not in factor_frame.columns]
    if missing:
        raise ValueError(f"missing factor columns: {missing}")
    if "rf" not in factor_frame.columns:
        factor_frame["rf"] = 0.0

    aligned = pd.concat([y, factor_frame[["mkt_rf", "smb", "hml", "rf"]]], axis=1).dropna()
    if aligned.shape[0] < 20:
        raise ValueError("not enough observations for factor regression")

    dependent = aligned["strategy_return"] - aligned["rf"]
    independent = sm.add_constant(aligned[["mkt_rf", "smb", "hml"]])
    result = sm.OLS(dependent, independent).fit()

    exposure = pd.DataFrame(
        {
            "coefficient": result.params,
            "p_value": result.pvalues,
            "t_stat": result.tvalues,
        }
    )
    exposure.loc["r_squared", "coefficient"] = result.rsquared
    return result, exposure
