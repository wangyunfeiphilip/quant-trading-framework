"""Performance, downside risk, and benchmark-relative analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_returns(values: pd.Series | pd.DataFrame, value_col: str = "total_value") -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if "daily_return" in values.columns:
            returns = values["daily_return"]
        else:
            returns = values[value_col].pct_change()
    else:
        returns = values
    return pd.to_numeric(returns, errors="coerce").dropna()


def cumulative_return(returns: pd.Series) -> pd.Series:
    """Compute compounded cumulative return path."""

    r = _to_returns(returns)
    return (1.0 + r).cumprod() - 1.0


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize compounded returns."""

    r = _to_returns(returns)
    if r.empty:
        return np.nan
    total = (1.0 + r).prod()
    return float(total ** (periods_per_year / len(r)) - 1.0)


def monthly_returns(portfolio_value: pd.DataFrame, value_col: str = "total_value") -> pd.Series:
    """Resample portfolio value to month-end returns."""

    frame = portfolio_value.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    monthly_value = frame.set_index("date")[value_col].resample("M").last()
    return monthly_value.pct_change().dropna()


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize return standard deviation."""

    r = _to_returns(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Sharpe ratio using excess arithmetic returns."""

    r = _to_returns(returns)
    excess = r - risk_free_rate / periods_per_year
    vol = excess.std(ddof=1)
    return float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol and np.isfinite(vol) else np.nan


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Sortino ratio using downside deviation."""

    r = _to_returns(returns)
    excess = r - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std else np.nan


def maximum_drawdown(returns: pd.Series) -> tuple[float, pd.Series]:
    """Return maximum drawdown and full drawdown path."""

    cumulative = (1.0 + _to_returns(returns)).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative.div(running_max).sub(1.0)
    return float(drawdown.min()), drawdown


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Estimate annualized CAPM alpha and beta versus a benchmark."""

    aligned = pd.concat([_to_returns(returns), _to_returns(benchmark_returns)], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan, np.nan
    aligned.columns = ["strategy", "benchmark"]
    excess_strategy = aligned["strategy"] - risk_free_rate / periods_per_year
    excess_benchmark = aligned["benchmark"] - risk_free_rate / periods_per_year
    variance = excess_benchmark.var(ddof=1)
    if variance == 0 or not np.isfinite(variance):
        return np.nan, np.nan
    beta = excess_strategy.cov(excess_benchmark) / variance
    alpha_daily = excess_strategy.mean() - beta * excess_benchmark.mean()
    return float(alpha_daily * periods_per_year), float(beta)


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of active returns."""

    aligned = pd.concat([_to_returns(returns), _to_returns(benchmark_returns)], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Create a compact performance and risk summary."""

    r = _to_returns(returns)
    max_dd, _ = maximum_drawdown(r)
    summary = {
        "cumulative_return": float((1.0 + r).prod() - 1.0),
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(r, risk_free_rate, periods_per_year),
        "maximum_drawdown": max_dd,
    }
    if benchmark_returns is not None:
        alpha, beta = alpha_beta(r, benchmark_returns, risk_free_rate, periods_per_year)
        summary["alpha_vs_benchmark"] = alpha
        summary["beta_vs_benchmark"] = beta
        summary["tracking_error"] = tracking_error(r, benchmark_returns, periods_per_year)
    return pd.Series(summary)
