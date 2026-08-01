"""Mean-variance portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_mean_returns(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Estimate expected annual returns from daily asset returns."""

    return returns.mean() * periods_per_year


def annualized_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Estimate annualized covariance matrix from daily asset returns."""

    return returns.cov() * periods_per_year


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Return expected return, volatility, and Sharpe ratio."""

    w = np.asarray(weights, dtype=float)
    expected = float(np.dot(w, expected_returns.to_numpy()))
    volatility = float(np.sqrt(w.T @ covariance.to_numpy() @ w))
    sharpe = (expected - risk_free_rate) / volatility if volatility > 0 else np.nan
    return expected, volatility, float(sharpe)


def _bounds(n_assets: int) -> tuple[tuple[float, float], ...]:
    return tuple((0.0, 1.0) for _ in range(n_assets))


def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}


def minimum_volatility_portfolio(expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Compute long-only minimum volatility portfolio weights."""

    from scipy.optimize import minimize

    n_assets = len(expected_returns)
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        return portfolio_performance(weights, expected_returns, covariance)[1]

    result = minimize(objective, initial, method="SLSQP", bounds=_bounds(n_assets), constraints=[_sum_to_one_constraint()])
    if not result.success:
        raise RuntimeError(f"minimum volatility optimization failed: {result.message}")
    return pd.Series(result.x, index=expected_returns.index, name="weight")


def maximum_sharpe_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute long-only maximum Sharpe ratio portfolio weights."""

    from scipy.optimize import minimize

    n_assets = len(expected_returns)
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        return -portfolio_performance(weights, expected_returns, covariance, risk_free_rate)[2]

    result = minimize(objective, initial, method="SLSQP", bounds=_bounds(n_assets), constraints=[_sum_to_one_constraint()])
    if not result.success:
        raise RuntimeError(f"maximum Sharpe optimization failed: {result.message}")
    return pd.Series(result.x, index=expected_returns.index, name="weight")


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    points: int = 25,
) -> pd.DataFrame:
    """Generate a long-only efficient frontier over target returns."""

    from scipy.optimize import minimize

    n_assets = len(expected_returns)
    initial = np.full(n_assets, 1.0 / n_assets)
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), points)
    rows = []

    for target in target_returns:
        constraints = [
            _sum_to_one_constraint(),
            {"type": "eq", "fun": lambda weights, target_return=target: np.dot(weights, expected_returns) - target_return},
        ]

        def objective(weights: np.ndarray) -> float:
            return portfolio_performance(weights, expected_returns, covariance)[1]

        result = minimize(objective, initial, method="SLSQP", bounds=_bounds(n_assets), constraints=constraints)
        if result.success:
            ret, vol, sharpe = portfolio_performance(result.x, expected_returns, covariance)
            rows.append({"target_return": target, "expected_return": ret, "volatility": vol, "sharpe": sharpe})

    return pd.DataFrame(rows)
