"""Discrete delta-hedging experiments for short European option positions."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np
import pandas as pd

from backtesting.execution import ExecutionModel, Order
from backtesting.portfolio import Portfolio
from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price


def _option_payoff(contract: OptionContract, spot: float) -> float:
    if contract.option_type == "call":
        return max(spot - contract.strike, 0.0)
    return max(contract.strike - spot, 0.0)


def _simulate_gbm_path(contract: OptionContract, n_steps: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = contract.maturity / n_steps
    shocks = rng.standard_normal(n_steps)
    increments = (
        (contract.rate - contract.dividend_yield - 0.5 * contract.volatility**2) * dt
        + contract.volatility * sqrt(dt) * shocks
    )
    log_path = np.r_[0.0, np.cumsum(increments)]
    return contract.spot * np.exp(log_path)


def _trade_to_target(
    portfolio: Portfolio,
    execution_model: ExecutionModel,
    date: pd.Timestamp,
    ticker: str,
    target_quantity: int,
    price: float,
) -> None:
    current = portfolio.quantity(ticker)
    if target_quantity > current:
        quantity = target_quantity - current
        max_affordable = int(portfolio.cash // execution_model.estimated_cash_required(1, price))
        quantity = min(quantity, max_affordable)
        if quantity > 0:
            order = Order(date=date, ticker=ticker, side="BUY", quantity=quantity, reference_price=price)
            portfolio.execute_fill(execution_model.execute(order))
    elif target_quantity < current:
        quantity = current - target_quantity
        order = Order(date=date, ticker=ticker, side="SELL", quantity=quantity, reference_price=price)
        portfolio.execute_fill(execution_model.execute(order))


def simulate_delta_hedge_path(
    contract: OptionContract,
    hedge_interval_days: int = 5,
    n_steps: int = 252,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    seed: int | None = None,
    ticker: str = "UNDERLYING",
    option_units: int = 1,
    contract_multiplier: int = 100,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate a short option plus dynamically rebalanced stock hedge.

    The stock leg is processed through the project `Portfolio` and
    `ExecutionModel`. A neutral cash buffer is added and removed at the end so
    the reported replication error reflects hedging PnL rather than financing.
    """

    contract.validate()
    if contract.option_type != "call":
        raise ValueError("current portfolio accounting supports short-call hedging experiments")
    if hedge_interval_days <= 0:
        raise ValueError("hedge_interval_days must be positive")
    if n_steps <= 1:
        raise ValueError("n_steps must be greater than one")

    prices = _simulate_gbm_path(contract, n_steps=n_steps, seed=seed)
    dates = pd.date_range("2026-01-02", periods=n_steps + 1, freq="B")
    multiplier = option_units * contract_multiplier
    option_premium = black_scholes_price(contract) * multiplier
    cash_buffer = contract.spot * multiplier
    portfolio = Portfolio(initial_cash=cash_buffer + option_premium)
    execution_model = ExecutionModel(transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps)

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    dt = contract.maturity / n_steps

    for step in range(n_steps):
        if step % hedge_interval_days != 0:
            continue
        remaining_maturity = max(contract.maturity - step * dt, dt)
        hedge_contract = replace(contract, spot=float(prices[step]), maturity=remaining_maturity)
        delta = black_scholes_greeks(hedge_contract)["delta"]
        target_quantity = int(round(delta * multiplier))
        before_cash = portfolio.cash
        _trade_to_target(portfolio, execution_model, dates[step], ticker, target_quantity, float(prices[step]))
        rows.append(
            {
                "date": dates[step],
                "step": step,
                "spot": float(prices[step]),
                "remaining_maturity": remaining_maturity,
                "delta": delta,
                "target_quantity": target_quantity,
                "actual_quantity": portfolio.quantity(ticker),
                "cash": portfolio.cash,
                "trade_cash_impact": portfolio.cash - before_cash,
            }
        )

    _trade_to_target(portfolio, execution_model, dates[-1], ticker, 0, float(prices[-1]))
    final_value_before_liability = portfolio.total_value({ticker: float(prices[-1])})
    option_liability = _option_payoff(contract, float(prices[-1])) * multiplier
    replication_error = final_value_before_liability - cash_buffer - option_liability
    transaction_cost = sum(float(row["transaction_cost"]) for row in portfolio.trade_history)

    summary = {
        "hedge_interval_days": float(hedge_interval_days),
        "n_hedges": float(len(rows)),
        "terminal_spot": float(prices[-1]),
        "option_premium": float(option_premium),
        "option_liability": float(option_liability),
        "transaction_cost": float(transaction_cost),
        "replication_error": float(replication_error),
        "absolute_replication_error": float(abs(replication_error)),
    }
    return pd.DataFrame(rows), summary


def hedging_frequency_experiment(
    contract: OptionContract,
    hedge_intervals: tuple[int, ...] = (1, 2, 5, 10, 21),
    n_paths: int = 100,
    n_steps: int = 252,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    seed: int = 11,
) -> pd.DataFrame:
    """Compare replication error and transaction costs by hedge frequency."""

    rows = []
    for interval in hedge_intervals:
        errors = []
        costs = []
        for path_index in range(n_paths):
            _, summary = simulate_delta_hedge_path(
                contract,
                hedge_interval_days=interval,
                n_steps=n_steps,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                seed=seed + path_index,
            )
            errors.append(summary["replication_error"])
            costs.append(summary["transaction_cost"])

        error_array = np.asarray(errors, dtype=float)
        cost_array = np.asarray(costs, dtype=float)
        rows.append(
            {
                "hedge_interval_days": interval,
                "hedges_per_year": n_steps / interval,
                "sqrt_delta_t": sqrt(interval / n_steps),
                "mean_replication_error": float(error_array.mean()),
                "mean_abs_replication_error": float(np.abs(error_array).mean()),
                "rmse_replication_error": float(sqrt(np.mean(error_array**2))),
                "avg_transaction_cost": float(cost_array.mean()),
                "rmse_plus_cost": float(sqrt(np.mean(error_array**2)) + cost_array.mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("hedge_interval_days").reset_index(drop=True)
    result["is_min_cost_adjusted_frequency"] = result["rmse_plus_cost"].eq(result["rmse_plus_cost"].min())
    return result
