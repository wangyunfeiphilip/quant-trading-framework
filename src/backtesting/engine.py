"""Backtesting engine with signal lag, costs, slippage, and trade ledger."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from backtesting.execution import ExecutionModel, Order
from backtesting.portfolio import Portfolio

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outputs used by analytics and reporting modules."""

    portfolio_value: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


class BacktestEngine:
    """Long-only target-weight backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        signal_lag: int = 1,
        price_col: str = "adjusted_close",
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.execution_model = ExecutionModel(transaction_cost_bps, slippage_bps)
        self.signal_lag = int(signal_lag)
        self.price_col = price_col

    def _price_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data[["date", "ticker", self.price_col]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.pivot_table(index="date", columns="ticker", values=self.price_col, aggfunc="last").sort_index()

    def _target_matrix(self, target_weights: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
        weights = target_weights[["date", "ticker", "target_weight"]].copy()
        weights["date"] = pd.to_datetime(weights["date"])
        matrix = weights.pivot_table(index="date", columns="ticker", values="target_weight", aggfunc="last")
        matrix = matrix.reindex(dates).ffill().fillna(0.0)
        matrix = matrix.reindex(columns=tickers, fill_value=0.0)
        if self.signal_lag > 0:
            matrix = matrix.shift(self.signal_lag).fillna(0.0)
        return matrix.clip(lower=0.0)

    def run(self, price_data: pd.DataFrame, target_weights: pd.DataFrame) -> BacktestResult:
        """Run a daily mark-to-market backtest from target weights."""

        prices = self._price_matrix(price_data).dropna(how="all")
        if prices.empty:
            raise ValueError("price data has no usable prices")

        targets = self._target_matrix(target_weights, prices.index, prices.columns)
        portfolio = Portfolio(self.initial_capital)
        value_rows: list[dict[str, object]] = []
        position_rows: list[dict[str, object]] = []
        last_target = pd.Series(0.0, index=prices.columns)

        for date in prices.index:
            price_row = prices.loc[date].dropna()
            if price_row.empty:
                continue
            price_dict = price_row.to_dict()
            desired = targets.loc[date].reindex(prices.columns).fillna(0.0)

            if not np.allclose(desired.to_numpy(), last_target.to_numpy(), atol=1e-10):
                self._rebalance(date, desired, price_dict, portfolio)
                last_target = desired.copy()

            holdings = portfolio.holdings_value(price_dict)
            total = portfolio.cash + holdings
            value_rows.append(
                {
                    "date": date,
                    "cash": portfolio.cash,
                    "holdings": holdings,
                    "total_value": total,
                }
            )
            position_rows.extend(portfolio.position_snapshot(date, price_dict))

        portfolio_value = pd.DataFrame(value_rows).sort_values("date")
        portfolio_value["daily_return"] = portfolio_value["total_value"].pct_change().fillna(0.0)
        portfolio_value["cumulative_return"] = portfolio_value["total_value"].div(self.initial_capital).sub(1.0)
        trades = pd.DataFrame(portfolio.trade_history)
        positions = pd.DataFrame(position_rows)
        return BacktestResult(portfolio_value=portfolio_value, trades=trades, positions=positions)

    def _rebalance(
        self,
        date: pd.Timestamp,
        desired_weights: pd.Series,
        prices: dict[str, float],
        portfolio: Portfolio,
    ) -> None:
        total_value = portfolio.total_value(prices)
        desired_quantities: dict[str, int] = {}
        for ticker, weight in desired_weights.items():
            price = prices.get(ticker)
            if price is None or price <= 0:
                desired_quantities[ticker] = portfolio.quantity(ticker)
            else:
                desired_quantities[ticker] = int(np.floor((total_value * float(weight)) / price))

        for ticker, target_qty in desired_quantities.items():
            current_qty = portfolio.quantity(ticker)
            sell_qty = max(0, current_qty - target_qty)
            if sell_qty:
                order = Order(date=date, ticker=ticker, side="SELL", quantity=sell_qty, reference_price=prices[ticker])
                portfolio.execute_fill(self.execution_model.execute(order))

        for ticker, target_qty in desired_quantities.items():
            current_qty = portfolio.quantity(ticker)
            buy_qty = max(0, target_qty - current_qty)
            if not buy_qty:
                continue
            price = prices[ticker]
            max_affordable = int(np.floor(portfolio.cash / self.execution_model.estimated_cash_required(1, price)))
            executable_qty = min(buy_qty, max_affordable)
            if executable_qty <= 0:
                LOGGER.debug("Skipped buy for %s on %s due to cash constraint", ticker, date)
                continue
            order = Order(date=date, ticker=ticker, side="BUY", quantity=executable_qty, reference_price=price)
            portfolio.execute_fill(self.execution_model.execute(order))
