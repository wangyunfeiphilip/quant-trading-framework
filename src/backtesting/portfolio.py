"""Portfolio accounting for long-only equity backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backtesting.execution import Fill


@dataclass
class Position:
    """Single-name position state."""

    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0

    def market_value(self, price: float) -> float:
        """Return current marked value."""

        return self.quantity * price


class Portfolio:
    """Cash, positions, and trade ledger."""

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict[str, object]] = []

    def quantity(self, ticker: str) -> int:
        """Return held share quantity for a ticker."""

        position = self.positions.get(ticker)
        return 0 if position is None else position.quantity

    def holdings_value(self, prices: dict[str, float]) -> float:
        """Mark holdings to supplied prices."""

        return sum(position.market_value(prices.get(ticker, 0.0)) for ticker, position in self.positions.items())

    def total_value(self, prices: dict[str, float]) -> float:
        """Return cash plus marked holdings."""

        return self.cash + self.holdings_value(prices)

    def execute_fill(self, fill: Fill) -> None:
        """Apply a fill to cash, position state, and trade history."""

        ticker = fill.ticker
        position = self.positions.get(ticker, Position(ticker=ticker))

        if fill.side == "BUY":
            total_cost = fill.notional + fill.transaction_cost
            if total_cost > self.cash + 1e-8:
                raise ValueError(f"insufficient cash for {ticker} buy")
            new_quantity = position.quantity + fill.quantity
            if new_quantity <= 0:
                raise ValueError("buy produced non-positive position")
            position.avg_cost = (
                position.avg_cost * position.quantity + fill.executed_price * fill.quantity
            ) / new_quantity
            position.quantity = new_quantity
            self.cash -= total_cost
        else:
            if fill.quantity > position.quantity:
                raise ValueError(f"cannot sell more {ticker} shares than held")
            position.quantity -= fill.quantity
            self.cash += fill.notional - fill.transaction_cost
            if position.quantity == 0:
                position.avg_cost = 0.0

        if position.quantity:
            self.positions[ticker] = position
        else:
            self.positions.pop(ticker, None)

        row = asdict(fill)
        row["cash_after"] = self.cash
        self.trade_history.append(row)

    def position_snapshot(self, date: object, prices: dict[str, float]) -> list[dict[str, object]]:
        """Create position rows for reporting."""

        rows = []
        for ticker, position in sorted(self.positions.items()):
            price = prices.get(ticker, 0.0)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "quantity": position.quantity,
                    "avg_cost": position.avg_cost,
                    "market_price": price,
                    "market_value": position.market_value(price),
                }
            )
        return rows
