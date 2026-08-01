import pandas as pd

from backtesting.engine import BacktestEngine


def test_backtest_generates_trade_ledger_and_value_path() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "adjusted_close": [100, 101, 102, 103, 104, 200, 201, 202, 203, 204],
        }
    )
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "ticker": ["AAPL", "MSFT"],
            "target_weight": [0.5, 0.5],
        }
    )
    result = BacktestEngine(initial_capital=100000, signal_lag=1).run(prices, weights)
    assert not result.portfolio_value.empty
    assert not result.trades.empty
    assert result.portfolio_value["cash"].min() >= -1e-8
