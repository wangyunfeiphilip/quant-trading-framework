import numpy as np
import pandas as pd

from risk.risk_metrics import alpha_beta, maximum_drawdown, performance_summary, tracking_error


def test_maximum_drawdown_detects_loss_path() -> None:
    returns = pd.Series([0.1, -0.2, 0.05])
    max_dd, drawdown = maximum_drawdown(returns)
    assert max_dd < 0
    assert len(drawdown) == 3


def test_benchmark_metrics_are_finite_for_valid_input() -> None:
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    strategy = pd.Series(np.linspace(0.001, 0.003, 30), index=dates)
    benchmark = pd.Series(np.linspace(0.0005, 0.002, 30), index=dates)
    alpha, beta = alpha_beta(strategy, benchmark)
    te = tracking_error(strategy, benchmark)
    assert np.isfinite(alpha)
    assert np.isfinite(beta)
    assert np.isfinite(te)


def test_performance_summary_contains_core_fields() -> None:
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    summary = performance_summary(returns)
    assert {"cumulative_return", "sharpe_ratio", "maximum_drawdown"}.issubset(summary.index)
