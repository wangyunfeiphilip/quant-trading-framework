import numpy as np
import pandas as pd

from strategies.factor_strategy import calculate_factor_scores, generate_factor_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=180, freq="B")
    rows = []
    for idx, ticker in enumerate(["AAPL", "MSFT", "NVDA", "SPY"]):
        base = 100 + 10 * idx
        trend = np.linspace(0, 30 + 5 * idx, len(dates))
        for i, date in enumerate(dates):
            close = base + trend[i]
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": close,
                    "return_126d": close / base - 1,
                    "pe_ratio": 20 + idx,
                    "pb_ratio": 4 + idx,
                    "roe": 0.1 + idx * 0.02,
                    "revenue_growth": 0.05 + idx * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_weights_sum_to_one_or_zero() -> None:
    weights = generate_momentum_weights(_feature_frame(), top_n=2)
    sums = weights.groupby("date")["target_weight"].sum()
    assert sums.le(1.0 + 1e-12).all()


def test_mean_reversion_outputs_signal_columns() -> None:
    weights = generate_mean_reversion_weights(_feature_frame(), window=20)
    assert {"target_weight", "signal", "zscore"}.issubset(weights.columns)


def test_factor_strategy_uses_cross_sectional_scores() -> None:
    scored = calculate_factor_scores(_feature_frame())
    weights = generate_factor_weights(scored, top_n=2)
    assert "factor_score" in scored.columns
    assert weights.groupby("date")["target_weight"].sum().le(1.0 + 1e-12).all()
