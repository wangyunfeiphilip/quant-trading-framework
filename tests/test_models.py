import numpy as np
import pandas as pd

from models.factor_model import build_proxy_factors
from models.prediction_model import chronological_train_test_split, create_supervised_dataset


def test_proxy_factor_builder_schema() -> None:
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    rows = []
    for ticker, market_cap, pb in [("AAPL", 3e12, 8), ("MSFT", 2e12, 7), ("NVDA", 4e12, 10), ("SPY", 5e11, 3)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "daily_return": 0.001 + i * 0.00001,
                    "market_cap": market_cap,
                    "pb_ratio": pb,
                }
            )
    factors = build_proxy_factors(pd.DataFrame(rows))
    assert {"date", "mkt_rf", "smb", "hml", "rf"}.issubset(factors.columns)


def test_supervised_dataset_and_chronological_split() -> None:
    frame = pd.DataFrame(
        {
            "daily_return": np.linspace(0.0, 0.1, 50),
            "volatility_21d": np.linspace(0.1, 0.2, 50),
            "future_21d_return": np.linspace(-0.02, 0.03, 50),
        }
    )
    x, y = create_supervised_dataset(frame, feature_columns=["daily_return", "volatility_21d"])
    x_train, x_test, y_train, y_test = chronological_train_test_split(x, y, test_size=0.2)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.index.max() < x_test.index.min()
