import io
import zipfile

import numpy as np
import pandas as pd

from models.factor_model import build_proxy_factors, fama_french_regression, parse_kenneth_french_daily_factors
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


def test_kenneth_french_daily_parser_reads_zip_percent_returns() -> None:
    csv_text = "\n".join(
        [
            "This file was created by CMPT_ME_BEME_RETS_DAILY",
            ",Mkt-RF,SMB,HML,RF",
            "20260102,1.00,0.25,-0.50,0.01",
            "20260105,-0.40,0.10,0.20,0.01",
            "Annual Factors: January-December",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", csv_text)

    factors = parse_kenneth_french_daily_factors(buffer.getvalue())
    assert list(factors.columns) == ["date", "mkt_rf", "smb", "hml", "rf"]
    assert factors.loc[0, "mkt_rf"] == 0.01
    assert factors.loc[1, "hml"] == 0.002


def test_fama_french_regression_uses_hac_covariance() -> None:
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    factors = pd.DataFrame(
        {
            "date": dates,
            "mkt_rf": np.linspace(-0.01, 0.01, len(dates)),
            "smb": np.sin(np.arange(len(dates))) * 0.001,
            "hml": np.cos(np.arange(len(dates))) * 0.001,
            "rf": 0.0001,
        }
    )
    returns = pd.Series(0.0002 + 1.1 * factors["mkt_rf"].to_numpy(), index=dates)
    result, exposure = fama_french_regression(returns, factors, hac_maxlags=3)

    assert result.cov_type == "HAC"
    assert exposure.loc["hac_maxlags", "coefficient"] == 3
    assert "standard_error" in exposure.columns


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


def test_supervised_dataset_sorts_by_date_when_available() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "daily_return": [0.03, 0.01, 0.02],
            "volatility_21d": [0.3, 0.1, 0.2],
            "future_21d_return": [0.3, 0.1, 0.2],
        }
    )
    x, y = create_supervised_dataset(frame, feature_columns=["daily_return", "volatility_21d"])

    assert list(x["daily_return"]) == [0.01, 0.02, 0.03]
    assert list(y) == [0.1, 0.2, 0.3]


def test_supervised_dataset_keeps_missing_features_for_pipeline_imputation() -> None:
    frame = pd.DataFrame(
        {
            "daily_return": [0.01, np.nan, 0.03],
            "volatility_21d": [0.1, 0.2, 0.3],
            "future_21d_return": [0.02, 0.01, np.nan],
        }
    )
    x, y = create_supervised_dataset(frame, feature_columns=["daily_return", "volatility_21d"])

    assert len(x) == 2
    assert x["daily_return"].isna().sum() == 1
    assert len(y) == 2
