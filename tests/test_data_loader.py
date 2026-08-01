import pandas as pd

from data.data_loader import clean_price_data, create_feature_dataset


def _raw_prices() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    rows = []
    for ticker, start in [("AAPL", 100.0), ("MSFT", 200.0)]:
        for i, date in enumerate(dates):
            price = start + i
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000 + i,
                }
            )
    rows.append(rows[-1].copy())
    rows[-1]["close"] = 999.0
    return pd.DataFrame(rows)


def test_clean_price_data_removes_duplicate_dates() -> None:
    cleaned = clean_price_data(_raw_prices())
    duplicates = cleaned.duplicated(["date", "ticker"]).sum()
    assert duplicates == 0


def test_create_feature_dataset_adds_model_columns() -> None:
    cleaned = clean_price_data(_raw_prices())
    features = create_feature_dataset(cleaned)
    assert {"daily_return", "volatility_21d", "rsi_14", "macd", "future_21d_return"}.issubset(features.columns)
