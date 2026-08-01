"""Market data loading, cleaning, adjustment, and feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only in minimal envs
    yf = None

from indicators.technical_indicators import bollinger_bands, ema, macd, rsi, sma

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = ("NVDA", "MSFT", "AAPL", "GOOGL", "META", "SPY")
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for reproducible equity data builds."""

    tickers: tuple[str, ...] = DEFAULT_TICKERS
    start: str = "2015-01-01"
    end: str = "2026-12-31"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    abnormal_return_threshold: float = 0.50


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
        "Stock_Splits": "stock_splits",
    }
    out = frame.rename(columns=rename_map).copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    return out


def _download_to_long_frame(downloaded: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame(columns=("date", "ticker", *PRICE_COLUMNS))

    records = []
    if isinstance(downloaded.columns, pd.MultiIndex):
        level_zero = set(map(str, downloaded.columns.get_level_values(0)))
        ticker_first = set(tickers).issubset(level_zero)

        for ticker in tickers:
            if ticker_first:
                one = downloaded[ticker].copy()
            else:
                one = downloaded.xs(ticker, axis=1, level=1).copy()
            one = one.reset_index()
            one["ticker"] = ticker
            records.append(_normalize_columns(one))
    else:
        one = downloaded.reset_index()
        one["ticker"] = tickers[0]
        records.append(_normalize_columns(one))

    return pd.concat(records, ignore_index=True, sort=False)


def download_price_data(
    tickers: tuple[str, ...] | list[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: str = "2026-12-31",
    raw_dir: str | Path | None = "data/raw",
) -> pd.DataFrame:
    """Download OHLCV and corporate-action data from yfinance.

    Prices are requested with `auto_adjust=False` so adjusted and unadjusted
    fields remain available for split and dividend-aware research.
    """

    if yf is None:
        raise ImportError("yfinance is required. Install project requirements first.")

    ticker_tuple = tuple(tickers)
    LOGGER.info("Downloading price data for %s from %s to %s", ticker_tuple, start, end)
    downloaded = yf.download(
        list(ticker_tuple),
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    long_frame = _download_to_long_frame(downloaded, ticker_tuple)

    if raw_dir is not None:
        raw_path = Path(raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        for ticker, group in long_frame.groupby("ticker", sort=False):
            group.to_csv(raw_path / f"{ticker}.csv", index=False)

    return long_frame


def clean_price_data(
    data: pd.DataFrame,
    abnormal_return_threshold: float = 0.50,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Clean OHLCV records and create split-adjusted price fields."""

    if data.empty:
        raise ValueError("price data is empty")

    frame = _normalize_columns(data)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], utc=False).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame = frame.dropna(subset=["date", "ticker"])
    frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    for column in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price_cols = ["open", "high", "low", "close", "adj_close"]
    frame[price_cols] = frame.groupby("ticker", group_keys=False)[price_cols].apply(
        lambda group: group.ffill().bfill()
    )
    frame["volume"] = frame["volume"].fillna(0).clip(lower=0)
    frame["dividends"] = frame["dividends"].fillna(0)
    frame["stock_splits"] = frame["stock_splits"].fillna(0)
    frame["adj_close"] = frame["adj_close"].fillna(frame["close"])

    adjustment_factor = frame["adj_close"].div(frame["close"]).replace([np.inf, -np.inf], np.nan)
    adjustment_factor = adjustment_factor.groupby(frame["ticker"]).ffill().bfill().fillna(1.0)
    frame["adjustment_factor"] = adjustment_factor

    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column] * frame["adjustment_factor"]
    frame["adjusted_close"] = frame["adj_close"]

    frame["daily_return"] = frame.groupby("ticker")["adjusted_close"].pct_change()
    frame["abnormal_return_flag"] = frame["daily_return"].abs() > abnormal_return_threshold
    frame["clean_daily_return"] = frame["daily_return"].clip(
        lower=-abnormal_return_threshold,
        upper=abnormal_return_threshold,
    )
    frame["split_flag"] = frame["stock_splits"].fillna(0).ne(0)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)

    return frame.reset_index(drop=True)


def generate_data_quality_report(
    data: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize missing values, duplicates, abnormal prices, and date coverage."""

    frame = _normalize_columns(data)
    if "date" not in frame.columns or "ticker" not in frame.columns:
        raise ValueError("data quality report requires date and ticker columns")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    price_cols = [col for col in ("open", "high", "low", "close", "adj_close", "adjusted_close") if col in frame]

    rows = []
    for ticker, group in frame.groupby("ticker", sort=True):
        abnormal_count = 0
        if "abnormal_return_flag" in group.columns:
            abnormal_count = int(group["abnormal_return_flag"].fillna(False).sum())
        elif "adjusted_close" in group.columns:
            abnormal_count = int(group["adjusted_close"].pct_change().abs().gt(0.50).sum())

        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(group)),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "duplicate_records": int(group.duplicated(["date", "ticker"]).sum()),
                "missing_price_values": int(group[price_cols].isna().sum().sum()) if price_cols else 0,
                "invalid_dates": int(group["date"].isna().sum()),
                "non_positive_prices": int((group[price_cols] <= 0).sum().sum()) if price_cols else 0,
                "abnormal_price_moves": abnormal_count,
            }
        )

    report = pd.DataFrame(rows)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
    return report


def load_fundamental_features(tickers: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Load fundamental snapshot features from yfinance metadata."""

    rows: list[dict[str, object]] = []
    for ticker in tickers:
        info: dict[str, object] = {}
        if yf is not None:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception as exc:  # pragma: no cover - network/provider edge
                LOGGER.warning("Could not load fundamentals for %s: %s", ticker, exc)

        rows.append(
            {
                "ticker": str(ticker).upper(),
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "market_cap": info.get("marketCap", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "revenue_growth": info.get("revenueGrowth", np.nan),
            }
        )
    return pd.DataFrame(rows)


def add_technical_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Add return, volatility, moving-average, RSI, MACD, and band features."""

    frames = []
    for ticker, group in price_data.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = group.copy()
        close = g["adjusted_close"].astype(float)
        ret = close.pct_change()

        g["daily_return"] = ret
        g["volatility_21d"] = ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        g["sma_20"] = sma(close, 20)
        g["sma_60"] = sma(close, 60)
        g["ema_20"] = ema(close, 20)
        g["rsi_14"] = rsi(close, 14)

        macd_frame = macd(close)
        g["macd"] = macd_frame["macd"]
        g["macd_signal"] = macd_frame["signal"]
        g["macd_histogram"] = macd_frame["histogram"]

        bands = bollinger_bands(close, window=20, num_std=2.0)
        g["bb_middle"] = bands["middle"]
        g["bb_upper"] = bands["upper"]
        g["bb_lower"] = bands["lower"]
        g["zscore_20"] = bands["zscore"]

        g["return_21d"] = close.pct_change(21)
        g["return_63d"] = close.pct_change(63)
        g["return_126d"] = close.pct_change(126)
        g["future_21d_return"] = close.shift(-21).div(close).sub(1.0)
        g["ticker"] = ticker
        frames.append(g)

    return pd.concat(frames, ignore_index=True, sort=False)


def create_feature_dataset(
    clean_data: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create the model-ready feature dataset from clean price data."""

    features = add_technical_features(clean_data)
    if fundamentals is not None and not fundamentals.empty:
        fund = fundamentals.copy()
        fund["ticker"] = fund["ticker"].astype(str).str.upper()
        features = features.merge(fund, on="ticker", how="left")

    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out, index=False)

    return features


def build_market_dataset(config: MarketDataConfig = MarketDataConfig()) -> pd.DataFrame:
    """Run the full data pipeline and save raw, clean, and feature datasets."""

    raw = download_price_data(config.tickers, config.start, config.end, config.raw_dir)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_price_data(
        raw,
        abnormal_return_threshold=config.abnormal_return_threshold,
        output_path=config.processed_dir / "clean_stock_data.csv",
    )
    generate_data_quality_report(clean, output_path=config.processed_dir / "data_quality_report.csv")
    fundamentals = load_fundamental_features(config.tickers)
    fundamentals.to_csv(config.processed_dir / "fundamental_features.csv", index=False)
    return create_feature_dataset(
        clean,
        fundamentals=fundamentals,
        output_path=config.processed_dir / "feature_dataset.csv",
    )
