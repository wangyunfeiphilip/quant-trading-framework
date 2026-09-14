"""Safe project market-data refresh helpers for dashboard and automations."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from config_utils import load_config
from data.data_loader import (
    DEFAULT_TICKERS,
    MarketDataConfig,
    build_market_dataset,
    latest_market_date,
    market_dataset_is_stale,
    next_yfinance_end_date,
)


def project_market_data_status(project_root: Path) -> dict[str, Any]:
    """Return the local processed data freshness status."""

    path = project_root / "data" / "processed" / "feature_dataset.csv"
    if not path.exists():
        return {"path": str(path), "latest_date": None, "stale": True, "rows": 0, "tickers": 0}

    frame = pd.read_csv(path, usecols=["date", "ticker"])
    latest = latest_market_date(frame)
    return {
        "path": str(path),
        "latest_date": latest.isoformat() if latest is not None else None,
        "stale": market_dataset_is_stale(frame),
        "rows": int(len(frame)),
        "tickers": int(frame["ticker"].nunique()) if "ticker" in frame else 0,
    }


def refresh_project_market_data(project_root: Path, tickers: list[str] | tuple[str, ...], start: str) -> dict[str, Any]:
    """Rebuild the project price and feature dataset without corrupting the old copy on failure."""

    ticker_list = [str(ticker).upper() for ticker in tickers if str(ticker).strip()]
    if not ticker_list:
        ticker_list = list(DEFAULT_TICKERS)

    with tempfile.TemporaryDirectory(prefix="qtf_market_refresh_") as tmp:
        tmp_root = Path(tmp)
        config = MarketDataConfig(
            tickers=tuple(ticker_list),
            start=start,
            end=next_yfinance_end_date(),
            raw_dir=tmp_root / "raw",
            processed_dir=tmp_root / "processed",
        )
        refreshed = build_market_dataset(config)

        ticker_count = refreshed["ticker"].nunique() if "ticker" in refreshed.columns else 0
        minimum_coverage = max(1, int(len(ticker_list) * 0.90))
        if ticker_count < minimum_coverage:
            raise ValueError(f"downloaded {ticker_count}/{len(ticker_list)} tickers; keeping existing dataset")

        if market_dataset_is_stale(refreshed):
            latest = latest_market_date(refreshed)
            raise ValueError(f"latest downloaded date is {latest.date() if latest is not None else 'missing'}")

        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in (tmp_root / "processed").glob("*.csv"):
            shutil.copy2(csv_file, processed_dir / csv_file.name)

        raw_dir = project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in (tmp_root / "raw").glob("*.csv"):
            shutil.copy2(csv_file, raw_dir / csv_file.name)

    latest = latest_market_date(refreshed)
    return {
        "refreshed": True,
        "latest_date": latest.isoformat() if latest is not None else None,
        "rows": int(len(refreshed)),
        "tickers": int(refreshed["ticker"].nunique()) if "ticker" in refreshed else 0,
    }


def refresh_from_project_config(project_root: Path, config_file: str = "config.yaml") -> dict[str, Any]:
    """Refresh local market data using the project dashboard config."""

    config = load_config(project_root / config_file)
    data_config = config.get("data", {})
    return refresh_project_market_data(
        project_root=project_root,
        tickers=config.get("universe") or list(DEFAULT_TICKERS),
        start=str(data_config.get("start", "2015-01-01")),
    )
