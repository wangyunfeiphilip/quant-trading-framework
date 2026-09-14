from __future__ import annotations

import time
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from data import data_loader


def _raw_prices() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": np.linspace(100.0, 140.0, len(dates)),
            "high": np.linspace(101.0, 141.0, len(dates)),
            "low": np.linspace(99.0, 139.0, len(dates)),
            "close": np.linspace(100.0, 140.0, len(dates)),
            "adj_close": np.linspace(100.0, 140.0, len(dates)),
            "volume": 1_000,
        }
    )


def test_fetch_live_ticker_quote_uses_short_history_without_file_download(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {"Close": [123.45]},
                index=pd.DatetimeIndex(["2026-09-11 15:59:00",], tz="America/New_York"),
            )

    monkeypatch.setattr(data_loader.yf, "Ticker", lambda ticker: FakeTicker())
    monkeypatch.setattr(data_loader, "download_price_data", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    quote = data_loader.fetch_live_ticker_quote("AAPL")

    assert quote["price"] == 123.45
    assert calls[0]["period"] == "1d"
    assert calls[0]["interval"] == "1m"
    assert calls[0]["timeout"] == 30


def test_rate_limit_enters_cooldown_and_returns_cached_quote(monkeypatch) -> None:
    class YFRateLimitError(Exception):
        pass

    app._live_quote_memory.clear()
    app._live_rate_limit_until = 0.0
    cached = {"price": 101.0, "market_data_timestamp": "2026-09-11T15:59:00-04:00"}
    app._live_quote_memory["AAPL"] = cached

    def fail(_ticker: str):
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr(app, "load_live_ticker_quote", fail)

    quote, status = app.get_live_quote("AAPL")

    assert quote == cached
    assert status == "cached"
    assert app._live_rate_limit_until >= time.monotonic() + app.LIVE_RATE_LIMIT_COOLDOWN - 1

    calls = 0

    def should_not_call(_ticker: str):
        nonlocal calls
        calls += 1
        return {"price": 102.0}

    monkeypatch.setattr(app, "load_live_ticker_quote", should_not_call)
    quote, status = app.get_live_quote("AAPL")
    assert quote == cached
    assert status == "cached"
    assert calls == 0

    app._live_rate_limit_until = 0.0
    app._live_quote_memory.clear()


def _capture_status(monkeypatch):
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(app.st, "caption", lambda value: messages.append(("caption", value)))
    monkeypatch.setattr(app.st, "warning", lambda value: messages.append(("warning", value)))
    monkeypatch.setattr(app.st, "info", lambda value: messages.append(("info", value)))
    return messages


def test_fresh_quote_is_explicitly_live(monkeypatch) -> None:
    messages = _capture_status(monkeypatch)
    checked_at = datetime(2026, 9, 11, 10, 42, 30, tzinfo=ZoneInfo("America/New_York"))
    quote = {"price": 123.45, "market_data_timestamp": "2026-09-11T10:42:00-04:00"}

    app.render_live_quote_status("live", checked_at, quote)

    text = " ".join(value for _, value in messages)
    assert "LIVE" in text
    assert "Live quote retrieved" in text
    assert app.quote_price_label("live") == "Latest Live Price"


def test_cached_quote_is_not_presented_as_live_and_keeps_data_timestamp(monkeypatch) -> None:
    messages = _capture_status(monkeypatch)
    checked_at = datetime(2026, 9, 11, 10, 42, 30, tzinfo=ZoneInfo("America/New_York"))
    quote = {"price": 101.0, "market_data_timestamp": "2026-09-11T10:41:30-04:00"}

    app.render_live_quote_status("cached", checked_at, quote)

    text = " ".join(value for _, value in messages)
    assert "CACHED" in text
    assert "Data timestamp: 2026-09-11 10:41:30 EDT" in text
    assert "Last checked: 10:42:30 EDT" in text
    assert app.quote_price_label("cached") == "Last Cached Price"
    assert "Latest Live Price" not in text


def test_market_closed_quote_is_not_presented_as_live(monkeypatch) -> None:
    messages = _capture_status(monkeypatch)
    checked_at = datetime(2026, 9, 11, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    quote = {"price": 101.0, "market_data_timestamp": "2026-09-11T15:59:00-04:00"}

    app.render_live_quote_status("market_closed", checked_at, quote)

    text = " ".join(value for _, value in messages)
    assert "MARKET CLOSED" in text
    assert "latest available market price" in text
    assert app.quote_price_label("market_closed") == "Latest Available Price"
    assert "Latest Live Price" not in text


def test_historical_fallback_is_explicitly_historical(monkeypatch) -> None:
    messages = _capture_status(monkeypatch)
    checked_at = datetime(2026, 9, 11, 10, 42, 30, tzinfo=ZoneInfo("America/New_York"))

    app.render_live_quote_status("historical_fallback", checked_at, None, "2026-09-10")

    text = " ".join(value for _, value in messages)
    assert "HISTORICAL FALLBACK" in text
    assert "2026-09-10 00:00:00 EDT" in text
    assert app.quote_price_label("historical_fallback") == "Latest Historical Price"
    assert "Latest Live Price" not in text


def test_market_closed_detection_pauses_live_polling() -> None:
    closed = datetime(2026, 9, 11, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    weekend = datetime(2026, 9, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert not app.us_market_is_open(closed)
    assert not app.us_market_is_open(weekend)


def test_historical_and_fundamental_fetches_are_cached_separately(monkeypatch) -> None:
    historical_calls = 0
    fundamentals_calls = 0
    def fake_history(*args, **kwargs):
        nonlocal historical_calls
        historical_calls += 1
        return _raw_prices()

    def fake_fundamentals(*args, **kwargs):
        nonlocal fundamentals_calls
        fundamentals_calls += 1
        return pd.DataFrame([{"ticker": "AAPL"}])

    monkeypatch.setattr(app, "load_external_ticker_prices", fake_history)
    monkeypatch.setattr(app, "load_external_ticker_fundamentals", fake_fundamentals)
    app.load_external_ticker_features.clear()

    first = app.load_external_ticker_features("AAPL", "2015-01-01", "2026-12-31")
    second = app.load_external_ticker_features("AAPL", "2015-01-01", "2026-12-31")

    assert not first.empty and not second.empty
    assert historical_calls == 1
    assert fundamentals_calls == 1
    assert app.LIVE_HISTORY_TTL == 1800
    assert app.LIVE_FUNDAMENTALS_TTL == 14400


def test_live_quote_cache_is_30_seconds_and_does_not_use_shared_csv(monkeypatch) -> None:
    app.load_live_ticker_quote.clear()
    calls = 0

    def fake_quote(_ticker: str):
        nonlocal calls
        calls += 1
        return {"price": 100.0, "market_data_timestamp": "2026-09-11T15:59:00-04:00"}

    monkeypatch.setattr(app, "fetch_live_ticker_quote", fake_quote)
    first = app.load_live_ticker_quote("AAPL")
    second = app.load_live_ticker_quote("AAPL")

    assert first == second
    assert calls == 1
    assert app.LIVE_QUOTE_TTL == 30


def test_live_quote_fetch_does_not_write_csv(monkeypatch) -> None:
    writes: list[object] = []

    def fail_to_csv(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("live quote fetch must not write CSV files")

    class FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame(
                {"Close": [123.45]},
                index=pd.DatetimeIndex(["2026-09-11 15:59:00"], tz="America/New_York"),
            )

    monkeypatch.setattr(data_loader.yf, "Ticker", lambda ticker: FakeTicker())
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    quote = data_loader.fetch_live_ticker_quote("AAPL")

    assert quote["price"] == 123.45
    assert writes == []


def test_auto_refresh_disabled_preserves_project_dataset(monkeypatch) -> None:
    monkeypatch.setattr(app, "AUTO_REFRESH_DATA", False)
    monkeypatch.setattr(app, "market_dataset_is_stale", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        app,
        "refresh_project_market_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("project refresh must be disabled")),
    )

    features = _raw_prices()
    assert app.maybe_auto_refresh_project_data([], "2015-01-01", features) is features
