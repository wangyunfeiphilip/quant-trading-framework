"""Moomoo watchlist, valuation, technical-level, and daily OHLC inputs."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def load_moomoo_watchlist(paths: list[str] | None) -> list[dict[str, Any]]:
    """Load exported Moomoo watchlist rows from JSON, CSV, or XLSX files."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        loaded = _load_watchlist_path(path)
        for row in loaded:
            normalized = _normalize_watchlist_row(row, str(path))
            ticker = normalized.get("ticker")
            if not ticker or ticker in seen:
                continue
            rows.append(normalized)
            seen.add(str(ticker))
    return rows


def build_watchlists_from_moomoo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create dashboard watchlist groups from Moomoo group labels."""

    groups: dict[str, list[str]] = {}
    for row in rows:
        ticker = str(row.get("ticker", ""))
        for group in row.get("groups", []):
            if group in {"All", "持仓"}:
                continue
            groups.setdefault(str(group), []).append(ticker)
    return [
        {"name": f"moomoo {name}", "source": "moomoo_watchlist_export", "tickers": sorted(set(tickers))}
        for name, tickers in sorted(groups.items())
    ]


def load_ohlc_history(paths: list[str] | None, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Load daily candles from Moomoo JSON files or cached OHLC CSV directories."""

    wanted = {ticker.upper().replace("US.", "") for ticker in tickers}
    histories: dict[str, dict[str, Any]] = {}
    for raw_path in paths or []:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        if path.is_dir():
            _load_ohlc_directory(path, wanted, histories)
            continue
        if path.suffix.lower() == ".csv":
            ticker = path.stem.upper().replace("US.", "")
            if ticker in wanted and ticker not in histories:
                candles = _read_ohlc_csv(path)
                if candles:
                    histories[ticker] = {"source": str(path), "candles": candles[-90:], "source_type": "cached_daily_csv"}
            continue
        if path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for ticker in wanted:
            if ticker in histories:
                continue
            candles = _extract_candles(payload, ticker)
            if candles:
                histories[ticker] = {"source": str(path), "candles": candles[-90:], "source_type": "moomoo_daily_json"}
    return histories


def latest_close_by_ticker(histories: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return the newest close for each ticker from loaded daily candles."""

    closes: dict[str, float] = {}
    for ticker, history in histories.items():
        candles = history.get("candles", [])
        if not candles:
            continue
        close = _float(candles[-1].get("close"))
        if close is not None:
            closes[str(ticker).upper().replace("US.", "")] = close
    return closes


def _load_ohlc_directory(path: Path, wanted: set[str], histories: dict[str, dict[str, Any]]) -> None:
    for ticker in wanted:
        if ticker in histories:
            continue
        csv_path = path / f"{ticker}.csv"
        if not csv_path.exists():
            csv_path = path / f"US.{ticker}.csv"
        if not csv_path.exists():
            continue
        candles = _read_ohlc_csv(csv_path)
        if candles:
            histories[ticker] = {"source": str(csv_path), "candles": candles[-90:], "source_type": "cached_daily_csv"}


def _read_ohlc_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    candles = [_normalize_candle(row) for row in rows]
    candles = [item for item in candles if item]
    return sorted(candles, key=lambda item: str(item.get("time", "")))


def estimated_candles(as_of: date, current_price: float, supports: list[dict[str, Any]], resistances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a clearly marked fallback OHLC path when no historical daily bars exist."""

    lower = min([float(item.get("low", current_price)) for item in supports] + [current_price * 0.88])
    upper = max([float(item.get("high", current_price)) for item in resistances] + [current_price * 1.08])
    span = max(upper - lower, current_price * 0.08)
    start = current_price - span * 0.18
    candles: list[dict[str, Any]] = []
    day = as_of - timedelta(days=95)
    last_close = start
    while len(candles) < 60:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
        phase = len(candles)
        drift = (current_price - start) / 60.0
        close = max(lower * 0.98, min(upper * 1.02, last_close + drift + ((phase % 7) - 3) * span * 0.004))
        open_price = last_close
        high = max(open_price, close) + span * (0.012 + (phase % 3) * 0.003)
        low = min(open_price, close) - span * (0.010 + (phase % 4) * 0.002)
        candles.append(
            {
                "time": day.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
            }
        )
        last_close = close
    candles[-1]["close"] = round(current_price, 2)
    candles[-1]["high"] = round(max(candles[-1]["high"], current_price), 2)
    candles[-1]["low"] = round(min(candles[-1]["low"], current_price), 2)
    return candles


def _load_watchlist_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [item for item in payload["rows"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".xlsx":
        return _read_xlsx_first_sheet(path)
    return []


def _normalize_watchlist_row(row: dict[str, Any], source_path: str) -> dict[str, Any]:
    code = str(row.get("code") or row.get("symbol") or row.get("ticker") or "")
    ticker = code.split(".")[-1].upper()
    groups = row.get("groups", [])
    if isinstance(groups, str):
        groups = [part.strip() for part in groups.split(",") if part.strip()]
    price = _float(row.get("last_price", row.get("price")))
    technical = row.get("technical") if isinstance(row.get("technical"), dict) else {}
    valuation = row.get("valuation") if isinstance(row.get("valuation"), dict) else {}
    supports = technical.get("supports") if isinstance(technical, dict) else None
    resistances = technical.get("resistances") if isinstance(technical, dict) else None
    return {
        "ticker": ticker,
        "code": code or f"US.{ticker}",
        "name": row.get("name", ""),
        "type": row.get("stock_type") or row.get("type", ""),
        "groups": list(groups),
        "last_price": price,
        "change_rate": _normalize_change_rate(row.get("change_rate")),
        "trend": (technical or {}).get("trend") or row.get("trend", ""),
        "supports": supports if isinstance(supports, list) else _parse_level_text(str(row.get("support", ""))),
        "resistances": resistances if isinstance(resistances, list) else _parse_level_text(str(row.get("resistance", ""))),
        "indicators": (technical or {}).get("indicators", {}),
        "valuation": valuation,
        "valuation_text": row.get("valuation", "") if not isinstance(row.get("valuation"), dict) else _valuation_text(row.get("valuation", {})),
        "forward_valuation_text": row.get("forward_valuation", ""),
        "action": row.get("action", ""),
        "update_time": str(row.get("update_time", "")),
        "source": "moomoo_watchlist_export",
        "source_path": source_path,
    }


def _extract_candles(payload: dict[str, Any], ticker: str) -> list[dict[str, Any]]:
    keys = [ticker, f"US.{ticker}"]
    containers: list[dict[str, Any]] = []
    for name in ["kline", "daily_klines_formula_75", "daily_klines_unadjusted_core"]:
        item = payload.get(name)
        if isinstance(item, dict):
            containers.append(item)
    for container in containers:
        for key in keys:
            if key not in container:
                continue
            raw = container[key]
            raw_candles = raw.get("data", []) if isinstance(raw, dict) else raw
            candles = [_normalize_candle(item) for item in raw_candles if isinstance(item, dict)]
            candles = [item for item in candles if item]
            if candles:
                return candles
    return []


def _normalize_candle(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "time": str(row.get("time") or row.get("time_key") or row.get("date") or "")[:10],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": _float(row.get("volume")),
        }
    except (KeyError, TypeError, ValueError):
        return {}


def _read_xlsx_first_sheet(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        sheet_names = [name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
        if not sheet_names:
            return []
        root = ET.fromstring(zf.read(sorted(sheet_names)[0]))
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[Any]] = []
    for row_node in root.findall(".//main:sheetData/main:row", ns):
        values: list[Any] = []
        for cell in row_node.findall("main:c", ns):
            ref = cell.attrib.get("r", "")
            index = _column_index(ref)
            while len(values) < index:
                values.append(None)
            values.append(_cell_value(cell, shared, ns))
        rows.append(values)
    if not rows:
        return []
    headers = [str(item or "").strip() for item in rows[0]]
    return [{headers[idx]: value for idx, value in enumerate(row) if idx < len(headers)} for row in rows[1:]]


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("main:si", ns):
        texts = [node.text or "" for node in item.findall(".//main:t", ns)]
        values.append("".join(texts))
    return values


def _cell_value(cell: ET.Element, shared: list[str], ns: dict[str, str]) -> Any:
    cell_type = cell.attrib.get("t")
    value = cell.find("main:v", ns)
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//main:t", ns)]
        return "".join(texts)
    if value is None or value.text is None:
        return None
    text = value.text
    if cell_type == "s":
        try:
            return shared[int(text)]
        except (IndexError, ValueError):
            return text
    number = _float(text)
    return number if number is not None else text


def _column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - ord("A") + 1
    return max(index - 1, 0)


def _parse_level_text(text: str) -> list[dict[str, Any]]:
    levels = []
    for low, high, strength in re.findall(r"([\d,.]+)-([\d,.]+)\(([^)]+)\)", text):
        low_value = _float(low)
        high_value = _float(high)
        if low_value is None or high_value is None:
            continue
        levels.append({"low": low_value, "high": high_value, "center": round((low_value + high_value) / 2, 4), "strength": strength, "labels": []})
    return levels


def _valuation_text(valuation: dict[str, Any]) -> str:
    if not valuation:
        return ""
    if valuation.get("status") and valuation.get("status") != "ok":
        return str(valuation.get("status"))
    parts = []
    if valuation.get("current_pe") is not None:
        parts.append(f"PE {float(valuation['current_pe']):.2f}")
    if valuation.get("forward_pe") is not None:
        parts.append(f"Fwd {float(valuation['forward_pe']):.2f}")
    percentile = valuation.get("pe_percentile") or valuation.get("percentile")
    if percentile is not None:
        parts.append(f"分位 {float(percentile):.1f}%")
    return " / ".join(parts)


def _normalize_change_rate(value: Any) -> float | None:
    number = _float(value)
    if number is None:
        return None
    return number / 100.0 if abs(number) > 1 else number


def _float(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
