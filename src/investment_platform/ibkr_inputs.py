"""Offline IBKR position inputs until the live IBKR adapter is connected."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def load_ibkr_account(path: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the newest read-only IBKR account snapshot when available."""

    if not path:
        return [], {}
    source = Path(path).expanduser()
    if not source.exists():
        return [], {}
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return [], {}
        summary = payload.get("account_summary", {})
        positions = payload.get("positions", [])
        if not isinstance(summary, dict) or not isinstance(positions, list):
            return [], {}
        rows = _normalize_json_positions(positions, summary, source)
        return rows, {**summary, "source": payload.get("source", "IBKR read-only snapshot"), "as_of": payload.get("as_of")}
    return load_ibkr_positions(path), {}


def load_ibkr_positions(path: str | None) -> list[dict[str, Any]]:
    """Load a local IBKR position snapshot from an existing Python report script."""

    if not path:
        return []
    source = Path(path).expanduser()
    if not source.exists():
        return []
    positions = _extract_positions(source)
    total_market_value = sum(float(item.get("mv", 0.0) or 0.0) for item in positions)
    rows = []
    for item in positions:
        ticker = str(item.get("symbol", "")).upper()
        market_value = float(item.get("mv", 0.0) or 0.0)
        rows.append(
            {
                "ticker": ticker,
                "quantity": item.get("qty"),
                "avg_cost": item.get("avg"),
                "ibkr_price": item.get("ibkr_px"),
                "market_value": market_value,
                "unrealized_pnl": item.get("upl"),
                "allocation": round(market_value / total_market_value, 4) if total_market_value else 0.0,
                "sector": _sector_for(ticker),
                "sector_exposure": 0.0,
                "source": "ibkr_local_snapshot",
                "source_path": str(source),
            }
        )
    sector_totals: dict[str, float] = {}
    for item in rows:
        sector_totals[item["sector"]] = sector_totals.get(item["sector"], 0.0) + float(item.get("allocation", 0.0))
    for item in rows:
        item["sector_exposure"] = round(sector_totals.get(item["sector"], 0.0), 4)
    return rows


def _normalize_json_positions(positions: list[dict[str, Any]], summary: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    net_liquidation = float(summary.get("net_liquidation", 0.0) or 0.0)
    gross_position_value = float(summary.get("gross_position_value", 0.0) or 0.0)
    denominator = net_liquidation or gross_position_value or sum(float(item.get("market_value", 0.0) or 0.0) for item in positions)
    rows = []
    for item in positions:
        ticker = _normalize_ticker(str(item.get("ticker") or item.get("contract_description") or "").upper())
        market_value = float(item.get("market_value", 0.0) or 0.0)
        rows.append(
            {
                "ticker": ticker,
                "quantity": item.get("quantity") or item.get("position"),
                "avg_cost": item.get("average_price"),
                "ibkr_price": item.get("market_price"),
                "market_value": market_value,
                "unrealized_pnl": item.get("unrealized_pnl"),
                "daily_pnl": item.get("daily_pnl"),
                "allocation": round(market_value / denominator, 4) if denominator else 0.0,
                "sector": _sector_for(ticker),
                "sector_exposure": 0.0,
                "source": "ibkr_live_snapshot",
                "source_path": str(source),
            }
        )
    sector_totals: dict[str, float] = {}
    for item in rows:
        sector_totals[item["sector"]] = sector_totals.get(item["sector"], 0.0) + float(item.get("allocation", 0.0))
    for item in rows:
        item["sector_exposure"] = round(sector_totals.get(item["sector"], 0.0), 4)
    return rows


def _normalize_ticker(value: str) -> str:
    if value == "BRK B":
        return "BRK.B"
    return value.replace(" ", ".")


def _extract_positions(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "POSITIONS":
                    value = ast.literal_eval(node.value)
                    return value if isinstance(value, list) else []
    return []


def _sector_for(ticker: str) -> str:
    mapping = {
        "AMZN": "Consumer Internet",
        "AVGO": "Semiconductors",
        "BRK.B": "Diversified Financials",
        "COHR": "Optical / AI Infrastructure",
        "CSCO": "Networking",
        "DRAM": "Semiconductors",
        "GLW": "Optical / Connectivity",
        "GOOGL": "Internet / AI",
        "ISRG": "Healthcare",
        "LITE": "Optical / AI Infrastructure",
        "LYTE": "Speculative Growth",
        "LLY": "Healthcare",
        "LUNR": "Space / Speculative",
        "MSFT": "Software / AI",
        "MU": "Semiconductors",
        "NOK": "Networking",
        "NVDA": "Semiconductors",
        "ORCL": "Software / Cloud",
        "QQQ": "ETF",
        "SCHD": "Dividend ETF",
        "SGOV": "Cash / T-Bills",
        "SNDK": "Storage",
        "SOXX": "ETF",
        "TSM": "Semiconductors",
        "VST": "Power / Utilities",
    }
    return mapping.get(ticker, "Unknown")
