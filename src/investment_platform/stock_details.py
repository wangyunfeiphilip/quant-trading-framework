"""Per-stock research detail records for the dashboard."""

from __future__ import annotations

from datetime import date
from typing import Any

from investment_platform.moomoo_inputs import estimated_candles


def build_stock_details(
    *,
    as_of: date,
    scores: list[Any],
    theses: list[Any],
    valuations: list[Any],
    sizing: list[Any],
    sentiment: list[Any],
    portfolio: list[dict[str, Any]],
    watchlists: list[dict[str, Any]],
    moomoo_rows: list[dict[str, Any]],
    ohlc_history: dict[str, dict[str, Any]],
    catalysts: list[dict[str, Any]],
    smart_money: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge research, portfolio, Moomoo levels, and OHLC into ticker detail cards."""

    score_by_ticker = {item.ticker: item for item in scores}
    thesis_by_ticker = {item.ticker: item for item in theses}
    valuation_by_ticker = {item.ticker: item for item in valuations}
    sizing_by_ticker = {item.ticker: item for item in sizing}
    sentiment_by_ticker = {item.ticker: item for item in sentiment}
    portfolio_by_ticker = {str(item.get("ticker", "")): item for item in portfolio}
    moomoo_by_ticker = {str(item.get("ticker", "")): item for item in moomoo_rows}
    catalyst_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for item in catalysts:
        catalyst_by_ticker.setdefault(str(item.get("ticker", "")), []).append(item)
    smart_money_by_ticker = {str(item.get("ticker", "")): item for item in smart_money}
    watchlist_by_ticker = _watchlist_memberships(watchlists)

    tickers = sorted(
        {
            *score_by_ticker,
            *thesis_by_ticker,
            *valuation_by_ticker,
            *sizing_by_ticker,
            *portfolio_by_ticker,
            *moomoo_by_ticker,
            *watchlist_by_ticker,
        }
    )
    details = []
    for ticker in tickers:
        moomoo = moomoo_by_ticker.get(ticker, {})
        history = ohlc_history.get(ticker)
        price = _first_number(
            _latest_history_close(history),
            moomoo.get("last_price"),
            portfolio_by_ticker.get(ticker, {}).get("ibkr_price"),
            _valuation_price(valuation_by_ticker.get(ticker)),
        )
        supports = moomoo.get("supports", []) or []
        resistances = moomoo.get("resistances", []) or []
        valuation = _valuation_payload(ticker, valuation_by_ticker.get(ticker), moomoo, price)
        if history:
            candles = history["candles"]
            candle_source = _candle_source_label(history)
            candle_source_path = history.get("source", "")
        elif price:
            candles = estimated_candles(as_of, float(price), supports, resistances)
            candle_source = "点位估算日K，待接入实时历史日线"
            candle_source_path = ""
        else:
            candles = []
            candle_source = "暂无日K"
            candle_source_path = ""
        score = score_by_ticker.get(ticker)
        details.append(
            {
                "ticker": ticker,
                "code": moomoo.get("code", f"US.{ticker}"),
                "name": moomoo.get("name", ""),
                "price": price,
                "change_rate": moomoo.get("change_rate"),
                "trend": moomoo.get("trend", ""),
                "groups": moomoo.get("groups", []),
                "watchlists": watchlist_by_ticker.get(ticker, []),
                "portfolio": portfolio_by_ticker.get(ticker),
                "score": getattr(score, "total_score", None),
                "factor_scores": getattr(score, "factor_scores", {}),
                "score_explanation": getattr(score, "explanation", []),
                "valuation": valuation,
                "supports": supports,
                "resistances": resistances,
                "indicators": moomoo.get("indicators", {}),
                "action": moomoo.get("action", ""),
                "thesis": _dataclass_payload(thesis_by_ticker.get(ticker)),
                "sizing": _dataclass_payload(sizing_by_ticker.get(ticker)),
                "sentiment": _dataclass_payload(sentiment_by_ticker.get(ticker)),
                "catalysts": catalyst_by_ticker.get(ticker, []),
                "smart_money": smart_money_by_ticker.get(ticker),
                "candles": candles,
                "candle_source": candle_source,
                "candle_source_path": candle_source_path,
                "data_sources": _sources_for(ticker, moomoo, portfolio_by_ticker.get(ticker), candle_source_path),
            }
        )
    return sorted(details, key=lambda item: _sort_key(item))


def _watchlist_memberships(watchlists: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for watchlist in watchlists:
        for ticker in watchlist.get("tickers", []):
            output.setdefault(str(ticker), []).append(str(watchlist.get("name", "")))
    return output


def _valuation_payload(ticker: str, existing: Any, moomoo: dict[str, Any], price: float | None) -> dict[str, Any]:
    if existing:
        scenarios = [
            {
                "name": item.name,
                "fair_value": item.fair_value,
                "probability": item.probability,
                "driver": item.driver,
            }
            for item in existing.scenarios
        ]
        return {
            "ticker": ticker,
            "weighted_fair_value": existing.weighted_fair_value,
            "upside_to_price": existing.upside_to_price,
            "scenarios": scenarios,
            "source": "scenario_model",
        }
    valuation = moomoo.get("valuation", {}) if isinstance(moomoo.get("valuation"), dict) else {}
    scenarios = _scenarios_from_moomoo_valuation(valuation)
    if scenarios:
        weighted = sum(item["fair_value"] * item["probability"] for item in scenarios)
        upside = None if not price else weighted / float(price) - 1.0
        return {
            "ticker": ticker,
            "weighted_fair_value": round(weighted, 2),
            "upside_to_price": upside,
            "scenarios": scenarios,
            "source": "moomoo_forward_valuation",
            "valuation_text": moomoo.get("valuation_text", ""),
            "conclusion": valuation.get("conclusion", ""),
        }
    return {
        "ticker": ticker,
        "weighted_fair_value": None,
        "upside_to_price": None,
        "scenarios": [],
        "source": "not_available",
        "valuation_text": moomoo.get("valuation_text", ""),
        "conclusion": valuation.get("conclusion", "") if isinstance(valuation, dict) else "",
    }


def _scenarios_from_moomoo_valuation(valuation: dict[str, Any]) -> list[dict[str, Any]]:
    low = _midpoint(valuation.get("fwd1_low"))
    base = _midpoint(valuation.get("fwd1_base"))
    high = _midpoint(valuation.get("fwd1_high"))
    if not (low and base and high):
        low = _midpoint(valuation.get("fwd2_low"))
        base = _midpoint(valuation.get("fwd2_base"))
        high = _midpoint(valuation.get("fwd2_high"))
    if not (low and base and high):
        return []
    return [
        {"name": "Bull", "fair_value": round(high, 2), "probability": 0.25, "driver": "高估值带/盈利继续上修"},
        {"name": "Base", "fair_value": round(base, 2), "probability": 0.55, "driver": "中枢估值带"},
        {"name": "Bear", "fair_value": round(low, 2), "probability": 0.20, "driver": "低估值带/预期下修"},
    ]


def _midpoint(value: Any) -> float | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return (float(value[0]) + float(value[1])) / 2.0
    except (TypeError, ValueError):
        return None


def _valuation_price(existing: Any) -> float | None:
    return None if not existing else None


def _latest_history_close(history: dict[str, Any] | None) -> float | None:
    if not history:
        return None
    candles = history.get("candles", [])
    if not candles:
        return None
    return _first_number(candles[-1].get("close"))


def _candle_source_label(history: dict[str, Any]) -> str:
    source_type = str(history.get("source_type", ""))
    if source_type == "cached_daily_csv":
        return "本地缓存真实日K"
    if source_type == "moomoo_daily_json":
        return "Moomoo 历史日K"
    return "历史日K"


def _dataclass_payload(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if hasattr(item, "__dict__"):
        output = dict(item.__dict__)
        if "invalidation_conditions" in output:
            output["invalidation_conditions"] = [dict(condition.__dict__) for condition in output["invalidation_conditions"]]
        return output
    return None


def _sources_for(ticker: str, moomoo: dict[str, Any], portfolio: dict[str, Any] | None, candle_path: str) -> list[str]:
    sources = []
    if portfolio:
        sources.append("IBKR 本地持仓快照")
    if moomoo:
        sources.append("Moomoo 自选股估值/点位导出")
    if candle_path:
        sources.append("Moomoo 日K采集文件")
    if not sources:
        sources.append("平台配置")
    return sources


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value in (None, "", "-"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    in_portfolio = 0 if item.get("portfolio") else 1
    score = item.get("score")
    return (in_portfolio, -float(score or 0.0), str(item.get("ticker", "")))
