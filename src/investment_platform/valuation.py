"""Scenario valuation utilities."""

from __future__ import annotations

from typing import Any

from investment_platform.schemas import ValuationScenario, ValuationSummary


def build_scenarios(raw_items: list[dict[str, object]]) -> list[ValuationScenario]:
    scenarios = [
        ValuationScenario(
            name=str(item["name"]),
            fair_value=float(item["fair_value"]),
            probability=float(item["probability"]),
            driver=str(item.get("driver", "")),
        )
        for item in raw_items
    ]
    total_probability = sum(item.probability for item in scenarios)
    if not 0.99 <= total_probability <= 1.01:
        raise ValueError("valuation scenario probabilities must sum to 1.0")
    return scenarios


def summarize_valuation(ticker: str, scenarios: list[ValuationScenario], current_price: float | None = None) -> ValuationSummary:
    weighted = sum(item.fair_value * item.probability for item in scenarios)
    upside = None if current_price in (None, 0) else weighted / float(current_price) - 1.0
    return ValuationSummary(ticker=ticker, weighted_fair_value=round(weighted, 2), upside_to_price=upside, scenarios=scenarios)


def build_forward_2027_scenarios(
    ticker: str,
    *,
    price: float | None,
    score: float | None = None,
    moomoo_valuation: dict[str, Any] | None = None,
    assumptions: dict[str, Any] | None = None,
) -> list[ValuationScenario]:
    """Build low/base/high 2027E valuation cases from forward earnings assumptions.

    The engine intentionally avoids blindly using provider high/low valuation bands. It
    starts from a forward earnings base, applies three growth tracks, then applies
    reasonable multiple bands with caps so a noisy provider estimate cannot produce
    absurd target prices.
    """

    valuation = moomoo_valuation or {}
    if valuation.get("status") and valuation.get("status") != "ok":
        return []
    assumptions = assumptions or {}
    earnings_base = _first_float(
        assumptions.get("eps_2026e"),
        assumptions.get("fcf_per_share_2026e"),
        valuation.get("eps_fwd1"),
        valuation.get("eps_ttm"),
    )
    if earnings_base is None or earnings_base <= 0:
        return []

    growth = _growth_tracks(ticker, valuation, assumptions, score)
    multiples = _multiple_tracks(ticker, valuation, assumptions, score)
    probabilities = assumptions.get("probabilities", {"low": 0.25, "base": 0.50, "high": 0.25})
    raw_cases = [
        ("Bear", "low", "低增速档：盈利增速放缓/估值回到保守区间"),
        ("Base", "base", "中增速档：2027E盈利按主线需求继续增长"),
        ("Bull", "high", "高增速档：需求或利润率超预期，市场愿意给更高倍数"),
    ]
    scenarios: list[ValuationScenario] = []
    for name, key, driver in raw_cases:
        fair_value = earnings_base * (1.0 + growth[key]) * multiples[key]
        if price:
            fair_value = _cap_against_price(float(fair_value), float(price), key)
        scenarios.append(
            ValuationScenario(
                name=name,
                fair_value=round(float(fair_value), 2),
                probability=float(probabilities.get(key, 0.0)),
                driver=(
                    f"2027E {driver}；"
                    f"基准EPS/FCF {earnings_base:.2f}，增速 {growth[key]:.0%}，倍数 {multiples[key]:.1f}x"
                ),
            )
        )
    return _normalize_probabilities(scenarios)


def _growth_tracks(ticker: str, valuation: dict[str, Any], assumptions: dict[str, Any], score: float | None) -> dict[str, float]:
    if isinstance(assumptions.get("growth"), dict):
        return {
            "low": float(assumptions["growth"].get("low", 0.05)),
            "base": float(assumptions["growth"].get("base", 0.10)),
            "high": float(assumptions["growth"].get("high", 0.18)),
        }
    provider_growth = _first_float((valuation.get("growth") or {}).get("cagr") if isinstance(valuation.get("growth"), dict) else None)
    forward_growth = _forward_eps_growth(valuation)
    raw_growth = _first_float(provider_growth, forward_growth, 0.10) or 0.10
    quality_boost = 0.04 if (score or 0.0) >= 82 else 0.0
    base = min(0.28, max(0.04, raw_growth * 0.45 + quality_boost))
    if ticker.upper() in {"NVDA", "AVGO", "ANET", "MU", "VRT"}:
        base = max(base, 0.16)
    return {"low": max(0.00, base * 0.55), "base": base, "high": min(0.42, base * 1.55)}


def _multiple_tracks(ticker: str, valuation: dict[str, Any], assumptions: dict[str, Any], score: float | None) -> dict[str, float]:
    if isinstance(assumptions.get("multiple"), dict):
        return {
            "low": float(assumptions["multiple"].get("low", 16.0)),
            "base": float(assumptions["multiple"].get("base", 22.0)),
            "high": float(assumptions["multiple"].get("high", 30.0)),
        }
    current_forward = _first_float(valuation.get("forward_pe"), valuation.get("current_pe"), 22.0) or 22.0
    avg_pe = _first_float(valuation.get("avg_pe"), current_forward) or current_forward
    quality = 1.10 if (score or 0.0) >= 82 else 1.0
    base = max(12.0, min(42.0, (current_forward * 0.55 + avg_pe * 0.45) * quality))
    ticker_upper = ticker.upper()
    if ticker_upper in {"NVDA", "AVGO", "ANET", "VRT"}:
        base = max(base, 30.0)
    ticker_caps = {
        "NVDA": 34.0,
        "AVGO": 32.0,
        "ANET": 34.0,
        "VRT": 34.0,
        "MU": 22.0,
        "CSCO": 24.0,
        "BRK.B": 24.0,
        "SCHD": 18.0,
    }
    if ticker_upper in ticker_caps:
        base = min(base, ticker_caps[ticker_upper])
    if ticker_upper in {"CSCO", "BRK.B", "SCHD"}:
        base = min(base, 24.0)
    return {"low": max(8.0, base * 0.78), "base": base, "high": min(58.0, base * 1.25)}


def _forward_eps_growth(valuation: dict[str, Any]) -> float | None:
    fwd1 = _first_float(valuation.get("eps_fwd1"))
    fwd2 = _first_float(valuation.get("eps_fwd2"))
    if fwd1 is None or fwd1 <= 0 or fwd2 is None:
        return None
    return max(-0.20, min(0.80, fwd2 / fwd1 - 1.0))


def _cap_against_price(value: float, price: float, key: str) -> float:
    caps = {"low": 1.35, "base": 1.90, "high": 2.55}
    floors = {"low": 0.35, "base": 0.55, "high": 0.75}
    return min(price * caps[key], max(price * floors[key], value))


def _normalize_probabilities(scenarios: list[ValuationScenario]) -> list[ValuationScenario]:
    total = sum(item.probability for item in scenarios)
    if total <= 0:
        total = 1.0
    return [
        ValuationScenario(
            name=item.name,
            fair_value=item.fair_value,
            probability=round(item.probability / total, 4),
            driver=item.driver,
        )
        for item in scenarios
    ]


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, "", "-"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
