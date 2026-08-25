"""Technical signal interpretation helpers for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class TechnicalSummary:
    """Human-readable interpretation of the latest technical indicators."""

    stance: str
    headline: str
    bullets: list[str]


def _number(row: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _percent(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2%}"


def _level(score: int) -> tuple[str, str]:
    if score >= 4:
        return "Bullish", "Trend, momentum, and confirmation signals are aligned to the upside."
    if score <= -4:
        return "Bearish", "Price structure and momentum are weak; downside risk deserves priority."
    return "Neutral", "Signals are mixed, so confirmation matters more than directional conviction."


def generate_technical_summary(row: Mapping[str, object]) -> TechnicalSummary:
    """Convert the latest indicator row into a concise research conclusion."""

    price = _number(row, "adjusted_close")
    sma_20 = _number(row, "sma_20")
    sma_60 = _number(row, "sma_60")
    rsi = _number(row, "rsi_14")
    macd = _number(row, "macd")
    macd_signal = _number(row, "macd_signal")
    macd_hist = _number(row, "macd_histogram")
    zscore = _number(row, "zscore_20")
    vol_21d = _number(row, "volatility_21d")
    ret_21d = _number(row, "return_21d")
    ret_63d = _number(row, "return_63d")
    ret_126d = _number(row, "return_126d")

    score = 0
    bullets: list[str] = []

    if price is not None and sma_20 is not None and sma_60 is not None:
        if price > sma_20 > sma_60:
            score += 2
            bullets.append("Trend: price is above SMA20 and SMA60, with a bullish short-to-medium-term moving-average stack.")
        elif price < sma_20 < sma_60:
            score -= 2
            bullets.append("Trend: price is below SMA20 and SMA60, with a bearish short-to-medium-term moving-average stack.")
        elif price > sma_20 and price > sma_60:
            score += 1
            bullets.append("Trend: price remains above the major moving averages, but the structure is not fully aligned.")
        elif price < sma_20 and price < sma_60:
            score -= 1
            bullets.append("Trend: price has broken below the major moving averages, leaving the short-term trend under pressure.")
        else:
            bullets.append("Trend: price and moving averages are crossed, so the trend signal is not yet clear.")

    momentum_values = [value for value in (ret_21d, ret_63d, ret_126d) if value is not None]
    if momentum_values:
        positives = sum(value > 0 for value in momentum_values)
        negatives = sum(value < 0 for value in momentum_values)
        if positives == len(momentum_values):
            score += 2
            bullets.append(
                f"Momentum: 21/63/126-day returns are all positive at {_percent(ret_21d)}, {_percent(ret_63d)}, and {_percent(ret_126d)}."
            )
        elif negatives == len(momentum_values):
            score -= 2
            bullets.append(
                f"Momentum: 21/63/126-day returns are all negative at {_percent(ret_21d)}, {_percent(ret_63d)}, and {_percent(ret_126d)}."
            )
        else:
            bullets.append(
                f"Momentum: return windows are mixed; 21/63/126-day returns are {_percent(ret_21d)}, {_percent(ret_63d)}, and {_percent(ret_126d)}."
            )

    if rsi is not None:
        if rsi >= 70:
            score -= 1
            bullets.append(f"RSI: {rsi:.1f}, in overbought territory; chasing strength carries higher short-term risk.")
        elif rsi <= 30:
            score += 1
            bullets.append(f"RSI: {rsi:.1f}, in oversold territory; this can be a mean-reversion watch point.")
        elif rsi >= 55:
            score += 1
            bullets.append(f"RSI: {rsi:.1f}, positive momentum has the edge without reaching an extreme zone.")
        elif rsi <= 45:
            score -= 1
            bullets.append(f"RSI: {rsi:.1f}, buying momentum is relatively weak.")
        else:
            bullets.append(f"RSI: {rsi:.1f}, in a neutral range.")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal and (macd_hist is None or macd_hist >= 0):
            score += 1
            bullets.append("MACD: the fast line is above the signal line, confirming positive short-term momentum.")
        elif macd < macd_signal and (macd_hist is None or macd_hist <= 0):
            score -= 1
            bullets.append("MACD: the fast line is below the signal line, confirming negative short-term momentum.")
        else:
            bullets.append("MACD: the fast line and histogram do not fully agree, so confirmation is moderate.")

    if zscore is not None:
        if zscore >= 2:
            score -= 1
            bullets.append(f"Bollinger/Z-score: {zscore:.2f}; price is materially above its rolling mean and may be crowded short term.")
        elif zscore <= -2:
            score += 1
            bullets.append(f"Bollinger/Z-score: {zscore:.2f}; price is materially below its rolling mean, creating a rebound watch point.")
        else:
            bullets.append(f"Bollinger/Z-score: {zscore:.2f}; price is not materially stretched versus its 20-day mean.")

    if vol_21d is not None:
        bullets.append(f"Risk: 21-day annualized volatility is about {_percent(vol_21d)}; position sizing should match the volatility regime.")

    stance, headline = _level(score)
    return TechnicalSummary(stance=stance, headline=headline, bullets=bullets[:6])
