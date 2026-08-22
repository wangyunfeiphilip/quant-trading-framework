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
        return "偏多", "趋势、动量和技术确认度较高，当前技术面偏强。"
    if score <= -4:
        return "偏空", "价格结构和动量信号偏弱，应优先关注回撤风险。"
    return "中性", "指标之间存在分歧，更适合继续观察确认信号。"


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
            bullets.append("趋势：价格位于 SMA20 和 SMA60 上方，短中期均线呈多头排列。")
        elif price < sma_20 < sma_60:
            score -= 2
            bullets.append("趋势：价格位于 SMA20 和 SMA60 下方，短中期均线呈空头排列。")
        elif price > sma_20 and price > sma_60:
            score += 1
            bullets.append("趋势：价格仍在主要均线上方，但均线结构尚未完全顺畅。")
        elif price < sma_20 and price < sma_60:
            score -= 1
            bullets.append("趋势：价格跌破主要均线，短期趋势承压。")
        else:
            bullets.append("趋势：价格与均线位置交错，趋势信号暂不明确。")

    momentum_values = [value for value in (ret_21d, ret_63d, ret_126d) if value is not None]
    if momentum_values:
        positives = sum(value > 0 for value in momentum_values)
        negatives = sum(value < 0 for value in momentum_values)
        if positives == len(momentum_values):
            score += 2
            bullets.append(
                f"动量：21/63/126 日收益均为正，分别为 {_percent(ret_21d)}、{_percent(ret_63d)}、{_percent(ret_126d)}。"
            )
        elif negatives == len(momentum_values):
            score -= 2
            bullets.append(
                f"动量：21/63/126 日收益均为负，分别为 {_percent(ret_21d)}、{_percent(ret_63d)}、{_percent(ret_126d)}。"
            )
        else:
            bullets.append(
                f"动量：不同周期收益分化，21/63/126 日分别为 {_percent(ret_21d)}、{_percent(ret_63d)}、{_percent(ret_126d)}。"
            )

    if rsi is not None:
        if rsi >= 70:
            score -= 1
            bullets.append(f"RSI：{rsi:.1f}，进入超买区，继续追涨的风险上升。")
        elif rsi <= 30:
            score += 1
            bullets.append(f"RSI：{rsi:.1f}，进入超卖区，可能存在均值回归观察点。")
        elif rsi >= 55:
            score += 1
            bullets.append(f"RSI：{rsi:.1f}，多头动能相对占优但未进入极端区间。")
        elif rsi <= 45:
            score -= 1
            bullets.append(f"RSI：{rsi:.1f}，买盘动能偏弱。")
        else:
            bullets.append(f"RSI：{rsi:.1f}，处于中性区间。")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal and (macd_hist is None or macd_hist >= 0):
            score += 1
            bullets.append("MACD：快线位于信号线上方，短期动量确认偏正面。")
        elif macd < macd_signal and (macd_hist is None or macd_hist <= 0):
            score -= 1
            bullets.append("MACD：快线位于信号线下方，短期动量确认偏负面。")
        else:
            bullets.append("MACD：快线与柱状图信号不完全一致，动量确认度一般。")

    if zscore is not None:
        if zscore >= 2:
            score -= 1
            bullets.append(f"布林带：Z-score 为 {zscore:.2f}，价格明显高于滚动均值，短线可能偏拥挤。")
        elif zscore <= -2:
            score += 1
            bullets.append(f"布林带：Z-score 为 {zscore:.2f}，价格明显低于滚动均值，存在反弹观察点。")
        else:
            bullets.append(f"布林带：Z-score 为 {zscore:.2f}，价格没有显著偏离 20 日均值。")

    if vol_21d is not None:
        bullets.append(f"风险：21 日年化波动率约为 {_percent(vol_21d)}，仓位和止损应与波动水平匹配。")

    stance, headline = _level(score)
    return TechnicalSummary(stance=stance, headline=headline, bullets=bullets[:6])
