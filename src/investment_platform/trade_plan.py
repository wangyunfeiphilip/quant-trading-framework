"""Technical entry, exit, and risk plans for held and discovered stocks."""

from __future__ import annotations

from typing import Any


def build_trade_plans(
    stock_details: list[dict[str, Any]],
    allocation_plan: dict[str, Any],
    theme_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create price-level plans from Moomoo levels, valuation, and portfolio constraints."""

    allocation_by_ticker = {str(row.get("ticker", "")): row for row in allocation_plan.get("rows", [])}
    discovery_by_ticker = {str(row.get("ticker", "")): row for row in theme_candidates}
    plans = []
    for detail in stock_details:
        ticker = str(detail.get("ticker", ""))
        if not ticker:
            continue
        allocation = allocation_by_ticker.get(ticker, {})
        discovery = discovery_by_ticker.get(ticker, {})
        portfolio = detail.get("portfolio") or {}
        if not portfolio and not discovery:
            continue
        plans.append(_build_one_plan(detail, allocation, discovery))
    return sorted(plans, key=_sort_key)


def _build_one_plan(detail: dict[str, Any], allocation: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    ticker = str(detail.get("ticker", ""))
    price = _float(detail.get("price"))
    supports = [item for item in detail.get("supports", []) if isinstance(item, dict)]
    resistances = [item for item in detail.get("resistances", []) if isinstance(item, dict)]
    valuation = detail.get("valuation") if isinstance(detail.get("valuation"), dict) else {}
    fair_value = _float(valuation.get("weighted_fair_value"))
    upside = _float(valuation.get("upside_to_price"))
    allocation_action = str(allocation.get("action", ""))
    is_discovery = bool(discovery)
    diagnosis = _diagnose(detail, allocation, discovery, upside)

    if price is None or not supports or not resistances:
        return {
            "ticker": ticker,
            "source": _source_label(detail, is_discovery),
            "price": price,
            "decision": "no_trade_until_data",
            "buy_zone": "暂无可执行买点",
            "breakout_buy": "暂无",
            "stop_loss": "暂无",
            "trim_zone": "暂无",
            "sell_zone": "暂无",
            "position_note": _position_note(allocation),
            "diagnosis": diagnosis,
            "problem_area": diagnosis["problem_area"],
            "reason": "缺少当前价格、日K、支撑位或压力位；先加入观察，不给硬编点位。",
            "data_quality": "missing_levels",
        }

    support = _nearest_support(price, supports)
    next_support = _next_level_below(support, supports)
    resistance = _nearest_resistance(price, resistances)
    next_resistance = _next_level_above(resistance, resistances)

    buy_low = _float(support.get("low")) or price * 0.96
    buy_high = min(_float(support.get("high")) or price * 0.99, price * 1.01)
    stop_base = _float(next_support.get("low")) if next_support else buy_low
    stop_loss = min(buy_low * 0.975, stop_base * 0.985)
    breakout_level = (_float(resistance.get("high")) or price * 1.04) * 1.005
    trim_low = _float(resistance.get("low")) or price * 1.03
    trim_high = _float(resistance.get("high")) or price * 1.06
    if next_resistance:
        trim_high = max(trim_high, _float(next_resistance.get("low")) or trim_high)
    valuation_outlier = bool(fair_value and fair_value > price * 2.0)
    sell_zone = _sell_zone(None if valuation_outlier else fair_value, trim_high, price)
    decision = _decision(
        allocation_action=allocation_action,
        is_discovery=is_discovery,
        price=price,
        buy_low=buy_low,
        buy_high=buy_high,
        resistance=resistance,
        upside=upside,
        diagnosis=diagnosis,
        allocation=allocation,
    )
    reason = _reason(
        decision=decision,
        ticker=ticker,
        price=price,
        support=support,
        resistance=resistance,
        fair_value=fair_value,
        upside=upside,
        allocation=allocation,
        discovery=discovery,
        diagnosis=diagnosis,
    )
    return {
        "ticker": ticker,
        "source": _source_label(detail, is_discovery),
        "price": round(price, 2),
        "decision": decision,
        "buy_zone": _range_text(buy_low, buy_high),
        "breakout_buy": _money(breakout_level),
        "stop_loss": _money(stop_loss),
        "trim_zone": _range_text(trim_low, trim_high),
        "sell_zone": sell_zone,
        "position_note": _position_note(allocation),
        "diagnosis": diagnosis,
        "problem_area": diagnosis["problem_area"],
        "reason": reason,
        "valuation_outlier": valuation_outlier,
        "support_labels": ", ".join(str(label) for label in support.get("labels", [])[:4]),
        "resistance_labels": ", ".join(str(label) for label in resistance.get("labels", [])[:4]),
        "data_quality": "level_based",
    }


def _decision(
    *,
    allocation_action: str,
    is_discovery: bool,
    price: float,
    buy_low: float,
    buy_high: float,
    resistance: dict[str, Any],
    upside: float | None,
    diagnosis: dict[str, Any],
    allocation: dict[str, Any],
) -> str:
    resistance_low = _float(resistance.get("low")) or price * 1.03
    current = _float(allocation.get("current_allocation")) or 0.0
    max_allocation = _float(allocation.get("max_allocation")) or max(current, 0.0)
    if diagnosis["fundamental"] == "problem":
        return "reduce_fundamental_risk"
    if is_discovery and not allocation and diagnosis["news"] == "problem":
        return "watch_only"
    if diagnosis["news"] == "problem" and price >= resistance_low * 0.98:
        return "trim_speculative_strength"
    if diagnosis["valuation"] == "problem" and diagnosis["fundamental"] != "strong":
        return "trim_valuation_risk"
    if str(allocation.get("bucket", "")) == "退出观察/不新增":
        return "reduce_low_conviction"
    if allocation_action == "wait_for_cash_or_trim":
        return "wait_for_cash"
    if buy_low <= price <= buy_high and diagnosis["valuation"] != "problem":
        if current > max_allocation and diagnosis["fundamental"] != "strong":
            return "hold_risk_limit_no_add"
        return "staged_buy_near_support"
    if price >= resistance_low * 0.99 and not (diagnosis["fundamental"] == "strong" and (upside or 0.0) >= 0.18):
        return "do_not_chase_wait_breakout"
    if is_discovery and (upside is None or upside < 0.10):
        return "watch_only"
    if allocation_action == "trim_to_target":
        return "hold_winner_no_forced_trim"
    if diagnosis["fundamental"] == "strong" and diagnosis["valuation"] == "attractive":
        return "buy_pullback_or_breakout"
    return "wait_for_pullback"


def _reason(
    *,
    decision: str,
    ticker: str,
    price: float,
    support: dict[str, Any],
    resistance: dict[str, Any],
    fair_value: float | None,
    upside: float | None,
    allocation: dict[str, Any],
    discovery: dict[str, Any],
    diagnosis: dict[str, Any],
) -> str:
    support_text = _range_text(_float(support.get("low")) or price, _float(support.get("high")) or price)
    resistance_text = _range_text(_float(resistance.get("low")) or price, _float(resistance.get("high")) or price)
    parts = [f"现价接近 {support_text} 支撑和 {resistance_text} 压力之间。"]
    parts.append(
        "诊断："
        f"基本面{diagnosis['fundamental_label']}，"
        f"估值{diagnosis['valuation_label']}，"
        f"技术面{diagnosis['technical_label']}，"
        f"消息面{diagnosis['news_label']}，"
        f"仓位{diagnosis['position_label']}。"
    )
    if fair_value:
        if fair_value > price * 2.0:
            parts.append(f"模型合理价 {_money(fair_value)} 距现价过远，先视为估值离群，不作为卖出点位。")
        else:
            parts.append(f"概率加权合理价约 {_money(fair_value)}，相对现价空间 {upside or 0:.1%}。")
    if allocation:
        parts.append(
            f"组合目标仓位 {float(allocation.get('target_allocation', 0.0)):.1%}，当前 {float(allocation.get('current_allocation', 0.0)):.1%}。"
        )
    if discovery:
        parts.append(f"潜力主题：{discovery.get('theme', '')}；下一步核验：{discovery.get('next_check', '')}。")
    action_reason = {
        "reduce_fundamental_risk": "卖出/减仓原因来自基本面或投资逻辑走弱，不是因为占比高。",
        "trim_valuation_risk": "减仓原因来自估值没有安全边际，除非后续盈利预期上修，否则不应只因价格回调就补仓。",
        "trim_speculative_strength": "消息/情绪偏投机且价格靠近压力，反弹更适合减风险而不是追。",
        "reduce_low_conviction": "降权原因来自低确定性或主题重复，不是单纯因为仓位高；只有后续基本面、估值和催化证据明显改善，才重新进入核心观察。",
        "hold_winner_no_forced_trim": "仓位高只是约束，不构成卖出理由；基本面和估值没有坏掉时，先持有强票，不机械减仓。",
        "hold_risk_limit_no_add": "价格在支撑区，但仓位已接近/超过风险上限，先持有，不因为便宜一点就破坏总组合波动目标。",
        "wait_for_cash": "现金和再平衡资金不足，先等减仓成交或新增现金。",
        "staged_buy_near_support": "价格在支撑区附近，且没有基本面/估值硬伤，可以小仓分批，但必须接受支撑跌破后的止损。",
        "do_not_chase_wait_breakout": "价格靠近压力，不适合追；要么回落到支撑，要么放量站上突破买点。",
        "watch_only": "主题有潜力但估值/证据不足，先观察，不急于建仓。",
        "buy_pullback_or_breakout": "公司逻辑强、2027E估值仍有空间，适合等回调到支撑或有效突破后加仓。",
        "wait_for_pullback": "不在支撑区也未有效突破，等待回调买点更合理。",
    }.get(decision, "")
    parts.append(action_reason)
    return " ".join(part for part in parts if part)


def _diagnose(
    detail: dict[str, Any],
    allocation: dict[str, Any],
    discovery: dict[str, Any],
    upside: float | None,
) -> dict[str, Any]:
    thesis = detail.get("thesis") if isinstance(detail.get("thesis"), dict) else {}
    sentiment = detail.get("sentiment") if isinstance(detail.get("sentiment"), dict) else {}
    trend = str(detail.get("trend", ""))
    score = _float(detail.get("score"))
    current = _float(allocation.get("current_allocation")) or _float((detail.get("portfolio") or {}).get("allocation")) or 0.0
    target = _float(allocation.get("target_allocation")) or current
    max_allocation = _float(allocation.get("max_allocation")) or max(target, current)

    thesis_status = str(thesis.get("status", "active"))
    if thesis_status == "weakened":
        fundamental = "problem"
        fundamental_label = "有问题"
    elif (score or 0.0) >= 78 or (bool(thesis) and thesis_status == "active"):
        fundamental = "strong"
        fundamental_label = "较强"
    else:
        fundamental = "neutral"
        fundamental_label = "中性"

    if upside is None:
        valuation = "unknown"
        valuation_label = "不确定"
    elif upside <= -0.12:
        valuation = "problem"
        valuation_label = "偏贵/安全边际不足"
    elif upside >= 0.15:
        valuation = "attractive"
        valuation_label = "有吸引力"
    else:
        valuation = "fair"
        valuation_label = "基本合理"

    if any(token in trend for token in ["偏弱", "承压", "破位"]):
        technical = "problem"
        technical_label = "偏弱"
    elif any(token in trend for token in ["偏强", "上行", "突破"]):
        technical = "strong"
        technical_label = "较强"
    else:
        technical = "neutral"
        technical_label = "中性"

    if sentiment.get("speculation_risk") == "high" or sentiment.get("label") == "Speculation driven":
        news = "problem"
        news_label = "投机/消息风险高"
    elif sentiment.get("label") == "Fundamental driven":
        news = "strong"
        news_label = "基本面驱动"
    else:
        news = "neutral"
        news_label = "中性"

    if current > max_allocation:
        position = "above_hard_cap"
        position_label = "超过风险上限"
    elif current > target:
        position = "above_soft_target"
        position_label = "高于软目标"
    elif discovery:
        position = "new_idea"
        position_label = "潜力股观察"
    else:
        position = "ok"
        position_label = "合理"

    bucket = str(allocation.get("bucket", ""))
    if fundamental == "problem":
        problem_area = "基本面"
    elif valuation == "problem":
        problem_area = "估值"
    elif news == "problem":
        problem_area = "消息面"
    elif technical == "problem":
        problem_area = "技术面"
    elif bucket == "退出观察/不新增":
        problem_area = "组合精简"
    elif position in {"above_hard_cap", "above_soft_target"}:
        problem_area = "仓位约束"
    else:
        problem_area = "无明显问题"

    return {
        "fundamental": fundamental,
        "fundamental_label": fundamental_label,
        "valuation": valuation,
        "valuation_label": valuation_label,
        "technical": technical,
        "technical_label": technical_label,
        "news": news,
        "news_label": news_label,
        "position": position,
        "position_label": position_label,
        "problem_area": problem_area,
    }


def _sell_zone(fair_value: float | None, trim_high: float, price: float) -> str:
    if fair_value and fair_value > price:
        low = max(trim_high, fair_value * 0.92)
        high = fair_value * 1.03
        return _range_text(low, high)
    return _range_text(trim_high, trim_high * 1.05)


def _nearest_support(price: float, supports: list[dict[str, Any]]) -> dict[str, Any]:
    return min(supports, key=lambda item: abs((_float(item.get("center")) or _float(item.get("high")) or price) - price))


def _nearest_resistance(price: float, resistances: list[dict[str, Any]]) -> dict[str, Any]:
    above = [item for item in resistances if (_float(item.get("high")) or 0.0) >= price * 0.995]
    return min(above or resistances, key=lambda item: abs((_float(item.get("center")) or _float(item.get("low")) or price) - price))


def _next_level_below(level: dict[str, Any], levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    center = _float(level.get("center")) or _float(level.get("low")) or 0.0
    below = [item for item in levels if (_float(item.get("center")) or 0.0) < center]
    return max(below, key=lambda item: _float(item.get("center")) or 0.0) if below else None


def _next_level_above(level: dict[str, Any], levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    center = _float(level.get("center")) or _float(level.get("high")) or 0.0
    above = [item for item in levels if (_float(item.get("center")) or 0.0) > center]
    return min(above, key=lambda item: _float(item.get("center")) or 0.0) if above else None


def _position_note(allocation: dict[str, Any]) -> str:
    if not allocation:
        return "非持仓/潜力股观察"
    return f"当前 {float(allocation.get('current_allocation', 0.0)):.1%} / 目标 {float(allocation.get('target_allocation', 0.0)):.1%}"


def _source_label(detail: dict[str, Any], is_discovery: bool) -> str:
    if detail.get("portfolio"):
        return "持仓股"
    if is_discovery:
        return "潜力股"
    return "自选股"


def _range_text(low: float, high: float) -> str:
    return f"{_money(min(low, high))}-{_money(max(low, high))}"


def _money(value: float | None) -> str:
    if value is None:
        return "暂无"
    return f"${value:,.2f}"


def _float(value: Any) -> float | None:
    try:
        if value in (None, "", "-"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
    source_rank = {"持仓股": 0, "潜力股": 1, "自选股": 2}.get(str(item.get("source")), 3)
    return (source_rank, str(item.get("ticker", "")))
