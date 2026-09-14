"""Post-earnings review queue and verdict helpers."""

from __future__ import annotations

from datetime import date
from typing import Any


ANALYSIS_TOOLS = [
    "financial-research-agent: SEC/10-Q/10-K RAG、FinBERT语气与风险语言变化",
    "ai-berkshire earnings-review: 财报精读、现金流质量、管理层承诺追踪",
    "ai-berkshire investment-checklist: 护城河、长期投资价值和安全边际复核",
]


def build_earnings_reviews(
    earnings_calendar: list[dict[str, Any]],
    review_inputs: list[dict[str, Any]],
    tracked_tickers: set[str],
    as_of: date,
) -> list[dict[str, Any]]:
    """Build post-earnings analysis rows for tracked holdings/watchlist names."""

    inputs_by_ticker = {str(item.get("ticker", "")).upper(): item for item in review_inputs if item.get("ticker")}
    reviews: list[dict[str, Any]] = []
    for event in earnings_calendar:
        ticker = str(event.get("ticker", "")).upper()
        if not ticker or ticker not in tracked_tickers:
            continue
        event_date = _parse_date(str(event.get("date", "")))
        status = str(event.get("status", ""))
        has_reported = status in {"reported_needs_analysis", "analyzed", "reported"} or (
            event_date is not None and event_date < as_of
        )
        if not has_reported and ticker not in inputs_by_ticker:
            continue
        reviews.append(_build_review(event, inputs_by_ticker.get(ticker, {}), as_of))
    return sorted(reviews, key=lambda item: str(item.get("date", "")), reverse=True)


def enrich_earnings_reviews_with_news(
    reviews: list[dict[str, Any]],
    news_by_ticker: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched = []
    for review in reviews:
        ticker = str(review.get("ticker", "")).upper()
        news_items = news_by_ticker.get(ticker, [])
        if not news_items:
            enriched.append(review)
            continue
        source_note = str(review.get("source_note", ""))
        if "Moomoo" not in source_note:
            source_note = f"{source_note}；已接入 Moomoo 财报/电话会相关报道"
        key_points = list(review.get("key_points", []))
        key_points.append(f"Moomoo 已抓到 {len(news_items)} 条财报/电话会相关报道，可作为辅助证据")
        next_steps = list(review.get("next_steps", []))
        next_steps.append("阅读 Moomoo 报道和电话会摘要，核对是否与公司原始财报一致")
        enriched.append({**review, "moomoo_news": news_items, "source_note": source_note, "key_points": key_points, "next_steps": next_steps})
    return enriched


def _build_review(event: dict[str, Any], source: dict[str, Any], as_of: date) -> dict[str, Any]:
    ticker = str(event.get("ticker", "")).upper()
    metrics = source.get("metrics", {}) if isinstance(source.get("metrics"), dict) else {}
    signals = _metric_signals(metrics)
    data_available = _has_source_evidence(source, metrics)
    missing_data = _missing_data(metrics)

    if data_available:
        score = _quality_score(signals, source)
        verdict = str(source.get("verdict") or _verdict(score))
        thesis_impact = str(source.get("thesis_impact") or _thesis_impact(score))
        investment_value = str(source.get("investment_value") or _investment_value(score))
        action = str(source.get("action") or _action(score))
        status = "analyzed"
    else:
        score = None
        verdict = "待原始数据"
        thesis_impact = "待确认"
        investment_value = "不下投资结论"
        action = "等待财报原文/10-Q/电话会纪要进入系统，不因标题新闻交易"
        status = "queued"

    watch_items = event.get("watch_items", [])
    if isinstance(watch_items, list):
        focus = "；".join(str(item) for item in watch_items)
    else:
        focus = str(watch_items)

    return {
        "ticker": ticker,
        "company": event.get("company", ticker),
        "date": event.get("date", as_of.isoformat()),
        "time": event.get("time", ""),
        "status": status,
        "verdict": verdict,
        "quality_score": score,
        "thesis_impact": thesis_impact,
        "investment_value": investment_value,
        "action": action,
        "focus": focus,
        "key_points": source.get("key_points", _default_key_points(signals, missing_data)),
        "red_flags": source.get("red_flags", [] if data_available else ["尚未取得可核验财报指标/电话会纪要，不能判断好坏"]),
        "next_steps": source.get("next_steps", _default_next_steps(ticker, missing_data)),
        "tools_used": source.get("tools_used", ANALYSIS_TOOLS),
        "evidence_grade": source.get("evidence_grade", "B" if data_available else "C"),
        "source_note": source.get("source_note", "等待 financial-research-agent / SEC / OpenBB 输入原始财报数据"),
        "missing_data": missing_data,
        "reported_metrics": source.get("reported_metrics", []),
        "call_highlights": source.get("call_highlights", []),
        "management_tone": source.get("management_tone", ""),
        "moomoo_news": source.get("moomoo_news", []),
        "source_documents": source.get("source_documents", []),
        "analysis_summary": source.get("analysis_summary", ""),
    }


def _metric_signals(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "revenue": _score_growth(metrics.get("revenue_growth"), good=0.10, bad=0.00),
        "eps": _score_growth(metrics.get("eps_growth"), good=0.10, bad=-0.05),
        "margin": _score_growth(metrics.get("gross_margin_change_bps"), good=100, bad=-100),
        "fcf": _score_growth(metrics.get("free_cash_flow_margin"), good=0.15, bad=0.05),
        "guidance": _score_text(metrics.get("guidance_signal")),
        "tone": _score_text(metrics.get("management_tone")),
        "risk": -_score_text(metrics.get("risk_language")),
    }


def _has_source_evidence(source: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Return true only when the review has actual evidence, not just an empty shell."""

    if source.get("data_available") is True:
        return True
    if source.get("verdict") or source.get("investment_value") or source.get("action"):
        return True
    if any(value not in (None, "") for value in metrics.values()):
        return True
    for key in ("reported_metrics", "call_highlights", "source_documents"):
        value = source.get(key)
        if isinstance(value, list) and value:
            return True
    return bool(source.get("analysis_summary") or source.get("management_tone") or source.get("source_note"))


def _quality_score(signals: dict[str, float], source: dict[str, Any]) -> float:
    weights = {"revenue": 0.18, "eps": 0.16, "margin": 0.16, "fcf": 0.18, "guidance": 0.16, "tone": 0.10, "risk": 0.06}
    base = 50.0 + sum(signals[key] * weights[key] * 50 for key in weights)
    evidence_grade = str(source.get("evidence_grade", "B")).upper()
    if evidence_grade == "C":
        base -= 8
    elif evidence_grade == "A":
        base += 4
    return round(max(0.0, min(100.0, base)), 1)


def _score_growth(value: Any, good: float, bad: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number >= good:
        return 1.0
    if number <= bad:
        return -1.0
    return (number - bad) / (good - bad) * 2 - 1


def _score_text(value: Any) -> float:
    normalized = str(value or "").lower()
    if normalized in {"beat", "positive", "raise", "raised", "improving", "strong", "good", "bullish", "利好", "上调", "改善"}:
        return 1.0
    if normalized in {"miss", "negative", "cut", "lowered", "weakening", "weak", "bad", "bearish", "利空", "下调", "恶化"}:
        return -1.0
    return 0.0


def _verdict(score: float) -> str:
    if score >= 72:
        return "好于预期"
    if score >= 58:
        return "中性偏好"
    if score >= 45:
        return "中性"
    return "低于预期"


def _thesis_impact(score: float) -> str:
    if score >= 72:
        return "强化"
    if score >= 50:
        return "无明显削弱"
    if score >= 40:
        return "削弱"
    return "可能破裂"


def _investment_value(score: float) -> str:
    if score >= 72:
        return "投资价值提高，可进入加仓候选但仍看价格和仓位"
    if score >= 58:
        return "投资价值维持，继续跟踪指引兑现"
    if score >= 45:
        return "投资价值中性，等待更多证据"
    return "投资价值下降，优先复核 thesis 和风险敞口"


def _action(score: float) -> str:
    if score >= 72:
        return "若价格接近支撑且组合风险允许，可小额分批加仓"
    if score >= 58:
        return "持有观察，等待电话会/10-Q验证"
    if score >= 45:
        return "不加仓，观察管理层指引是否兑现"
    return "降低风险或暂不买入"


def _missing_data(metrics: dict[str, Any]) -> list[str]:
    required = {
        "revenue_growth": "收入增速",
        "eps_growth": "EPS增速",
        "gross_margin_change_bps": "毛利率变化",
        "free_cash_flow_margin": "自由现金流率",
        "guidance_signal": "管理层指引",
        "management_tone": "管理层语气",
        "risk_language": "风险语言变化",
    }
    return [label for key, label in required.items() if metrics.get(key) in (None, "")]


def _default_key_points(signals: dict[str, float], missing_data: list[str]) -> list[str]:
    if missing_data:
        return ["财报已进入解析队列", "等待原始财报、10-Q/10-K或电话会纪要进入系统", "取得可核验数据后自动判断好于/低于预期和投资价值变化"]
    positives = [key for key, value in signals.items() if value > 0.4]
    negatives = [key for key, value in signals.items() if value < -0.4]
    return [f"正面项：{', '.join(positives) or '暂无'}", f"负面项：{', '.join(negatives) or '暂无'}"]


def _default_next_steps(ticker: str, missing_data: list[str]) -> list[str]:
    if missing_data:
        return [
            f"用 financial-research-agent 抓取 {ticker} 最新 SEC filing / earnings release",
            "用 earnings-review 检查收入、EPS、毛利率、FCF、指引和电话会语气",
            "将结论写入日报和网站财报解析列表",
        ]
    return ["复核电话会Q&A", "对照投资逻辑失效条件", "更新买入/卖出点位和仓位建议"]


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
