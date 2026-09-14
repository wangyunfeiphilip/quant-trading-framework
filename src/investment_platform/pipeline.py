"""Daily orchestration for the personal AI investment research platform."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from config_utils import load_config
from investment_platform.allocation import build_allocation_plan
from investment_platform.behavior import BehaviorAnalyzer
from investment_platform.discovery import classify_catalysts, rank_theme_candidates, summarize_smart_money
from investment_platform.earnings import build_earnings_reviews, enrich_earnings_reviews_with_news
from investment_platform.events import (
    attach_recommendation_history,
    build_earnings_calendar,
    fetch_moomoo_earnings_news,
    fetch_nasdaq_earnings_calendar,
    fetch_rss_news,
)
from investment_platform.factors import DynamicFactorWeightEngine, score_stocks
from investment_platform.ibkr_inputs import load_ibkr_account, load_ibkr_positions
from investment_platform.industry import build_industry_nodes, group_by_theme
from investment_platform.integrations import integration_status
from investment_platform.market_regime import MarketRegimeDetector
from investment_platform.moomoo_inputs import build_watchlists_from_moomoo, latest_close_by_ticker, load_moomoo_watchlist, load_ohlc_history
from investment_platform.position_sizing import PositionSizingEngine
from investment_platform.predictions import PredictionTracker, build_prediction
from investment_platform.reporting import render_daily_report
from investment_platform.research_stack import build_research_stack, build_search_playbooks
from investment_platform.schemas import dataclass_to_dict
from investment_platform.sentiment import SentimentAnalyzer
from investment_platform.stock_details import build_stock_details
from investment_platform.thesis import ThesisTracker, build_thesis
from investment_platform.trade_plan import build_trade_plans
from investment_platform.valuation import build_forward_2027_scenarios, build_scenarios, summarize_valuation


def run_daily_research(config_path: str | Path | dict[str, Any] = "investment_platform.yaml") -> dict[str, Any]:
    """Run one offline research cycle and write report artifacts."""

    config = _load_platform_config(config_path)
    as_of = _resolve_as_of(config.get("as_of"))
    output_dir = Path(str(config.get("output_dir", "results/investment_platform")))
    output_dir.mkdir(parents=True, exist_ok=True)

    moomoo_rows = load_moomoo_watchlist(config.get("moomoo_watchlist_files", []))
    ibkr_positions, account_summary = load_ibkr_account(config.get("ibkr_account_snapshot_file"))
    if not ibkr_positions:
        ibkr_positions = load_ibkr_positions(config.get("ibkr_positions_file"))
    regime = MarketRegimeDetector().detect(config["market_regime_inputs"])
    weights = DynamicFactorWeightEngine().weights_for(regime)
    scores = score_stocks(config["factor_inputs"], weights)
    score_by_ticker = {item.ticker: item.total_score for item in scores}

    theses = [build_thesis(item) for item in config.get("theses", [])]
    theses = ThesisTracker().evaluate(theses, config.get("thesis_metrics", {}))

    current_prices = {str(item["ticker"]): float(item["price"]) for item in config.get("current_prices", [])}
    for item in moomoo_rows:
        if item.get("last_price") is not None:
            current_prices[str(item["ticker"])] = float(item["last_price"])
    for item in ibkr_positions:
        if item.get("ibkr_price") is not None:
            current_prices[str(item["ticker"])] = float(item["ibkr_price"])
    all_tickers = sorted(
        {
            *(item.ticker for item in scores),
            *(str(item.get("ticker", "")) for item in ibkr_positions),
            *(str(item.get("ticker", "")) for item in config.get("portfolio", [])),
            *(str(item.get("ticker", "")) for item in moomoo_rows),
        }
    )
    ohlc_history = load_ohlc_history(config.get("ohlc_sources", []), all_tickers)
    current_prices.update(latest_close_by_ticker(ohlc_history))
    valuations = _build_valuations(config, moomoo_rows, current_prices, score_by_ticker)

    risk_by_ticker = {str(item["ticker"]): item for item in config.get("risk_inputs", [])}
    portfolio = ibkr_positions or config.get("portfolio", [])
    sizing_engine = PositionSizingEngine()
    sizing = []
    for position in portfolio:
        ticker = str(position["ticker"])
        risk = risk_by_ticker.get(ticker, {})
        sizing.append(
            sizing_engine.recommend(
                ticker=ticker,
                score=score_by_ticker.get(ticker, 50.0),
                current_allocation=float(position.get("allocation", 0.0)),
                volatility=float(risk.get("volatility", 0.25)),
                max_drawdown=float(risk.get("max_drawdown", -0.20)),
                sector_exposure=float(position.get("sector_exposure", 0.0)),
                average_correlation=float(risk.get("average_correlation", 0.50)),
            )
        )

    predictions = [build_prediction(item) for item in config.get("predictions", [])]
    prediction_tracker = PredictionTracker()
    outcomes = prediction_tracker.evaluate(predictions, as_of, config.get("price_history", {}))
    prediction_summary = prediction_tracker.summary(outcomes)

    sentiment = SentimentAnalyzer().classify(config.get("sentiment_inputs", []))
    behaviors = BehaviorAnalyzer().analyze(config.get("trades", []), portfolio)
    theme_candidates = rank_theme_candidates(config.get("theme_candidates", []))
    theme_candidates = attach_recommendation_history(theme_candidates, config.get("previous_recommendations", []), as_of)
    catalysts = classify_catalysts(config.get("catalysts", []))
    smart_money = summarize_smart_money(config.get("smart_money", []))
    watchlists = config.get("watchlists", [])
    moomoo_watchlists = build_watchlists_from_moomoo(moomoo_rows)
    if moomoo_watchlists:
        watchlists = [item for item in watchlists if item.get("source") != "moomoo_placeholder"] + moomoo_watchlists
    tracked_tickers = set(all_tickers)
    for watchlist in watchlists:
        tracked_tickers.update(str(ticker).upper() for ticker in watchlist.get("tickers", []) if ticker)
    auto_calendar_rows = _fetch_auto_earnings_calendar(config, tracked_tickers, as_of)
    earnings_calendar = build_earnings_calendar([*config.get("earnings_calendar", []), *auto_calendar_rows], as_of)
    moomoo_earnings_news = _fetch_moomoo_earnings_news(config, earnings_calendar)
    fed_news = fetch_rss_news(config.get("fed_news_feeds", []), config.get("fed_news", []), "Fed")
    international_news = fetch_rss_news(config.get("international_news_feeds", []), config.get("international_news", []), "International")
    data_sources = config.get("data_sources", [])
    earnings_reviews = build_earnings_reviews(
        earnings_calendar,
        config.get("earnings_review_inputs", []),
        tracked_tickers,
        as_of,
    )
    earnings_reviews = enrich_earnings_reviews_with_news(earnings_reviews, moomoo_earnings_news)
    earnings_reviews = _write_earnings_review_files(output_dir, as_of, earnings_reviews)
    industry_nodes = build_industry_nodes(config.get("industry_map", []))
    industry_map = group_by_theme(industry_nodes)
    integrations = integration_status()
    data_sources = _refresh_data_source_status(data_sources, integrations)
    research_stack = build_research_stack(integrations)
    search_playbooks = build_search_playbooks()
    portfolio_focus = config.get("portfolio_focus", {})
    allocation_plan = build_allocation_plan(portfolio, sizing, score_by_ticker, account_summary, config.get("portfolio_policy"))
    portfolio_daily_review = _build_portfolio_daily_review(portfolio, ohlc_history, account_summary)
    stock_details = build_stock_details(
        as_of=as_of,
        scores=scores,
        theses=theses,
        valuations=valuations,
        sizing=sizing,
        sentiment=sentiment,
        portfolio=portfolio,
        watchlists=watchlists,
        moomoo_rows=moomoo_rows,
        ohlc_history=ohlc_history,
        catalysts=catalysts,
        smart_money=smart_money,
    )
    trade_plans = build_trade_plans(stock_details, allocation_plan, theme_candidates)
    top_recommendations = _build_top_recommendations(trade_plans, allocation_plan)

    report = render_daily_report(
        as_of=as_of,
        regime=regime,
        weights=weights,
        scores=scores,
        theses=theses,
        valuations=valuations,
        sizing=sizing,
        predictions=outcomes,
        prediction_summary=prediction_summary,
        sentiment=sentiment,
        behaviors=behaviors,
        integrations=integrations,
        theme_candidates=theme_candidates,
        catalysts=catalysts,
        smart_money=smart_money,
        earnings_calendar=earnings_calendar,
        earnings_reviews=earnings_reviews,
        fed_news=fed_news,
        international_news=international_news,
        account_summary=account_summary,
        allocation_plan=allocation_plan,
        trade_plans=trade_plans,
        portfolio_focus=portfolio_focus,
        portfolio_daily_review=portfolio_daily_review,
        portfolio=portfolio,
        watchlists=watchlists,
        data_sources=data_sources,
    )

    report_path = output_dir / f"daily_report_{as_of.isoformat()}.md"
    snapshot_path = output_dir / f"run_snapshot_{as_of.isoformat()}.json"
    report_path.write_text(report, encoding="utf-8")
    snapshot = {
        "as_of": as_of.isoformat(),
        "market_regime": dataclass_to_dict(regime),
        "factor_weights": dataclass_to_dict(weights),
        "scores": dataclass_to_dict(scores),
        "theses": dataclass_to_dict(theses),
        "valuations": dataclass_to_dict(valuations),
        "sizing": dataclass_to_dict(sizing),
        "prediction_outcomes": dataclass_to_dict(outcomes),
        "prediction_summary": prediction_summary,
        "sentiment": dataclass_to_dict(sentiment),
        "behaviors": dataclass_to_dict(behaviors),
        "theme_candidates": theme_candidates,
        "catalysts": catalysts,
        "smart_money": smart_money,
        "earnings_calendar": earnings_calendar,
        "earnings_reviews": earnings_reviews,
        "fed_news": fed_news,
        "international_news": international_news,
        "account_summary": account_summary,
        "allocation_plan": allocation_plan,
        "portfolio_focus": portfolio_focus,
        "portfolio_daily_review": portfolio_daily_review,
        "trade_plans": trade_plans,
        "top_recommendations": top_recommendations,
        "portfolio": portfolio,
        "watchlists": watchlists,
        "moomoo_watchlist_rows": moomoo_rows,
        "stock_details": stock_details,
        "data_sources": data_sources,
        "research_stack": research_stack,
        "search_playbooks": search_playbooks,
        "industry_map": dataclass_to_dict(industry_map),
        "integrations": dataclass_to_dict(integrations),
        "report_path": str(report_path),
    }
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot


def _build_valuations(
    config: dict[str, Any],
    moomoo_rows: list[dict[str, Any]],
    current_prices: dict[str, float],
    score_by_ticker: dict[str, float],
) -> list[Any]:
    valuations = []
    seen: set[str] = set()
    assumptions_by_ticker = {
        str(item.get("ticker", "")).upper(): item
        for item in config.get("forward_2027_valuation_assumptions", [])
        if item.get("ticker")
    }
    for row in moomoo_rows:
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        scenarios = build_forward_2027_scenarios(
            ticker,
            price=current_prices.get(ticker),
            score=score_by_ticker.get(ticker),
            moomoo_valuation=row.get("valuation") if isinstance(row.get("valuation"), dict) else {},
            assumptions=assumptions_by_ticker.get(ticker),
        )
        if not scenarios:
            continue
        valuations.append(summarize_valuation(ticker, scenarios, current_prices.get(ticker)))
        seen.add(ticker)
    for item in config.get("valuations", []):
        ticker = str(item["ticker"]).upper()
        if ticker in seen:
            continue
        scenarios = build_scenarios(item["scenarios"])
        valuations.append(summarize_valuation(ticker, scenarios, current_prices.get(ticker)))
        seen.add(ticker)
    return valuations


def _build_top_recommendations(trade_plans: list[dict[str, Any]], allocation_plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_ticker = {str(row.get("ticker", "")): row for row in allocation_plan.get("rows", [])}
    priority = {
        "reduce_fundamental_risk": 0,
        "trim_valuation_risk": 1,
        "trim_speculative_strength": 2,
        "reduce_low_conviction": 3,
        "buy_pullback_or_breakout": 4,
        "staged_buy_near_support": 5,
        "hold_winner_no_forced_trim": 4,
        "hold_risk_limit_no_add": 6,
        "do_not_chase_wait_breakout": 7,
        "wait_for_pullback": 8,
        "wait_for_cash": 9,
        "watch_only": 10,
        "no_trade_until_data": 11,
    }
    output = []
    for plan in trade_plans:
        ticker = str(plan.get("ticker", ""))
        row = rows_by_ticker.get(ticker, {})
        delta = float(row.get("delta_value", 0.0) or 0.0)
        feasible = float(row.get("feasible_trade_value", 0.0) or 0.0)
        decision = str(plan.get("decision", ""))
        suggested_amount = _suggested_amount(decision, delta, feasible)
        output.append(
            {
                "ticker": ticker,
                "decision": decision,
                "source": plan.get("source"),
                "price": plan.get("price"),
                "problem_area": plan.get("problem_area"),
                "diagnosis": plan.get("diagnosis"),
                "target_allocation": row.get("target_allocation"),
                "current_allocation": row.get("current_allocation"),
                "suggested_amount": round(suggested_amount, 2),
                "buy_zone": plan.get("buy_zone"),
                "trim_zone": plan.get("trim_zone"),
                "stop_loss": plan.get("stop_loss"),
                "sell_zone": plan.get("sell_zone"),
                "reason": plan.get("reason"),
            }
        )
    return sorted(output, key=lambda item: (priority.get(str(item.get("decision")), 9), str(item.get("ticker", ""))))[:10]


def _suggested_amount(decision: str, delta: float, feasible: float) -> float:
    sell_decisions = {"reduce_fundamental_risk", "trim_valuation_risk", "trim_speculative_strength", "reduce_low_conviction"}
    buy_decisions = {"staged_buy_near_support", "buy_pullback_or_breakout"}
    if decision in sell_decisions:
        return min(delta, 0.0)
    if decision in buy_decisions:
        return max(feasible, delta, 0.0)
    return 0.0


def _fetch_auto_earnings_calendar(config: dict[str, Any], tracked_tickers: set[str], as_of: date) -> list[dict[str, Any]]:
    settings = config.get("earnings_calendar_auto_refresh", {})
    if settings is False or (isinstance(settings, dict) and settings.get("enabled") is False):
        return []
    if not isinstance(settings, dict):
        settings = {}
    return fetch_nasdaq_earnings_calendar(
        tracked_tickers,
        as_of,
        lookahead_days=int(settings.get("lookahead_days", 21) or 21),
        timeout=int(settings.get("timeout", 8) or 8),
    )


def _fetch_moomoo_earnings_news(config: dict[str, Any], earnings_calendar: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    settings = config.get("moomoo_earnings_news", {})
    if settings is False or (isinstance(settings, dict) and settings.get("enabled") is False):
        return {}
    if not isinstance(settings, dict):
        settings = {}
    return fetch_moomoo_earnings_news(
        earnings_calendar,
        size=int(settings.get("size", 6) or 6),
        timeout=int(settings.get("timeout", 8) or 8),
    )


def _write_earnings_review_files(
    output_dir: Path,
    as_of: date,
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    review_dir = output_dir / "earnings_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    enriched = []
    for review in reviews:
        ticker = str(review.get("ticker", "")).upper()
        if not ticker:
            continue
        path = review_dir / f"{ticker}_earnings_review_{as_of.isoformat()}.md"
        hydrated = _hydrate_from_existing_earnings_report(review, path)
        path.write_text(_render_earnings_review_markdown(hydrated, as_of), encoding="utf-8")
        enriched.append({**hydrated, "report_path": str(path)})
    return enriched


def _hydrate_from_existing_earnings_report(review: dict[str, Any], path: Path) -> dict[str, Any]:
    """Preserve a real manually/tool-written earnings note instead of overwriting it with a queue stub."""

    if review.get("status") != "queued" or not path.exists():
        return review
    text = path.read_text(encoding="utf-8", errors="ignore")
    if _looks_like_generated_queued_report(text):
        return review
    parsed = _parse_report_bullets(text)
    key_points = _parse_markdown_list_after_heading(text, "核心结论")
    red_flags = _parse_markdown_list_after_heading(text, "红旗")
    next_steps = _parse_markdown_list_after_heading(text, "下一步")
    return {
        **review,
        "status": "analyzed",
        "verdict": parsed.get("财报好坏") or review.get("verdict") or "已解析，待结构化评分",
        "quality_score": _parse_optional_float(parsed.get("质量分")) or review.get("quality_score"),
        "thesis_impact": parsed.get("投资逻辑影响") or review.get("thesis_impact") or "待结构化复核",
        "investment_value": parsed.get("投资价值") or review.get("investment_value") or "已有原文解析，等待结构化估值联动",
        "action": parsed.get("操作建议") or review.get("action") or "阅读原文解析后再执行",
        "evidence_grade": parsed.get("证据等级") or review.get("evidence_grade") or "B",
        "source_note": parsed.get("来源说明") or review.get("source_note") or "来自本地原文解析报告",
        "key_points": key_points or review.get("key_points", []),
        "red_flags": red_flags or review.get("red_flags", []),
        "next_steps": next_steps or review.get("next_steps", []),
    }


def _looks_like_generated_queued_report(text: str) -> bool:
    markers = [
        "解析状态：queued",
        "财报好坏：待原始数据",
        "财报好坏：待原文解析",
        "等待 financial-research-agent / SEC / OpenBB 输入原始财报数据",
        "没有原始财报/SEC/电话会证据时，系统不会编造好坏结论",
    ]
    return sum(marker in text for marker in markers) >= 2


def _parse_report_bullets(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("- ") or "：" not in line:
            continue
        key, value = line[2:].split("：", 1)
        values[key.strip()] = value.strip()
    return values


def _parse_markdown_list_after_heading(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    values: list[str] = []
    in_section = False
    for line in lines:
        if line.startswith("## "):
            if in_section:
                break
            in_section = line.strip() == f"## {heading}"
            continue
        if in_section and line.startswith("- "):
            values.append(line[2:].strip())
    return values


def _parse_optional_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_earnings_review_markdown(review: dict[str, Any], as_of: date) -> str:
    lines = [
        f"# {review.get('ticker', '')} 财报解析 - {as_of.isoformat()}",
        "",
        "> 仅用于投研辅助。没有原始财报/SEC/电话会证据时，系统不会编造好坏结论。",
        "",
        f"- 公司：{review.get('company', '')}",
        f"- 财报日期：{review.get('date', '')} {review.get('time', '')}",
        f"- 解析状态：{review.get('status', '')}",
        f"- 财报好坏：{review.get('verdict', '')}",
        f"- 质量分：{review.get('quality_score', 'NA')}",
        f"- 投资逻辑影响：{review.get('thesis_impact', '')}",
        f"- 投资价值：{review.get('investment_value', '')}",
        f"- 操作建议：{review.get('action', '')}",
        f"- 证据等级：{review.get('evidence_grade', '')}",
        f"- 来源说明：{review.get('source_note', '')}",
        "",
        "## 重点观察",
        f"- {review.get('focus', '') or '暂无'}",
        "",
        "## 核心结论",
    ]
    for item in review.get("key_points", []) or ["暂无"]:
        lines.append(f"- {item}")
    if review.get("reported_metrics"):
        lines.extend(["", "## 已披露/核验指标"])
        for item in review.get("reported_metrics", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('metric', '')}: {item.get('actual', '')}；"
                f"预期 {item.get('estimate', '')}；变化 {item.get('change', '')}；{item.get('note', '')}"
            )
    if review.get("call_highlights"):
        lines.extend(["", "## 电话会 / 管理层要点"])
        for item in review.get("call_highlights", []):
            lines.append(f"- {item}")
    lines.extend(["", "## 红旗"])
    for item in review.get("red_flags", []) or ["暂无"]:
        lines.append(f"- {item}")
    if review.get("source_documents"):
        lines.extend(["", "## 原始证据 / 官网链接"])
        for item in review.get("source_documents", []):
            lines.append(f"- {item}")
    lines.extend(["", "## 下一步"])
    for item in review.get("next_steps", []) or ["暂无"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 使用工具"])
    for item in review.get("tools_used", []) or ["暂无"]:
        lines.append(f"- {item}")
    if review.get("moomoo_news"):
        lines.extend(["", "## Moomoo 财报/电话会相关报道"])
        for item in review.get("moomoo_news", [])[:8]:
            lines.append(f"- {item.get('title', '')} ({item.get('publish_time', '')}) {item.get('url', '')}")
    return "\n".join(lines) + "\n"


def _build_portfolio_daily_review(
    portfolio: list[dict[str, Any]],
    ohlc_history: dict[str, dict[str, Any]],
    account_summary: dict[str, Any],
) -> dict[str, Any]:
    net_liquidation = float(account_summary.get("net_liquidation", 0.0) or 0.0)
    account_as_of = _date_or_none(account_summary.get("as_of"))
    rows = []
    total_daily_pnl = 0.0
    for item in portfolio:
        ticker = str(item.get("ticker", ""))
        qty = _float_or_zero(item.get("quantity"))
        avg_cost = _float_or_none(item.get("avg_cost"))
        ibkr_price = _float_or_none(item.get("ibkr_price"))
        market_value = _float_or_none(item.get("market_value"))
        unrealized_pnl = _float_or_none(item.get("unrealized_pnl"))
        history = ohlc_history.get(ticker, {})
        candles = history.get("candles", []) if isinstance(history, dict) else []
        last = candles[-1] if candles else {}
        prev = candles[-2] if len(candles) >= 2 else {}
        latest_bar_close = _float_or_none(last.get("close"))
        last_close = ibkr_price or latest_bar_close
        prev_close = _float_or_none(prev.get("close"))
        daily_return = None if not latest_bar_close or not prev_close else latest_bar_close / prev_close - 1.0
        calculated_daily_pnl = None if daily_return is None else qty * (latest_bar_close - prev_close)
        latest_bar_date = _date_or_none(last.get("time"))
        ibkr_daily_pnl = _float_or_none(item.get("daily_pnl"))
        ibkr_daily_is_fresh = bool(
            account_as_of
            and ibkr_daily_pnl is not None
            and (latest_bar_date is None or account_as_of >= latest_bar_date)
        )
        daily_pnl = ibkr_daily_pnl if ibkr_daily_is_fresh else None
        if daily_pnl is None:
            daily_pnl = calculated_daily_pnl
        if ibkr_daily_is_fresh and (latest_bar_date is None or latest_bar_date < account_as_of):
            daily_return = _estimate_return_from_daily_pnl(daily_pnl, market_value)
        if daily_pnl is not None:
            total_daily_pnl += daily_pnl
        cost_basis = None if avg_cost is None else avg_cost * qty
        unrealized_pnl_pct = None
        if unrealized_pnl is not None and cost_basis and cost_basis != 0:
            unrealized_pnl_pct = unrealized_pnl / cost_basis
        elif avg_cost and last_close:
            unrealized_pnl_pct = last_close / avg_cost - 1.0
        rows.append(
            {
                "ticker": ticker,
                "quantity": qty,
                "avg_cost": avg_cost,
                "current_price": last_close,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "previous_close": prev_close,
                "latest_close": last_close,
                "daily_return": daily_return,
                "daily_pnl": daily_pnl,
                "daily_pnl_source": "IBKR" if ibkr_daily_is_fresh and ibkr_daily_pnl is not None else "日K估算",
                "latest_date": account_as_of.isoformat() if ibkr_daily_is_fresh and (latest_bar_date is None or latest_bar_date < account_as_of) else last.get("time"),
                "previous_date": prev.get("time"),
                "allocation": item.get("allocation"),
            }
        )
    rows.sort(key=lambda row: abs(float(row.get("daily_pnl") or 0.0)), reverse=True)
    return {
        "total_daily_pnl": round(total_daily_pnl, 2),
        "total_daily_return": None if not net_liquidation else round(total_daily_pnl / net_liquidation, 4),
        "rows": rows,
    }


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: Any) -> float:
    return _float_or_none(value) or 0.0


def _estimate_return_from_daily_pnl(daily_pnl: float | None, market_value: float | None) -> float | None:
    if daily_pnl is None or market_value is None:
        return None
    prior_value = market_value - daily_pnl
    if prior_value == 0:
        return None
    return daily_pnl / prior_value


def _date_or_none(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _load_platform_config(config_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path, dict):
        return config_path
    path = Path(config_path)
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("platform JSON config must contain an object")
        return loaded
    return load_config(path)


def _resolve_as_of(value: Any) -> date:
    """Resolve dashboard date labels without freezing the site to a stale config date."""

    if value in (None, "", "today", "auto", "current"):
        return date.today()
    return date.fromisoformat(str(value))


def _refresh_data_source_status(data_sources: list[dict[str, Any]], integrations: list[Any]) -> list[dict[str, Any]]:
    """Sync data-source readiness labels with installed optional engines."""

    installed = {str(item.name).lower(): bool(item.available) for item in integrations}
    aliases = {
        "openbb": "openbb",
        "qlib": "qlib",
        "finrl": "finrl",
        "edgartools": "edgartools",
        "financial-research-agent": "financial-research-agent",
        "ai-berkshire": "ai-berkshire",
    }
    installed_notes = {
        "openbb": "依赖已安装；配置数据源/API Key 后可刷新财务、新闻、宏观和期权数据",
        "qlib": "依赖已安装；接入Qlib数据目录后可运行机构式因子研究和回测",
        "finrl": "依赖已安装；仅作为强化学习研究沙盒，不直接触发真实交易建议",
        "edgartools": "依赖已安装；配置SEC身份信息后可抓取filings、13F和Form 4",
        "financial-research-agent": "依赖已安装；可接入SEC RAG、FinBERT和财报语气变化研究",
        "ai-berkshire": "技能/依赖已安装；可接入价值投资检查清单和反方研究流程",
    }
    refreshed: list[dict[str, Any]] = []
    for item in data_sources:
        copy = dict(item)
        key = aliases.get(str(copy.get("name", "")).lower())
        if key and installed.get(key) and copy.get("status") == "adapter_ready_not_installed":
            copy["status"] = "installed_adapter_ready"
            copy["notes"] = installed_notes[key]
        refreshed.append(copy)
    return refreshed
