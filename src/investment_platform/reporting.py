"""Markdown daily report generation."""

from __future__ import annotations

from datetime import date

from investment_platform.schemas import (
    BehaviorPattern,
    FactorWeights,
    InvestmentThesis,
    MarketRegime,
    PositionSizingRecommendation,
    PredictionOutcome,
    SentimentRead,
    StockScore,
    ValuationSummary,
)
from investment_platform.integrations import IntegrationStatus


def render_daily_report(
    as_of: date,
    regime: MarketRegime,
    weights: FactorWeights,
    scores: list[StockScore],
    theses: list[InvestmentThesis],
    valuations: list[ValuationSummary],
    sizing: list[PositionSizingRecommendation],
    predictions: list[PredictionOutcome],
    prediction_summary: dict[str, float],
    sentiment: list[SentimentRead],
    behaviors: list[BehaviorPattern],
    integrations: list[IntegrationStatus] | None = None,
    theme_candidates: list[dict[str, object]] | None = None,
    catalysts: list[dict[str, object]] | None = None,
    smart_money: list[dict[str, object]] | None = None,
    earnings_calendar: list[dict[str, object]] | None = None,
    earnings_reviews: list[dict[str, object]] | None = None,
    fed_news: list[dict[str, object]] | None = None,
    international_news: list[dict[str, object]] | None = None,
    account_summary: dict[str, object] | None = None,
    allocation_plan: dict[str, object] | None = None,
    trade_plans: list[dict[str, object]] | None = None,
    portfolio_focus: dict[str, object] | None = None,
    portfolio_daily_review: dict[str, object] | None = None,
    portfolio: list[dict[str, object]] | None = None,
    watchlists: list[dict[str, object]] | None = None,
    data_sources: list[dict[str, object]] | None = None,
) -> str:
    lines = [
        f"# 个人 AI 投研日报 - {as_of.isoformat()}",
        "",
        "> 仅用于投研辅助。系统不会自动下单、改单或撤单。",
        "",
        "## 市场环境",
        f"- 当前状态：**{_regime_cn(regime.name)}**",
        f"- 置信度：**{regime.confidence:.0%}**",
        f"- 风险偏好分数：**{regime.risk_on_score:.0%}**",
        f"- 支持因素：{', '.join(_driver_cn(item) for item in regime.drivers) if regime.drivers else '暂无强支持因素'}",
        f"- 风险提示：{', '.join(_driver_cn(item) for item in regime.cautions) if regime.cautions else '暂无重大警报'}",
        "",
        "## 动态因子权重",
        f"- 成长：{weights.growth:.0%}",
        f"- 质量：{weights.quality:.0%}",
        f"- 动量：{weights.momentum:.0%}",
        f"- 估值：{weights.value:.0%}",
        f"- 风险：{weights.risk:.0%}",
        f"- 原因：{_reason_cn(weights.reason)}",
        "",
        "## 股票综合排名",
        "| 排名 | 股票 | 分数 | 解释 |",
        "|---:|---|---:|---|",
    ]
    for rank, score in enumerate(scores[:10], start=1):
        lines.append(f"| {rank} | {score.ticker} | {score.total_score:.1f} | {'; '.join(_score_note_cn(item) for item in score.explanation)} |")

    if account_summary:
        lines.extend(
            [
                "",
                "## IBKR账户摘要",
                f"- 净值：${float(account_summary.get('net_liquidation', 0.0) or 0.0):,.2f}",
                f"- 现金/购买力：${float(account_summary.get('available_funds', 0.0) or 0.0):,.2f}",
                f"- 持仓市值：${float(account_summary.get('gross_position_value', 0.0) or 0.0):,.2f}",
            ]
        )

    if portfolio:
        lines.extend(["", "## 我的持仓", "| 股票 | 仓位 | 数量 | 成本 | 现价 | 市值 | 浮盈亏 | 盈亏率 | 昨日盈亏 | 行业 | 来源 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"])
        for item in portfolio:
            pnl = _float_cell(item.get("unrealized_pnl"))
            qty = _float_cell(item.get("quantity"))
            avg = _float_cell(item.get("avg_cost"))
            price = _float_cell(item.get("ibkr_price"))
            pnl_pct = "暂无"
            if pnl is not None and qty and avg:
                pnl_pct = f"{pnl / (qty * avg):.1%}"
            elif price is not None and avg:
                pnl_pct = f"{price / avg - 1.0:.1%}"
            lines.append(
                f"| {item.get('ticker', '')} | {float(item.get('allocation', 0.0)):.1%} | {item.get('quantity', '')} | "
                f"{_money_cell(item.get('avg_cost'))} | {_money_cell(item.get('ibkr_price'))} | {_money_cell(item.get('market_value'))} | "
                f"{_money_cell(item.get('unrealized_pnl'))} | {pnl_pct} | {_money_cell(item.get('daily_pnl'))} | "
                f"{item.get('sector', '')} | {_source_cn(str(item.get('source', 'manual/sample')))} |"
            )

    if portfolio_daily_review:
        lines.extend(
            [
                "",
                "## 昨日持仓涨跌回顾",
                f"- 合计影响：{_money_cell(portfolio_daily_review.get('total_daily_pnl'))}",
                f"- 占账户净值：{float(portfolio_daily_review.get('total_daily_return', 0.0) or 0.0):.1%}",
                "",
                "| 股票 | 前收 | 最新收盘 | 涨跌幅 | 昨日盈亏 | 成本 | 现价 | 总浮盈亏 | 盈亏率 | 来源 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in list(portfolio_daily_review.get("rows", []))[:30]:
            lines.append(
                f"| {item.get('ticker', '')} | {_money_cell(item.get('previous_close'))} | {_money_cell(item.get('latest_close'))} | "
                f"{float(item.get('daily_return', 0.0) or 0.0):.1%} | {_money_cell(item.get('daily_pnl'))} | "
                f"{_money_cell(item.get('avg_cost'))} | {_money_cell(item.get('current_price'))} | {_money_cell(item.get('unrealized_pnl'))} | "
                f"{float(item.get('unrealized_pnl_pct', 0.0) or 0.0):.1%} | {item.get('daily_pnl_source', '')} |"
            )

    if watchlists:
        lines.extend(["", "## 自选股分组", "| 分组 | 股票 | 来源 |", "|---|---|---|"])
        for item in watchlists:
            tickers = ", ".join(str(ticker) for ticker in item.get("tickers", []))
            lines.append(f"| {item.get('name', '')} | {tickers} | {_source_cn(str(item.get('source', '')))} |")

    lines.extend(["", "## 投资逻辑追踪", "| 股票 | 状态 | 触发条件 |", "|---|---|---|"])
    for thesis in theses:
        triggered = "; ".join(thesis.triggered_conditions) if thesis.triggered_conditions else "无"
        lines.append(f"| {thesis.ticker} | {_thesis_status_cn(thesis.status)} | {triggered} |")

    lines.extend(["", "## 情景估值", "| 股票 | 概率加权合理价值 | 相对当前价格 |", "|---|---:|---:|"])
    for valuation in valuations:
        upside = "n/a" if valuation.upside_to_price is None else f"{valuation.upside_to_price:.1%}"
        lines.append(f"| {valuation.ticker} | {valuation.weighted_fair_value:.2f} | {upside} |")

    if trade_plans:
        lines.extend(
            [
                "",
                "## 买入/卖出点位计划",
                "| 股票 | 类型 | 现价 | 判断 | 支撑买入区 | 突破买点 | 止损/失效 | 减仓区 | 目标卖出区 | 原因 |",
                "|---|---|---:|---|---|---|---|---|---|---|",
            ]
        )
        for item in trade_plans[:30]:
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('source', '')} | {_money_cell(item.get('price'))} | "
                f"{_trade_decision_cn(str(item.get('decision', '')))} | {item.get('buy_zone', '')} | "
                f"{item.get('breakout_buy', '')} | {item.get('stop_loss', '')} | {item.get('trim_zone', '')} | "
                f"{item.get('sell_zone', '')} | {item.get('reason', '')} |"
            )

    if portfolio_focus:
        lines.extend(
            [
                "",
                "## 精简组合方案",
                f"- 目标持仓数量：{portfolio_focus.get('target_position_count', '')}",
                f"- 原则：{portfolio_focus.get('principle', '')}",
                f"- 现金规则：{portfolio_focus.get('cash_policy', '')}",
            ]
        )
        for tier in portfolio_focus.get("tiers", []):
            if not isinstance(tier, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### {tier.get('name', '')}",
                    f"- 目标：{tier.get('objective', '')}",
                    "| 股票 | 角色 | 目标仓位 | 结论 | 原因 |",
                    "|---|---|---:|---|---|",
                ]
            )
            for item in tier.get("tickers", []):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {item.get('ticker', '')} | {item.get('role', '')} | "
                    f"{float(item.get('target', 0.0) or 0.0):.1%} | {item.get('decision', '')} | {item.get('reason', '')} |"
                )

    lines.extend(["", "## 单票风险约束诊断（不作为买卖信号）", "| 股票 | 当前仓位 | 风控上限 | 约束说明 |", "|---|---:|---:|---|"])
    for item in sizing:
        lines.append(
            f"| {item.ticker} | {item.current_allocation:.1%} | {item.max_allocation:.1%} | "
            f"{'; '.join(_sizing_reason_cn(reason) for reason in item.reasons)} |"
        )

    if allocation_plan:
        lines.extend(
            [
                "",
                "## 中等风险板块配置方案",
                f"- 风险档位：{allocation_plan.get('risk_profile', '中等风险')}",
                f"- 现金/购买力：${float(allocation_plan.get('cash_value', 0.0)):,.2f}",
                f"- 最低现金目标金额：${float(allocation_plan.get('target_cash_value', 0.0)):,.2f}",
                f"- 现金缺口：${float(allocation_plan.get('cash_shortfall', 0.0)):,.2f}",
                f"- 目标账户日波动：{float(allocation_plan.get('target_daily_volatility_min', 0.02)):.1%}-{float(allocation_plan.get('target_daily_volatility_max', 0.03)):.1%}",
                f"- 估算目标组合日波动：{float(allocation_plan.get('estimated_daily_volatility', 0.0)):.1%}（{_risk_budget_status_cn(str(allocation_plan.get('risk_budget_status', '')))}）",
                f"- {allocation_plan.get('constraint_note', '')}",
                "",
                "| 顺序 | 动作 | 股票/现金 | 板块 | 金额 | 原因 |",
                "|---:|---|---|---|---:|---|",
            ]
        )
        for item in allocation_plan.get("action_plan", [])[:30]:
            target = "现金缓冲" if item.get("ticker") == "CASH" else item.get("ticker", "")
            lines.append(
                f"| {item.get('step', '')} | {_allocation_side_cn(str(item.get('side', '')))} | {target} | "
                f"{item.get('bucket', '')} | ${float(item.get('amount', 0.0)):,.0f} | {item.get('reason', '')} |"
            )
        lines.extend(
            [
                "",
                "| 板块 | 当前占比 | 目标占比 | 需要买/卖 |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in allocation_plan.get("bucket_rows", []):
            lines.append(
                f"| {item.get('bucket', '')} | {float(item.get('current_allocation', 0.0)):.1%} | "
                f"{float(item.get('target_allocation', 0.0)):.1%} | ${float(item.get('delta_value', 0.0)):,.0f} |"
            )
        lines.extend(["", "| 股票 | 板块 | 当前占比 | 目标占比 | 需要买/卖 | 本次最多执行 | 动作 |", "|---|---|---:|---:|---:|---:|---|"])
        for item in allocation_plan.get("rows", [])[:20]:
            executable = item.get("feasible_trade_value")
            executable_text = "-" if executable is None or float(executable or 0.0) <= 0 else f"${float(executable):,.0f}"
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('bucket', '')} | {float(item.get('current_allocation', 0.0)):.1%} | "
                f"{float(item.get('target_allocation', 0.0)):.1%} | ${float(item.get('delta_value', 0.0)):,.0f} | "
                f"{executable_text} | {_allocation_action_cn(str(item.get('action', '')))} |"
            )

    lines.extend(
        [
            "",
            "## AI 预测准确率复盘",
            f"- 已复盘预测数：{int(prediction_summary['evaluated'])}",
            f"- 准确率：{prediction_summary['accuracy']:.1%}",
            f"- 平均实际收益：{prediction_summary['average_return']:.1%}",
        ]
    )
    for outcome in predictions[:10]:
        result = "正确" if outcome.correct else "错误"
        lines.append(f"- {outcome.ticker} {outcome.horizon_days}天：{outcome.actual_return:.1%}，{result}")

    lines.extend(["", "## 市场情绪", "| 股票 | 上涨来源 | 投机风险 |", "|---|---|---|"])
    for item in sentiment:
        lines.append(f"| {item.ticker} | {_sentiment_cn(item.label)} | {'高' if item.speculation_risk == 'high' else '正常'} |")

    if theme_candidates:
        lines.extend(["", "## 主题潜力股雷达", "| 股票 | 主题 | 潜力分 | 优先级 | 卡点 | 证据 |", "|---|---|---:|---|---|---|"])
        for item in theme_candidates[:10]:
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('theme', '')} | {float(item.get('potential_score', 0.0)):.1f} | "
                f"{_priority_cn(str(item.get('priority', '')))} | {item.get('bottleneck', '')} | {item.get('evidence', '')} |"
            )

    if catalysts:
        lines.extend(["", "## 事件催化日历", "| 股票 | 事件 | 日期 | 紧急度 | 期望波动 | 要盯什么 |", "|---|---|---|---|---:|---|"])
        for item in catalysts[:10]:
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('event', '')} | {item.get('date', '')} | "
                f"{'高' if item.get('urgency') == 'high' else '观察'} | {float(item.get('expected_move', 0.0)):.1%} | {item.get('watch_item', '')} |"
            )

    if earnings_calendar:
        lines.extend(["", "## 财报日历与财报后自动解析", "| 股票 | 公司 | 日期 | 时间 | 状态 | 重点 |", "|---|---|---|---|---|---|"])
        for item in earnings_calendar[:20]:
            watch_items = item.get("watch_items", [])
            watch = "；".join(str(value) for value in watch_items) if isinstance(watch_items, list) else str(watch_items)
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('company', '')} | {item.get('date', '')} | "
                f"{item.get('time', '')} | {_earnings_status_cn(str(item.get('status', '')))} | {watch} |"
            )

    if earnings_reviews:
        lines.extend(
            [
                "",
                "## 新出财报解析结果",
                "| 股票 | 财报好坏 | 质量分 | 投资逻辑影响 | 投资价值 | 操作建议 | Moomoo报道 | 证据等级 |",
                "|---|---|---:|---|---|---|---:|---|",
            ]
        )
        for item in earnings_reviews[:20]:
            score = item.get("quality_score")
            score_text = "NA" if score in (None, "") else f"{float(score):.1f}"
            news_count = len(item.get("moomoo_news", []) or [])
            lines.append(
                f"| {item.get('ticker', '')} | {item.get('verdict', '')} | {score_text} | "
                f"{item.get('thesis_impact', '')} | {item.get('investment_value', '')} | "
                f"{item.get('action', '')} | {news_count} | {item.get('evidence_grade', '')} |"
            )

    news_rows = (fed_news or []) + (international_news or [])
    if news_rows:
        lines.extend(["", "## 美联储与国际新闻", "| 类别 | 时间 | 标题 | 影响 | 观察点 |", "|---|---|---|---|---|"])
        for item in news_rows[:20]:
            lines.append(
                f"| {item.get('category', '')} | {item.get('time', '')} | {item.get('title', '')} | "
                f"{item.get('impact', '')} | {item.get('watch', '')} |"
            )

    if smart_money:
        lines.extend(["", "## 聪明钱追踪", "| 股票 | 信号 | 13F变化 | 内部人净买入 | 备注 |", "|---|---|---:|---:|---|"])
        for item in smart_money:
            lines.append(
                f"| {item.get('ticker', '')} | {_smart_money_cn(str(item.get('smart_money_signal', '')))} | "
                f"{float(item.get('institutional_change', 0.0)):.1%} | {float(item.get('insider_net_buying', 0.0)):.1%} | {item.get('notes', '')} |"
            )

    lines.extend(["", "## 个人投资行为提醒"])
    if behaviors:
        for item in behaviors:
            lines.append(f"- **{_behavior_label_cn(item.label)}（{_severity_cn(item.severity)}）**：{_behavior_evidence_cn(item.evidence)} {_behavior_reminder_cn(item.reminder)}")
    else:
        lines.append("- 今天没有触发明显行为偏差。")

    if integrations is not None:
        lines.extend(["", "## 开源引擎状态", "| 引擎 | 用途 | 状态 |", "|---|---|---|"])
        for item in integrations:
            status = "已可用" if item.available else "未安装"
            lines.append(f"| {item.name} | {item.role} | {status} |")

    if data_sources:
        lines.extend(["", "## 数据源状态", "| 数据源 | 用途 | 状态 | 备注 |", "|---|---|---|---|"])
        for item in data_sources:
            lines.append(f"| {item.get('name', '')} | {item.get('role', '')} | {_status_cn(str(item.get('status', '')))} | {item.get('notes', '')} |")

    return "\n".join(lines) + "\n"


def _regime_cn(value: str) -> str:
    return {"AI Growth Environment": "AI成长环境", "Risk On": "风险偏好开启", "Risk Off": "风险规避", "Transitional": "切换期"}.get(value, value)


def _driver_cn(value: str) -> str:
    mapping = {
        "VIX is calm": "VIX 较低，波动压力不高",
        "Nasdaq is above its long-term trend": "纳指位于长期趋势上方",
        "Market breadth is supportive": "市场宽度较好",
        "Credit stress is contained": "信用压力可控",
        "Rates are not pressuring duration assets": "利率暂未明显压制成长股",
    }
    return mapping.get(value, value)


def _reason_cn(value: str) -> str:
    return value.replace("Growth and momentum lead in an AI-led risk-on tape", "AI 主线和风险偏好较强时，成长与动量权重提高。")


def _score_note_cn(value: str) -> str:
    return value.replace("Strong", "强项").replace("Weak", "弱项").replace("growth", "成长").replace("quality", "质量").replace("momentum", "动量").replace("value", "估值").replace("risk", "风险")


def _source_cn(value: str) -> str:
    return {"moomoo_placeholder": "moomoo 占位，待真实接入", "manual_research": "手动研究池"}.get(value, value)


def _thesis_status_cn(value: str) -> str:
    return {"active": "有效", "weakened": "走弱"}.get(value, value)


def _action_cn(value: str) -> str:
    return {"trim_or_hold": "减仓或持有观察", "hold_or_watch": "持有或观察", "eligible_to_add": "风险预算内可研究加仓"}.get(value, value)


def _allocation_action_cn(value: str) -> str:
    return {
        "trim_to_target": "降到目标/不加仓",
        "add_if_cash_available": "有条件分批加仓",
        "wait_for_cash_or_trim": "现金不足，等待",
        "hold_near_target": "接近目标，持有",
    }.get(value, value)


def _trade_decision_cn(value: str) -> str:
    return {
        "reduce_fundamental_risk": "基本面走弱减仓",
        "trim_valuation_risk": "估值风险减仓",
        "trim_speculative_strength": "消息投机减仓",
        "reduce_low_conviction": "低确定性降权",
        "hold_winner_no_forced_trim": "强票持有/不强制卖",
        "hold_risk_limit_no_add": "仓位约束/不加",
        "buy_pullback_or_breakout": "强逻辑等买点",
        "wait_for_cash": "等现金",
        "staged_buy_near_support": "支撑区分批",
        "do_not_chase_wait_breakout": "不追高/等突破",
        "watch_only": "观察",
        "wait_for_pullback": "等回调",
        "no_trade_until_data": "数据不足不交易",
    }.get(value, value)


def _money_cell(value: object) -> str:
    try:
        if value in (None, "", "-"):
            return "暂无"
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "暂无"


def _float_cell(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _allocation_side_cn(value: str) -> str:
    return {
        "sell_or_stop_adding": "先卖出/停止加仓",
        "reserve_cash": "保留现金",
        "conditional_buy": "有条件买入",
    }.get(value, value)


def _risk_budget_status_cn(value: str) -> str:
    return {
        "within_target": "在目标内",
        "above_target": "高于目标",
        "below_target": "低于目标",
    }.get(value, value)


def _earnings_status_cn(value: str) -> str:
    return {
        "reported_needs_analysis": "已出，需要解析",
        "today": "今天",
        "upcoming": "即将公布",
        "date_unknown": "日期待确认",
    }.get(value, value)


def _sizing_reason_cn(value: str) -> str:
    mapping = {
        "Moderate volatility limits allocation": "波动率较高，限制仓位",
        "Large historical drawdown requires tighter sizing": "历史回撤较大，需要更严格仓位",
        "Sector exposure is already high": "行业暴露已经偏高",
        "High correlation reduces diversification benefit": "相关性较高，分散效果有限",
        "Current allocation is above recommended maximum": "当前仓位高于建议上限",
        "Allocation is within the current risk budget": "仓位在当前风险预算内",
        "High score leaves room under the risk cap": "评分较高，仍在风险上限以内",
    }
    return mapping.get(value, value)


def _sentiment_cn(value: str) -> str:
    return {"Fundamental driven": "基本面驱动", "Speculation driven": "投机情绪驱动", "Mixed": "混合驱动"}.get(value, value)


def _priority_cn(value: str) -> str:
    return {"top_research": "最高优先级", "high_research": "高优先级", "watch": "观察", "early_or_low_priority": "早期线索"}.get(value, value)


def _smart_money_cn(value: str) -> str:
    return {"accumulation": "机构/内部人偏增持", "distribution": "资金流出", "insider_selling_watch": "内部人卖出需观察", "neutral": "中性"}.get(value, value)


def _behavior_label_cn(value: str) -> str:
    return value.replace("Chasing strength after large one-day moves", "大涨后追入倾向")


def _behavior_evidence_cn(value: str) -> str:
    return value.replace("2 buys occurred after prior-day moves above 8%; success rate 50%.", "有 2 次买入发生在前一日涨幅超过 8% 之后，成功率约 50%。")


def _behavior_reminder_cn(value: str) -> str:
    return value.replace("Slow down after large single-day moves and require thesis confirmation before adding.", "单日大涨后放慢节奏，先确认投资逻辑和证据，再考虑加仓。")


def _severity_cn(value: str) -> str:
    return {"medium": "中", "high": "高", "low": "低"}.get(value, value)


def _status_cn(value: str) -> str:
    return {
        "pending_connection": "待接入",
        "adapter_ready_not_installed": "适配层已好，依赖未安装",
        "installed_adapter_ready": "依赖已安装，适配层可用",
        "local_snapshot": "本地快照可用",
    }.get(value, value)
