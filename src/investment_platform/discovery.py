"""Idea discovery, event catalyst, and smart-money research modules."""

from __future__ import annotations


def rank_theme_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank theme candidates using a research-stack-aware bottleneck score."""

    ranked: list[dict[str, object]] = []
    for row in rows:
        factors = row.get("factors", {})
        penalties = row.get("penalties", {})
        if not isinstance(factors, dict) or not isinstance(penalties, dict):
            continue
        gross = (
            float(factors.get("demand_inflection", 0)) * 15
            + float(factors.get("chokepoint_severity", 0)) * 15
            + float(factors.get("supplier_concentration", 0)) * 12
            + float(factors.get("expansion_difficulty", 0)) * 12
            + float(factors.get("evidence_quality", 0)) * 15
            + float(factors.get("valuation_disconnect", 0)) * 11
            + float(factors.get("catalyst_timing", 0)) * 10
            + float(factors.get("architecture_coupling", 0)) * 10
        ) / 5
        search_bonus = _search_bonus(row)
        penalty = sum(float(value) for value in penalties.values()) * 2
        score = max(0.0, min(100.0, gross + search_bonus - penalty))
        ranked.append(
            {
                **row,
                "potential_score": round(score, 1),
                "priority": _priority(score),
                "search_lanes": _search_lanes(row),
                "evidence_gates": _evidence_gates(row),
                "reject_rules": _reject_rules(row),
            }
        )
    return sorted(ranked, key=lambda item: item["potential_score"], reverse=True)


def classify_catalysts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize catalyst events and assign urgency labels."""

    result = []
    for row in rows:
        probability = float(row.get("probability", 0.5))
        payoff = float(row.get("payoff", 0.0))
        downside = abs(float(row.get("downside", 0.0)))
        expected_move = probability * payoff - (1 - probability) * downside
        urgency = "high" if int(row.get("days_until", 999)) <= 14 else "watch"
        result.append({**row, "expected_move": round(expected_move, 4), "urgency": urgency})
    return sorted(result, key=lambda item: (item["urgency"] != "high", -item["expected_move"]))


def summarize_smart_money(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create smart-money status records from ownership and insider data."""

    result = []
    for row in rows:
        insider = float(row.get("insider_net_buying", 0.0))
        institution = float(row.get("institutional_change", 0.0))
        signal = "neutral"
        if insider > 0 and institution > 0:
            signal = "accumulation"
        elif insider < 0 and institution < 0:
            signal = "distribution"
        elif insider < 0:
            signal = "insider_selling_watch"
        result.append({**row, "smart_money_signal": signal})
    return result


def _priority(score: float) -> str:
    if score >= 85:
        return "top_research"
    if score >= 70:
        return "high_research"
    if score >= 55:
        return "watch"
    return "early_or_low_priority"


def _search_bonus(row: dict[str, object]) -> float:
    """Reward candidates with catalysts that can be verified by multiple routes."""

    theme = f"{row.get('theme', '')} {row.get('bottleneck', '')} {row.get('evidence', '')}".lower()
    bonus = 0.0
    if any(token in theme for token in ["财报", "订单", "guidance", "rpo", "毛利率"]):
        bonus += 2.0
    if any(token in theme for token in ["癌症", "疫苗", "临床", "fda", "监管", "管线"]):
        bonus += 3.0
    if any(token in theme for token in ["ai", "hbm", "gpu", "asic", "数据中心", "电力", "散热"]):
        bonus += 2.5
    if row.get("next_check"):
        bonus += 1.0
    return min(6.0, bonus)


def _search_lanes(row: dict[str, object]) -> list[str]:
    theme = f"{row.get('theme', '')} {row.get('bottleneck', '')} {row.get('evidence', '')}".lower()
    lanes = ["OpenBB: 价格/财务/新闻/宏观刷新", "Qlib: 因子排名与回测验证"]
    if any(token in theme for token in ["财报", "guidance", "订单", "sec", "risk"]):
        lanes.append("financial-research-agent: SEC/财报语气与风险变化")
    if any(token in theme for token in ["癌症", "疫苗", "临床", "fda", "管线"]):
        lanes.append("事件驱动: 临床/FDA/会议窗口核验")
    if any(token in theme for token in ["现金流", "护城河", "roic", "价值", "利润率"]):
        lanes.append("ai-berkshire: 长期质量与安全边际检查")
    if any(token in theme for token in ["仓位", "组合", "再平衡"]):
        lanes.append("FinRL: 研究沙盒，不直接进入真实交易")
    return lanes


def _evidence_gates(row: dict[str, object]) -> list[str]:
    theme = f"{row.get('theme', '')} {row.get('bottleneck', '')} {row.get('next_check', '')}".lower()
    gates = ["必须有可验证价格/成交量数据", "必须有新闻/公告/财报/行业数据之一支持"]
    if any(token in theme for token in ["癌症", "疫苗", "临床", "fda"]):
        gates.extend(["必须核验临床阶段、样本量、安全性和监管时间表", "现金 runway 和融资稀释风险必须单独列出"])
    if any(token in theme for token in ["ai", "hbm", "gpu", "asic", "数据中心"]):
        gates.extend(["必须核验订单/收入/毛利率是否真实落地", "不能只因为 AI 主题热就入选"])
    return gates[:5]


def _reject_rules(row: dict[str, object]) -> list[str]:
    theme = f"{row.get('theme', '')} {row.get('bottleneck', '')}".lower()
    rules = ["无可验证催化不追", "流动性差或数据缺失不买", "估值空间和止损比不合理不买"]
    if any(token in theme for token in ["癌症", "疫苗", "临床", "fda"]):
        rules.append("临床数据质量不足或融资压力过大则淘汰")
    if any(token in theme for token in ["ai", "hbm", "gpu", "asic"]):
        rules.append("AI 叙事强但财务兑现弱则降级")
    return rules[:5]
