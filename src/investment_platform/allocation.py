"""Cash-aware target allocation planning."""

from __future__ import annotations

from typing import Any


def build_allocation_plan(
    portfolio: list[dict[str, Any]],
    sizing: list[Any],
    scores_by_ticker: dict[str, float],
    account_summary: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a medium-risk, bucket-aware allocation plan."""

    net_liquidation = float(account_summary.get("net_liquidation", 0.0) or 0.0)
    cash_value = float(account_summary.get("available_funds", account_summary.get("total_cash_value", 0.0)) or 0.0)
    cash_weight = cash_value / net_liquidation if net_liquidation else 0.0
    policy = policy or _default_policy()
    minimum_cash_weight = float(policy.get("minimum_cash_weight", 0.08))
    target_daily_volatility = policy.get("target_daily_volatility", {"min": 0.02, "max": 0.03})
    ticker_targets = _ticker_targets(policy)
    bucket_targets = _bucket_targets(policy)
    bucket_by_ticker = _bucket_by_ticker(policy)
    daily_vol_by_ticker = _daily_vol_by_ticker(policy)
    cap_by_ticker = {item.ticker: float(item.max_allocation) for item in sizing}
    sizing_by_ticker = {item.ticker: item for item in sizing}
    reserve_tickers = {str(item).upper() for item in policy.get("cash_reserve_tickers", ["SCHD", "SGOV", "BIL", "SHV"])}

    rows = []
    total_buy_need = 0.0
    total_trim_value = 0.0
    reserve = net_liquidation * minimum_cash_weight
    cash_reserve_value = 0.0
    spendable_cash = max(0.0, cash_value - reserve)
    current_bucket_weights: dict[str, float] = {"现金": cash_weight}

    for row in sorted(portfolio, key=lambda item: str(item.get("ticker", ""))):
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue
        current = float(row.get("allocation", 0.0) or 0.0)
        score = float(scores_by_ticker.get(ticker, 55.0))
        if ticker.upper() in reserve_tickers:
            bucket = "现金/防守储备"
            cash_reserve_value += current * net_liquidation
            current_bucket_weights[bucket] = current_bucket_weights.get(bucket, 0.0) + current
            rows.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "score": round(score, 1),
                    "current_allocation": round(current, 4),
                    "target_allocation": round(current, 4),
                    "max_allocation": round(max(current, cap_by_ticker.get(ticker, current)), 4),
                    "delta_value": 0.0,
                    "action": "cash_reserve_hold",
                    "note": "按现金/防守储备处理，不参与成长股风险预算；需要买股时再决定是否释放",
                    "feasible_trade_value": 0.0,
                }
            )
            continue
        bucket = bucket_by_ticker.get(ticker, _fallback_bucket(row))
        base_target = ticker_targets.get(ticker, _fallback_target(bucket, current, bucket_targets))
        cap = cap_by_ticker.get(ticker, max(base_target, current))
        risk = sizing_by_ticker.get(ticker)
        target = _dynamic_target(
            ticker=ticker,
            bucket=bucket,
            current=current,
            base_target=base_target,
            score=score,
            cap=cap,
            daily_vol=daily_vol_by_ticker.get(ticker, 0.02),
            sizing_reasons=getattr(risk, "reasons", []),
        )
        cap = max(target, cap)
        current_bucket_weights[bucket] = current_bucket_weights.get(bucket, 0.0) + current
        delta_weight = target - current
        delta_value = delta_weight * net_liquidation
        if delta_value > 25:
            total_buy_need += delta_value
            action = "add_if_cash_available"
        elif delta_value < -25:
            action = "trim_to_target"
            total_trim_value += abs(delta_value)
        else:
            action = "hold_near_target"
        rows.append(
            {
                "ticker": ticker,
                "bucket": bucket,
                "score": round(score, 1),
                "current_allocation": round(current, 4),
                "target_allocation": round(target, 4),
                "max_allocation": round(cap, 4),
                "delta_value": round(delta_value, 2),
                "action": action,
                "note": _note(action, spendable_cash),
            }
        )

    effective_reserve_value = cash_value + cash_reserve_value
    reserve_gap_after_cash_equivalents = max(0.0, reserve - cash_reserve_value)
    spendable_cash = max(0.0, cash_value - reserve_gap_after_cash_equivalents)
    rebalance_buying_power = max(0.0, cash_value + total_trim_value - reserve_gap_after_cash_equivalents)
    remaining_cash = rebalance_buying_power
    for row in sorted(rows, key=lambda item: (-float(item["score"]), -float(item["delta_value"]))):
        if row["action"] != "add_if_cash_available":
            continue
        required = float(row["delta_value"])
        if remaining_cash <= 25:
            row["action"] = "wait_for_cash_or_trim"
            row["note"] = "再平衡资金不足，等待减仓成交或新增现金"
            row["feasible_trade_value"] = 0.0
            continue
        feasible = min(required, remaining_cash)
        remaining_cash -= feasible
        row["feasible_trade_value"] = round(feasible, 2)
        if feasible < required:
            row["note"] = f"减仓后最多先买 ${feasible:,.0f}，剩余目标等下一次再平衡"
        else:
            row["note"] = "可用减仓资金覆盖目标差额，但仍需等买点和止损合理"

    rows.sort(key=lambda row: (str(row.get("bucket", "")), row["action"] != "trim_to_target", -abs(float(row["delta_value"]))))
    feasible_new_buying = min(rebalance_buying_power, total_buy_need)
    bucket_rows = _build_bucket_rows(bucket_targets, current_bucket_weights, net_liquidation)
    estimated_daily_volatility = _estimate_daily_volatility(rows, daily_vol_by_ticker, cash_weight)
    target_min = float(target_daily_volatility.get("min", 0.02))
    target_max = float(target_daily_volatility.get("max", 0.03))
    target_cash_value = net_liquidation * minimum_cash_weight
    cash_shortfall = max(0.0, target_cash_value - effective_reserve_value)
    action_plan = _build_action_plan(rows, cash_shortfall)
    return {
        "risk_profile": policy.get("risk_profile", "中等风险"),
        "net_liquidation": round(net_liquidation, 2),
        "cash_value": round(cash_value, 2),
        "cash_weight": round(cash_weight, 4),
        "target_cash_value": round(target_cash_value, 2),
        "cash_shortfall": round(cash_shortfall, 2),
        "minimum_cash_weight": round(minimum_cash_weight, 4),
        "target_daily_volatility_min": round(target_min, 4),
        "target_daily_volatility_max": round(target_max, 4),
        "estimated_daily_volatility": round(estimated_daily_volatility, 4),
        "risk_budget_status": _risk_budget_status(estimated_daily_volatility, target_min, target_max),
        "total_buy_need": round(total_buy_need, 2),
        "total_trim_value": round(total_trim_value, 2),
        "feasible_new_buying": round(feasible_new_buying, 2),
        "rebalance_buying_power": round(rebalance_buying_power, 2),
        "constraint_note": _constraint_note(
            cash_value, net_liquidation, minimum_cash_weight, total_buy_need, total_trim_value, estimated_daily_volatility, target_max
        ),
        "action_plan": action_plan,
        "bucket_rows": bucket_rows,
        "rows": rows,
    }


def _note(action: str, spendable_cash: float) -> str:
    if action == "trim_to_target":
        return "当前超过目标仓位，优先减到目标或不再加仓"
    if action == "add_if_cash_available":
        if spendable_cash <= 25:
            return "现金低于缓冲线，先不买"
        return "有加仓空间，仍需等买点和止损合理"
    return "接近目标仓位，先持有观察"


def _constraint_note(
    cash_value: float,
    net_liquidation: float,
    minimum_cash_weight: float,
    total_buy_need: float,
    total_trim_value: float,
    estimated_daily_volatility: float,
    target_max: float,
) -> str:
    reserve = net_liquidation * minimum_cash_weight
    spendable = max(0.0, cash_value + total_trim_value - reserve)
    vol_text = f"模型估算组合日波动约 {estimated_daily_volatility:.1%}，目标上限 {target_max:.1%}。"
    if estimated_daily_volatility > target_max:
        return vol_text + " 当前组合波动偏高，优先降低高波动硬件/单主题集中度。"
    if total_buy_need <= 0:
        return vol_text + " 当前主要任务不是扩仓，而是等待更好的买点或降低超配风险。"
    if spendable <= 0:
        return vol_text + " 现金需要优先保留为风险缓冲，不能按所有理论目标一次性加仓。"
    if total_buy_need > spendable:
        return vol_text + f" 减掉超配后可再分配约 ${spendable:,.0f}，理论加仓需求约 ${total_buy_need:,.0f}，需要按优先级分批。"
    return vol_text + " 减掉超配后释放的资金足以覆盖模型建议的新增仓位，但仍应分批执行。"


def _risk_budget_status(estimated_daily_volatility: float, target_min: float, target_max: float) -> str:
    if estimated_daily_volatility > target_max:
        return "above_target"
    if estimated_daily_volatility < target_min:
        return "below_target"
    return "within_target"


def _build_action_plan(rows: list[dict[str, Any]], cash_shortfall: float) -> list[dict[str, Any]]:
    """Build a cash-aware sequence that is easier to act on than raw targets."""

    actions: list[dict[str, Any]] = []
    trims = sorted(
        [row for row in rows if row.get("action") == "trim_to_target"],
        key=lambda item: float(item.get("delta_value", 0.0)),
    )
    buys = sorted(
        [row for row in rows if float(row.get("feasible_trade_value", 0.0) or 0.0) > 0],
        key=lambda item: (-float(item.get("score", 0.0) or 0.0), -float(item.get("feasible_trade_value", 0.0) or 0.0)),
    )
    for row in trims:
        amount = abs(float(row.get("delta_value", 0.0) or 0.0))
        actions.append(
            {
                "step": len(actions) + 1,
                "ticker": row.get("ticker"),
                "bucket": row.get("bucket"),
                "side": "sell_or_stop_adding",
                "amount": round(amount, 2),
                "reason": "先处理超配仓位，释放现金并把组合波动压在2-3%目标区间内",
            }
        )
    if cash_shortfall > 25:
        actions.append(
            {
                "step": len(actions) + 1,
                "ticker": "CASH",
                "bucket": "现金",
                "side": "reserve_cash",
                "amount": round(cash_shortfall, 2),
                "reason": "先补足最低现金缓冲，避免为了理论目标把账户现金打空",
            }
        )
    for row in buys:
        actions.append(
            {
                "step": len(actions) + 1,
                "ticker": row.get("ticker"),
                "bucket": row.get("bucket"),
                "side": "conditional_buy",
                "amount": round(float(row.get("feasible_trade_value", 0.0) or 0.0), 2),
                "reason": "只有在前面减仓/现金缓冲完成后才分批买入；仍需要看买点、支撑位和止损距离",
            }
        )
    return actions


def _default_policy() -> dict[str, Any]:
    return {
        "risk_profile": "中等风险",
        "minimum_cash_weight": 0.08,
        "cash_reserve_tickers": ["SCHD", "SGOV", "BIL", "SHV"],
        "target_daily_volatility": {"min": 0.02, "max": 0.03},
        "buckets": [
            {"name": "现金", "target": 0.08, "daily_volatility": 0.0, "tickers": {}},
            {"name": "核心指数", "target": 0.10, "daily_volatility": 0.012, "tickers": {"QQQ": 0.10}},
            {"name": "红利/防守质量", "target": 0.10, "daily_volatility": 0.010, "tickers": {"BRK.B": 0.10}},
            {"name": "AI硬件/半导体", "target": 0.24, "daily_volatility": 0.026, "tickers": {"NVDA": 0.07, "AVGO": 0.07, "MU": 0.05, "COHR": 0.03, "DRAM": 0.02}},
            {"name": "软件/云/互联网消费", "target": 0.22, "daily_volatility": 0.018, "tickers": {"MSFT": 0.06, "GOOGL": 0.06, "AMZN": 0.06, "ORCL": 0.04}},
            {"name": "通信/光网络", "target": 0.06, "daily_volatility": 0.021, "tickers": {"CSCO": 0.02, "GLW": 0.02, "NOK": 0.01, "LYTE": 0.01}},
            {"name": "电力/能源基础设施", "target": 0.04, "daily_volatility": 0.024, "tickers": {"VST": 0.04}},
            {"name": "医疗创新观察", "target": 0.04, "daily_volatility": 0.018, "tickers": {"ISRG": 0.04}},
        ],
    }


def _ticker_targets(policy: dict[str, Any]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for bucket in policy.get("buckets", []):
        tickers = bucket.get("tickers", {})
        if isinstance(tickers, dict):
            for ticker, target in tickers.items():
                targets[str(ticker).upper()] = float(target)
    return targets


def _bucket_targets(policy: dict[str, Any]) -> dict[str, float]:
    return {str(bucket.get("name", "")): float(bucket.get("target", 0.0)) for bucket in policy.get("buckets", [])}


def _bucket_by_ticker(policy: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bucket in policy.get("buckets", []):
        name = str(bucket.get("name", "未分类"))
        tickers = bucket.get("tickers", {})
        if isinstance(tickers, dict):
            for ticker in tickers:
                mapping[str(ticker).upper()] = name
    return mapping


def _daily_vol_by_ticker(policy: dict[str, Any]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for bucket in policy.get("buckets", []):
        vol = float(bucket.get("daily_volatility", 0.02))
        tickers = bucket.get("tickers", {})
        if isinstance(tickers, dict):
            for ticker in tickers:
                mapping[str(ticker).upper()] = vol
    return mapping


def _fallback_bucket(row: dict[str, Any]) -> str:
    sector = str(row.get("sector", ""))
    if "ETF" in sector:
        return "核心指数"
    if "Semiconductor" in sector or "AI Infrastructure" in sector:
        return "AI核心半导体"
    if "Software" in sector or "Internet" in sector:
        return "软件/云核心"
    if "Network" in sector or "Optical" in sector:
        return "退出观察/不新增"
    if "Healthcare" in sector:
        return "质量成长"
    if "Power" in sector or "Utilities" in sector:
        return "小卫星机会"
    if "Financial" in sector or "Dividend" in sector:
        return "现金/防守储备"
    return "未分类"


def _fallback_target(bucket: str, current: float, bucket_targets: dict[str, float]) -> float:
    if bucket == "未分类":
        return min(current, 0.02)
    return min(max(current, 0.01), bucket_targets.get(bucket, 0.03))


def _build_bucket_rows(bucket_targets: dict[str, float], current_bucket_weights: dict[str, float], net_liquidation: float) -> list[dict[str, Any]]:
    rows = []
    for bucket, target in {**bucket_targets, **{key: current_bucket_weights[key] for key in current_bucket_weights if key not in bucket_targets}}.items():
        current = current_bucket_weights.get(bucket, 0.0)
        rows.append(
            {
                "bucket": bucket,
                "current_allocation": round(current, 4),
                "target_allocation": round(target, 4),
                "delta_value": round((target - current) * net_liquidation, 2),
            }
        )
    return rows


def _dynamic_target(
    *,
    ticker: str,
    bucket: str,
    current: float,
    base_target: float,
    score: float,
    cap: float,
    daily_vol: float,
    sizing_reasons: list[str],
) -> float:
    """Tilt target allocations by quality, opportunity, and risk instead of fixed weights."""

    quality_tilt = 0.75 + max(0.0, min(score, 100.0)) / 100.0 * 0.55
    if daily_vol <= 0.014:
        vol_tilt = 1.12
    elif daily_vol <= 0.020:
        vol_tilt = 1.04
    elif daily_vol <= 0.026:
        vol_tilt = 0.94
    else:
        vol_tilt = 0.82
    if "High volatility caps allocation" in sizing_reasons or "Large historical drawdown requires tighter sizing" in sizing_reasons:
        vol_tilt *= 0.78
    if "High correlation reduces diversification benefit" in sizing_reasons:
        vol_tilt *= 0.90
    if "Sector exposure is already high" in sizing_reasons:
        vol_tilt *= 0.88
    potential_tilt = 1.10 if score >= 80 and current < base_target else 1.0
    target = base_target * quality_tilt * vol_tilt * potential_tilt
    if bucket in {"AI硬件/半导体", "AI核心半导体", "电力/能源基础设施", "小卫星机会"}:
        target = min(target, base_target * 1.28)
    if bucket in {"软件/云/互联网消费", "软件/云核心", "红利/防守质量", "现金/防守储备", "质量成长"}:
        target = min(target, base_target * 1.18)
    if bucket == "退出观察/不新增":
        target = min(target, base_target)
    target = min(target, cap)
    if ticker.upper() in {"NVDA", "AVGO"}:
        target = max(target, min(base_target, cap))
    return round(max(0.0, target), 4)


def _estimate_daily_volatility(rows: list[dict[str, Any]], daily_vol_by_ticker: dict[str, float], cash_weight: float) -> float:
    weighted = 0.0
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        weight = float(row.get("target_allocation", 0.0) or 0.0)
        vol = daily_vol_by_ticker.get(ticker, 0.02)
        weighted += weight * vol
    return max(0.0, weighted * 1.25 + cash_weight * 0.0)
