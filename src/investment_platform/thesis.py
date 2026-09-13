"""Investment thesis tracking and invalidation checks."""

from __future__ import annotations

from investment_platform.schemas import InvestmentThesis, ThesisCondition


def build_thesis(raw: dict[str, object]) -> InvestmentThesis:
    conditions = [
        ThesisCondition(
            metric=str(item["metric"]),
            operator=str(item["operator"]),
            threshold=float(item["threshold"]),
            periods=int(item.get("periods", 1)),
            message=str(item.get("message", "")),
        )
        for item in raw.get("invalidation_conditions", [])
        if isinstance(item, dict)
    ]
    return InvestmentThesis(
        ticker=str(raw["ticker"]),
        thesis=str(raw["thesis"]),
        why_follow=[str(item) for item in raw.get("why_follow", [])],
        supporting_evidence=[str(item) for item in raw.get("supporting_evidence", [])],
        key_risks=[str(item) for item in raw.get("key_risks", [])],
        invalidation_conditions=conditions,
    )


class ThesisTracker:
    """Evaluate whether stored theses remain active, weakened, or broken."""

    def evaluate(self, theses: list[InvestmentThesis], metrics_by_ticker: dict[str, dict[str, object]]) -> list[InvestmentThesis]:
        results: list[InvestmentThesis] = []
        for thesis in theses:
            triggered = []
            ticker_metrics = metrics_by_ticker.get(thesis.ticker, {})
            for condition in thesis.invalidation_conditions:
                if _condition_triggered(condition, ticker_metrics.get(condition.metric)):
                    triggered.append(condition.message or _default_message(condition))
            thesis.triggered_conditions = triggered
            thesis.status = "weakened" if triggered else "active"
            results.append(thesis)
        return results


def _condition_triggered(condition: ThesisCondition, value: object) -> bool:
    if value is None:
        return False
    values = value if isinstance(value, list) else [value]
    numeric = [float(item) for item in values if item is not None]
    if not numeric:
        return False
    lookback = numeric[-condition.periods :]
    if len(lookback) < condition.periods:
        return False
    return all(_compare(item, condition.operator, condition.threshold) for item in lookback)


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"unsupported operator: {operator}")


def _default_message(condition: ThesisCondition) -> str:
    return f"{condition.metric} {condition.operator} {condition.threshold}"
