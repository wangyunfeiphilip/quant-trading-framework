"""Shared data structures for the investment research platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class MarketRegime:
    name: str
    confidence: float
    risk_on_score: float
    drivers: list[str]
    cautions: list[str]


@dataclass(frozen=True)
class FactorWeights:
    growth: float
    quality: float
    momentum: float
    value: float
    risk: float = 0.0
    reason: str = ""

    def normalized(self) -> "FactorWeights":
        total = self.growth + self.quality + self.momentum + self.value + self.risk
        if total <= 0:
            raise ValueError("factor weights must sum to a positive number")
        return FactorWeights(
            growth=self.growth / total,
            quality=self.quality / total,
            momentum=self.momentum / total,
            value=self.value / total,
            risk=self.risk / total,
            reason=self.reason,
        )

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class StockScore:
    ticker: str
    total_score: float
    factor_scores: dict[str, float]
    explanation: list[str]


@dataclass(frozen=True)
class ThesisCondition:
    metric: str
    operator: str
    threshold: float
    periods: int = 1
    message: str = ""


@dataclass
class InvestmentThesis:
    ticker: str
    thesis: str
    why_follow: list[str]
    supporting_evidence: list[str]
    key_risks: list[str]
    invalidation_conditions: list[ThesisCondition]
    status: str = "active"
    triggered_conditions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    as_of: date
    ticker: str
    bullish_probability: float
    horizon_days: int
    expected_return: float | None = None
    thesis_ref: str | None = None


@dataclass(frozen=True)
class PredictionOutcome:
    prediction_id: str
    ticker: str
    horizon_days: int
    actual_return: float
    correct: bool
    due: bool


@dataclass(frozen=True)
class ValuationScenario:
    name: str
    fair_value: float
    probability: float
    driver: str


@dataclass(frozen=True)
class ValuationSummary:
    ticker: str
    weighted_fair_value: float
    upside_to_price: float | None
    scenarios: list[ValuationScenario]


@dataclass(frozen=True)
class PositionSizingRecommendation:
    ticker: str
    current_allocation: float
    max_allocation: float
    suggested_action: str
    reasons: list[str]


@dataclass(frozen=True)
class IndustryNode:
    theme: str
    layer: str
    ticker: str
    company: str
    moat: str
    competitive_risk: str


@dataclass(frozen=True)
class SentimentRead:
    ticker: str
    label: str
    speculation_risk: str
    reasons: list[str]


@dataclass(frozen=True)
class BehaviorPattern:
    label: str
    severity: str
    evidence: str
    reminder: str


def dataclass_to_dict(value: Any) -> Any:
    """Convert nested dataclasses into JSON-ready dictionaries."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value
