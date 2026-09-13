"""Dynamic factor weights and cross-sectional scoring."""

from __future__ import annotations

from investment_platform.schemas import FactorWeights, MarketRegime, StockScore


class DynamicFactorWeightEngine:
    """Map market regimes to factor weights and explain the shift."""

    def weights_for(self, regime: MarketRegime) -> FactorWeights:
        if regime.name == "AI Growth Environment":
            return FactorWeights(0.36, 0.18, 0.27, 0.09, 0.10, "Risk-aware AI growth tape: growth and momentum still lead, but risk gets a dedicated 10% weight")
        if regime.name == "Risk On":
            return FactorWeights(0.288, 0.207, 0.27, 0.135, 0.10, "Risk-on but not blind risk-on: factor weights are reduced 10% and risk gets 10%")
        if regime.name == "Risk Off":
            return FactorWeights(0.09, 0.35, 0.13, 0.23, 0.20, "Risk-off regime: risk control becomes a primary factor, while growth and momentum are cut")
        return FactorWeights(0.198, 0.27, 0.207, 0.18, 0.145, "Mixed tape: dynamic factors are reduced and risk receives a larger explicit weight")


def score_stocks(rows: list[dict[str, object]], weights: FactorWeights) -> list[StockScore]:
    """Score stocks using 0-100 factor inputs and normalized dynamic weights."""

    normalized = weights.normalized()
    scored: list[StockScore] = []
    for row in rows:
        ticker = str(row["ticker"])
        factors = {
            "growth": float(row.get("growth", 50.0)),
            "quality": float(row.get("quality", 50.0)),
            "momentum": float(row.get("momentum", 50.0)),
            "value": float(row.get("value", 50.0)),
            "risk": float(row.get("risk", 50.0)),
        }
        total = (
            normalized.growth * factors["growth"]
            + normalized.quality * factors["quality"]
            + normalized.momentum * factors["momentum"]
            + normalized.value * factors["value"]
            + normalized.risk * factors["risk"]
        )
        explanation = _score_explanation(factors)
        scored.append(StockScore(ticker=ticker, total_score=round(total, 1), factor_scores=factors, explanation=explanation))
    return sorted(scored, key=lambda item: item.total_score, reverse=True)


def _score_explanation(factors: dict[str, float]) -> list[str]:
    positives = [name for name, value in factors.items() if value >= 75]
    negatives = [name for name, value in factors.items() if value <= 40]
    notes: list[str] = []
    if positives:
        notes.append("Strong " + ", ".join(positives))
    if negatives:
        notes.append("Weak " + ", ".join(negatives))
    if not notes:
        notes.append("Balanced profile without an extreme factor edge")
    return notes
