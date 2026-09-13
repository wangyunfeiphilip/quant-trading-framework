"""Market regime detection from macro and market breadth inputs."""

from __future__ import annotations

from investment_platform.schemas import MarketRegime


class MarketRegimeDetector:
    """Classify the current market backdrop into an investable regime."""

    def detect(self, metrics: dict[str, float | str]) -> MarketRegime:
        risk_on = 0.0
        drivers: list[str] = []
        cautions: list[str] = []

        vix = float(metrics.get("vix", 20.0))
        if vix < 18:
            risk_on += 0.20
            drivers.append("VIX is calm")
        elif vix > 28:
            cautions.append("VIX is elevated")
        else:
            risk_on += 0.08

        nasdaq_trend = float(metrics.get("nasdaq_200dma_pct", 0.0))
        if nasdaq_trend > 0.05:
            risk_on += 0.22
            drivers.append("Nasdaq is above its long-term trend")
        elif nasdaq_trend < -0.05:
            cautions.append("Nasdaq is below its long-term trend")

        breadth = float(metrics.get("market_breadth", 0.5))
        if breadth > 0.58:
            risk_on += 0.18
            drivers.append("Market breadth is supportive")
        elif breadth < 0.42:
            cautions.append("Market breadth is weak")

        credit_spread = float(metrics.get("credit_spread", 1.5))
        if credit_spread < 1.2:
            risk_on += 0.14
            drivers.append("Credit stress is contained")
        elif credit_spread > 2.0:
            cautions.append("Credit spreads are widening")

        yield_change = float(metrics.get("yield_change_30d_bps", 0.0))
        if yield_change < 20:
            risk_on += 0.12
            drivers.append("Rates are not pressuring duration assets")
        elif yield_change > 45:
            cautions.append("Treasury yields are rising quickly")

        fed_policy = str(metrics.get("fed_policy_bias", "neutral")).lower()
        if fed_policy in {"easing", "dovish"}:
            risk_on += 0.14
            drivers.append("Fed policy bias is supportive")
        elif fed_policy in {"tightening", "hawkish"}:
            cautions.append("Fed policy bias is restrictive")
        else:
            risk_on += 0.06

        risk_on = max(0.0, min(1.0, risk_on))
        if risk_on >= 0.72 and nasdaq_trend > 0:
            name = "AI Growth Environment"
        elif risk_on >= 0.60:
            name = "Risk On"
        elif risk_on >= 0.42:
            name = "Transitional"
        else:
            name = "Risk Off"

        confidence = round(0.55 + abs(risk_on - 0.50) * 0.90, 2)
        return MarketRegime(
            name=name,
            confidence=min(confidence, 0.95),
            risk_on_score=round(risk_on, 2),
            drivers=drivers,
            cautions=cautions,
        )
