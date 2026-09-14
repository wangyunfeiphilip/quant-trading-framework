"""Position sizing rules for a personal equity research portfolio."""

from __future__ import annotations

from investment_platform.schemas import PositionSizingRecommendation


class PositionSizingEngine:
    """Recommend maximum allocation from score, risk, and portfolio concentration."""

    def recommend(
        self,
        ticker: str,
        score: float,
        current_allocation: float,
        volatility: float,
        max_drawdown: float,
        sector_exposure: float,
        average_correlation: float,
    ) -> PositionSizingRecommendation:
        reasons: list[str] = []
        score_cap = min(0.20, max(0.02, score / 100.0 * 0.20))
        max_allocation = score_cap

        if volatility > 0.45:
            max_allocation = min(max_allocation, 0.08)
            reasons.append("High volatility caps allocation")
        elif volatility > 0.32:
            max_allocation = min(max_allocation, 0.12)
            reasons.append("Moderate volatility limits allocation")

        if max_drawdown < -0.35:
            max_allocation = min(max_allocation, 0.08)
            reasons.append("Large historical drawdown requires tighter sizing")

        if sector_exposure > 0.40:
            max_allocation = min(max_allocation, 0.10)
            reasons.append("Sector exposure is already high")

        if average_correlation > 0.70:
            max_allocation = min(max_allocation, 0.12)
            reasons.append("High correlation reduces diversification benefit")

        max_allocation = round(max_allocation, 4)
        if current_allocation > max_allocation:
            action = "trim_or_hold"
            reasons.append("Current allocation is above recommended maximum")
        elif score >= 80 and current_allocation < max_allocation * 0.60:
            action = "eligible_to_add"
            reasons.append("High score leaves room under the risk cap")
        else:
            action = "hold_or_watch"
            reasons.append("Allocation is within the current risk budget")

        return PositionSizingRecommendation(
            ticker=ticker,
            current_allocation=round(current_allocation, 4),
            max_allocation=max_allocation,
            suggested_action=action,
            reasons=reasons,
        )
