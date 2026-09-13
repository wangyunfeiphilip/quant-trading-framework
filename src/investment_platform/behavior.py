"""Personal investment behavior learning."""

from __future__ import annotations

from investment_platform.schemas import BehaviorPattern


class BehaviorAnalyzer:
    """Detect recurring behavior patterns from user trade history."""

    def analyze(self, trades: list[dict[str, object]], portfolio: list[dict[str, object]]) -> list[BehaviorPattern]:
        patterns: list[BehaviorPattern] = []
        buys = [trade for trade in trades if str(trade.get("side", "")).lower() == "buy"]
        chase_buys = [trade for trade in buys if float(trade.get("prior_day_return", 0.0)) >= 0.08]
        if len(chase_buys) >= 2:
            success_rate = _success_rate(chase_buys)
            patterns.append(
                BehaviorPattern(
                    label="Chasing strength after large one-day moves",
                    severity="medium" if success_rate >= 0.5 else "high",
                    evidence=f"{len(chase_buys)} buys occurred after prior-day moves above 8%; success rate {success_rate:.0%}.",
                    reminder="Slow down after large single-day moves and require thesis confirmation before adding.",
                )
            )

        sector_weights: dict[str, float] = {}
        for position in portfolio:
            sector = str(position.get("sector", "Unknown"))
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(position.get("allocation", 0.0))
        if sector_weights:
            sector, weight = max(sector_weights.items(), key=lambda item: item[1])
            if weight > 0.45:
                patterns.append(
                    BehaviorPattern(
                        label="Sector concentration",
                        severity="high" if weight > 0.60 else "medium",
                        evidence=f"{sector} exposure is {weight:.0%} of portfolio.",
                        reminder="Check whether new ideas diversify the portfolio or simply add the same risk again.",
                    )
                )
        return patterns


def _success_rate(trades: list[dict[str, object]]) -> float:
    evaluated = [float(trade.get("subsequent_30d_return", 0.0)) for trade in trades if trade.get("subsequent_30d_return") is not None]
    if not evaluated:
        return 0.0
    return sum(1 for value in evaluated if value > 0) / len(evaluated)
