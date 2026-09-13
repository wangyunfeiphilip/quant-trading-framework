"""Market sentiment classification."""

from __future__ import annotations

from investment_platform.schemas import SentimentRead


class SentimentAnalyzer:
    """Separate fundamentally supported moves from speculation-driven moves."""

    def classify(self, rows: list[dict[str, object]]) -> list[SentimentRead]:
        reads: list[SentimentRead] = []
        for row in rows:
            ticker = str(row["ticker"])
            news_score = float(row.get("news_score", 0.0))
            social_heat = float(row.get("social_heat", 0.0))
            evidence_quality = float(row.get("evidence_quality", 0.0))
            reasons: list[str] = []

            if news_score > 0.55 and evidence_quality >= 0.65:
                label = "Fundamental driven"
                reasons.append("Positive news is backed by stronger evidence")
            elif social_heat > 0.75 and evidence_quality < 0.50:
                label = "Speculation driven"
                reasons.append("Social heat is high while evidence quality is weak")
            else:
                label = "Mixed"
                reasons.append("Move needs more evidence before classification")

            speculation_risk = "high" if social_heat > 0.75 and evidence_quality < 0.55 else "normal"
            reads.append(SentimentRead(ticker=ticker, label=label, speculation_risk=speculation_risk, reasons=reasons))
        return reads
