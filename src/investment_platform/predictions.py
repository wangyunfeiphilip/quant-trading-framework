"""AI prediction logging and accuracy review."""

from __future__ import annotations

from datetime import date, timedelta

from investment_platform.schemas import PredictionOutcome, PredictionRecord


def build_prediction(raw: dict[str, object]) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=str(raw["prediction_id"]),
        as_of=date.fromisoformat(str(raw["as_of"])),
        ticker=str(raw["ticker"]),
        bullish_probability=float(raw["bullish_probability"]),
        horizon_days=int(raw["horizon_days"]),
        expected_return=float(raw["expected_return"]) if raw.get("expected_return") is not None else None,
        thesis_ref=str(raw["thesis_ref"]) if raw.get("thesis_ref") is not None else None,
    )


class PredictionTracker:
    """Evaluate due AI predictions against realized price performance."""

    def evaluate(
        self,
        predictions: list[PredictionRecord],
        as_of: date,
        price_history: dict[str, dict[str, float]],
    ) -> list[PredictionOutcome]:
        outcomes: list[PredictionOutcome] = []
        for prediction in predictions:
            due_date = prediction.as_of + timedelta(days=prediction.horizon_days)
            if due_date > as_of:
                continue
            prices = price_history.get(prediction.ticker, {})
            start = prices.get(prediction.as_of.isoformat())
            end = prices.get(due_date.isoformat()) or prices.get(as_of.isoformat())
            if not start or not end:
                continue
            actual_return = end / start - 1.0
            correct = self._is_correct(prediction.bullish_probability, actual_return)
            outcomes.append(
                PredictionOutcome(
                    prediction_id=prediction.prediction_id,
                    ticker=prediction.ticker,
                    horizon_days=prediction.horizon_days,
                    actual_return=round(actual_return, 4),
                    correct=correct,
                    due=True,
                )
            )
        return outcomes

    @staticmethod
    def summary(outcomes: list[PredictionOutcome]) -> dict[str, float]:
        if not outcomes:
            return {"evaluated": 0, "accuracy": 0.0, "average_return": 0.0}
        accuracy = sum(1 for outcome in outcomes if outcome.correct) / len(outcomes)
        avg_return = sum(outcome.actual_return for outcome in outcomes) / len(outcomes)
        return {"evaluated": len(outcomes), "accuracy": round(accuracy, 3), "average_return": round(avg_return, 4)}

    @staticmethod
    def _is_correct(probability: float, actual_return: float) -> bool:
        if probability >= 0.60:
            return actual_return > 0
        if probability <= 0.40:
            return actual_return < 0
        return abs(actual_return) < 0.03
