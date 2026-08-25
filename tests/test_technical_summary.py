from dashboard.technical_summary import generate_technical_summary


def test_technical_summary_detects_bullish_alignment() -> None:
    summary = generate_technical_summary(
        {
            "adjusted_close": 120.0,
            "sma_20": 110.0,
            "sma_60": 100.0,
            "rsi_14": 62.0,
            "macd": 1.5,
            "macd_signal": 1.0,
            "macd_histogram": 0.5,
            "zscore_20": 0.8,
            "volatility_21d": 0.28,
            "return_21d": 0.08,
            "return_63d": 0.14,
            "return_126d": 0.22,
        }
    )

    assert summary.stance == "Bullish"
    assert any("bullish" in item.lower() for item in summary.bullets)


def test_technical_summary_detects_bearish_alignment() -> None:
    summary = generate_technical_summary(
        {
            "adjusted_close": 80.0,
            "sma_20": 90.0,
            "sma_60": 100.0,
            "rsi_14": 38.0,
            "macd": -1.4,
            "macd_signal": -0.8,
            "macd_histogram": -0.6,
            "zscore_20": -0.8,
            "volatility_21d": 0.35,
            "return_21d": -0.07,
            "return_63d": -0.12,
            "return_126d": -0.18,
        }
    )

    assert summary.stance == "Bearish"
    assert any("bearish" in item.lower() for item in summary.bullets)
