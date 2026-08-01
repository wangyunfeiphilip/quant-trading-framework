# Experiments

This file records experiments that should be run and updated only after actual execution.

## Experiment 1: Momentum Lookback Sensitivity

Research question:

Does the strategy perform better using short, medium, or mixed momentum windows?

Design:

- Universe: NVDA, MSFT, AAPL, GOOGL, META
- Benchmark: SPY
- Period: 2015-2026
- Compare: 21-day, 63-day, 126-day, and weighted combined scores
- Rebalancing: monthly
- Costs: 5 bps transaction cost and 2 bps slippage

Results:

To be filled after running `python main.py` or a notebook experiment.

## Experiment 2: Mean Reversion Entry Threshold

Research question:

Does a stricter Z-score entry threshold reduce false positives?

Design:

- Compare entry thresholds: -1.5, -2.0, -2.5
- Exit threshold: 0.0
- Window: 20 trading days

Results:

To be filled after actual execution.

## Experiment 3: Multi-Factor Score Robustness

Research question:

Does combining value, momentum, quality, and growth reduce dependence on a single signal?

Design:

- Compare equal-weight factors versus manually weighted factors
- Evaluate factor exposures with proxy Fama-French regression

Results:

To be filled after actual execution.
