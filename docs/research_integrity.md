# Research Integrity and Presentation Checklist

This project is intended to be presented as a reproducible research system,
not as a claim of guaranteed investment performance.

## What every published run must contain

1. `results/backtest_manifest.json`
   - data source and adjusted-price convention
   - actual data start and cutoff dates
   - universe and ticker count
   - transaction costs, slippage, rebalance frequency, and signal lag
   - configuration fingerprint and artifact SHA-256 hashes
2. `results/research_validation.json`
   - whether the required artifacts passed the publication gate
   - ordered portfolio dates and positive portfolio values
   - trade-ledger and data-quality checks
3. `results/trade_history.csv`
   - dated fills, side, quantity, reference price, executed price, notional,
     transaction cost, slippage cost, and cash after each fill
4. `results/portfolio_value.csv`
   - daily cash, holdings, total value, daily return, and cumulative return
5. `results/data_quality_report.csv`
   - per-ticker coverage, missing prices, duplicates, invalid dates, and
     abnormal moves

## What the current project claims

- Signals are lagged by at least one session before execution.
- Price cleaning and corporate-action adjustment factors never backfill a later
  observation into an earlier date.
- The default backtest includes explicit transaction cost and slippage assumptions.
- The public demo is an immutable snapshot, so its numbers do not silently change
  when a data vendor revises a historical row.

## What the current project does not claim

- The large-cap universe is not point-in-time historical index membership. It is
  selected ex post and therefore has survivorship and selection bias.
- yfinance fundamentals are current snapshots rather than point-in-time data.
- The execution model is a close-price research approximation, not a broker fill
  simulator.
- Baseline machine-learning diagnostics are not production trading signals.
- A backtest is not evidence of future returns or investment advice.

## Reproduce a run locally

```bash
python main.py
python scripts/validate_research_run.py --root .
```

The second command must print `Research validation: passed` before the outputs
are presented or published. Keep the manifest, validation report, CSV outputs,
and the commit that generated them together.

## Interview demonstration order

1. Start with the research question and data cutoff.
2. Open the audit panel and explain the one-session signal lag.
3. Open the trade ledger and show that costs and slippage are explicit.
4. Compare net-of-costs and zero-cost results.
5. Show parameter sensitivity and the 2020 stress result.
6. State survivorship, point-in-time, and execution-model limitations before
   discussing performance.
