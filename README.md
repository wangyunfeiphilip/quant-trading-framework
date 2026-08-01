# Quant Trading Framework

A personal quantitative research framework for equity strategy evaluation, portfolio construction, risk analysis, and factor modeling.

This repository is designed as a realistic Quantitative Finance / Financial Mathematics portfolio project. It does not present itself as a hedge fund system and does not include fabricated performance numbers. Backtest results and charts are generated only after running the code on market data.

## Project Overview

The project studies systematic trading ideas on a small U.S. large-cap equity universe:

- NVDA
- MSFT
- AAPL
- GOOGL
- META
- SPY as benchmark

The default data period is 2015-2026. The pipeline uses yfinance for public market data, builds technical and fundamental features, generates strategy target weights, runs a signal-lagged backtest, and produces risk reports and charts.

Main research question:

```text
Can systematic equity strategies improve risk-adjusted returns compared with passive investing in SPY?
```

## System Architecture

```text
quant-trading-framework/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── config.yaml
├── src/
│   ├── data/
│   ├── indicators/
│   ├── strategies/
│   ├── backtesting/
│   ├── models/
│   ├── risk/
│   ├── portfolio/
│   └── research/
├── cpp_engine/
├── notebooks/
│   └── quantitative_research.ipynb
├── results/
├── tests/
├── research/
├── docs/
│   └── screenshots/
└── .github/
    └── workflows/
```

Python is responsible for:

- data acquisition
- data cleaning and validation
- statistical analysis
- strategy research
- risk analysis
- visualization
- machine learning models

C++ is responsible for:

- order simulation
- position accounting
- portfolio state updates
- computational acceleration experiments

## Quantitative Methodology

### Momentum Strategy

Investment intuition:

Assets that recently outperformed may continue to outperform over intermediate horizons because of delayed information diffusion and investor underreaction.

Trading rule:

- compute 1-month, 3-month, and 6-month returns
- rank stocks by a weighted momentum score
- hold the top-ranked names
- rebalance monthly

Mathematical score:

```text
MomentumScore_i,t = w1 R_i,t-21 + w2 R_i,t-63 + w3 R_i,t-126
```

where `R_i,t-k` is the past return of asset `i` over `k` trading days.

### Mean Reversion Strategy

Investment intuition:

Short-term price moves may overreact relative to a rolling statistical range. A large negative deviation from the rolling mean can indicate a temporary dislocation.

The strategy uses Bollinger Bands and a Z-score:

```text
z_t = (P_t - MA_t) / sigma_t
```

where:

- `P_t` is adjusted close price
- `MA_t` is the rolling mean
- `sigma_t` is rolling standard deviation

Trading rule:

- buy when `z_t <= -2`
- sell when `z_t >= 0`

### Multi-Factor Model

The factor strategy ranks stocks using a composite score:

```text
FactorScore = 0.30 Value + 0.30 Momentum + 0.25 Quality + 0.15 Growth
```

Factors:

- Value factor: lower PE and lower PB
- Momentum factor: past return
- Quality factor: ROE
- Growth factor: revenue growth

Each factor is standardized cross-sectionally by date before aggregation.

### Fama-French Three-Factor Model

The factor regression estimates whether strategy returns are explained by common equity risk factors:

```text
R_i - R_f = alpha + beta_1(MKT - RF) + beta_2(SMB) + beta_3(HML) + epsilon
```

Variables:

- `R_i`: strategy return
- `R_f`: risk-free rate
- `MKT - RF`: market excess return
- `SMB`: size factor
- `HML`: value factor
- `alpha`: unexplained excess return
- `epsilon`: residual return

The implementation uses statsmodels OLS and reports coefficients, p-values, t-statistics, and R-squared.

## Portfolio Optimization

The project includes mean-variance optimization:

```text
Expected portfolio return = w' mu
Portfolio variance = w' Sigma w
```

where:

- `w` is the portfolio weight vector
- `mu` is expected return
- `Sigma` is the covariance matrix

Implemented outputs:

- efficient frontier
- minimum volatility portfolio
- maximum Sharpe portfolio

This demonstrates the risk-return tradeoff in classical Markowitz portfolio theory.

## Backtesting Methodology

The backtesting engine models:

- initial capital
- cash
- positions
- holdings value
- buy and sell orders
- transaction costs
- slippage
- trade history
- daily mark-to-market portfolio value

Look-ahead bias is reduced by applying `signal_lag=1`. A signal computed from day `t` data is executed no earlier than day `t+1`.

Data leakage controls:

- forward returns are used only as machine learning targets
- strategy signals use historical prices and historical features
- train/test split in prediction models is chronological

## Risk Metrics

Sharpe Ratio:

```text
Sharpe = mean(R_p - R_f) / std(R_p - R_f) * sqrt(252)
```

Sortino Ratio:

```text
Sortino = mean(R_p - R_f) / downside_std(R_p - R_f) * sqrt(252)
```

Maximum Drawdown:

```text
Drawdown_t = V_t / max(V_0, ..., V_t) - 1
```

Alpha and Beta:

```text
R_p - R_f = alpha + beta(R_m - R_f) + epsilon
```

Tracking Error:

```text
TE = std(R_p - R_b) * sqrt(252)
```

## Time Series Analysis

The project includes:

- rolling volatility
- moving-average trend analysis
- ARIMA forecasting

ARIMA is included as a baseline time-series forecasting model. Its limitations are documented because financial returns are noisy, non-stationary, and regime-dependent.

## Data Pipeline

The data module:

- downloads OHLCV data with yfinance
- preserves raw and adjusted price fields
- removes duplicate records
- handles missing values
- flags abnormal price movements
- creates a data quality report
- builds a feature dataset

Generated files:

```text
data/raw/*.csv
data/processed/clean_stock_data.csv
data/processed/fundamental_features.csv
data/processed/feature_dataset.csv
data/processed/data_quality_report.csv
```

## Example Usage

Installation:

```bash
pip install -r requirements.txt
```

Run the full research pipeline:

```bash
python main.py
```

Run tests:

```bash
pytest
```

Build the C++ engine:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_engine_demo
```

Generated outputs:

```text
results/portfolio_performance.png
results/drawdown_curve.png
results/factor_exposure.png
results/benchmark_comparison.png
results/performance_summary.csv
results/trade_history.csv
results/portfolio_value.csv
results/data_quality_report.csv
results/efficient_frontier.png
results/efficient_frontier.csv
```

## Configuration

Strategy settings, universe, date range, capital, transaction costs, and slippage are stored in:

```text
config.yaml
```

## Research Notebook

The main notebook is:

```text
notebooks/quantitative_research.ipynb
```

It is structured as a research report covering objective, data collection, exploratory analysis, strategies, backtesting, risk analysis, benchmark comparison, and conclusion.

## Limitations

- Historical results do not guarantee future performance.
- Market regimes change.
- yfinance fundamentals are not point-in-time historical fundamentals.
- The universe is small and not survivorship-bias-free.
- The execution model uses simplified close-price fills.
- Transaction cost and slippage assumptions may differ from live trading.
- Factor data is proxied from the local universe unless external factor data is supplied.

## Future Improvements

- Add point-in-time fundamentals.
- Add deep learning forecasting.
- Add options pricing models.
- Add high-frequency data support.
- Add alternative data features.
- Add pybind11 bindings for Python-to-C++ simulation.
- Add walk-forward optimization and stress testing.

## Disclaimer

This repository is for educational and research purposes only. It is not investment advice and should not be used as a live trading system without further validation.
