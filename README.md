# Quant Trading Framework

A personal quantitative research framework for equity strategy evaluation, portfolio construction, risk analysis, factor modeling, derivative pricing, and hedging research.

This repository is designed as a realistic Quantitative Finance / Financial Mathematics portfolio project. It does not present itself as a hedge fund system and does not include fabricated performance numbers. Backtest results and charts are generated only after running the code on market data.

## Project Overview

This project is a Python/C++ quantitative research framework that connects three areas commonly used in quantitative finance interviews and graduate-level financial mathematics coursework:

- systematic equity strategy research
- empirical asset pricing and factor regression
- derivative pricing under stochastic processes

The equity research component studies systematic trading ideas on a small U.S. large-cap universe:

- NVDA
- MSFT
- AAPL
- GOOGL
- META
- SPY as benchmark

The default data period is 2015-2026. The pipeline uses yfinance for public market data, builds technical and fundamental features, generates strategy target weights, runs a signal-lagged backtest, and produces risk reports and charts.

The asset-pricing component estimates Fama-French factor exposures using the official Kenneth French daily factor database and Newey-West HAC robust standard errors. This avoids constructing SMB and HML from an unrealistically narrow mega-cap technology universe.

The derivatives component adds a financial mathematics layer: Black-Scholes pricing, Greeks, binomial trees, Monte Carlo simulation with variance reduction, convergence diagnostics, and a discrete delta-hedging experiment for a short option position. The hedging simulation reuses the existing portfolio and execution-cost infrastructure.

Main research question:

```text
Can systematic equity strategies improve risk-adjusted returns compared with passive investing in SPY, and how do option-pricing models behave under analytical, numerical, and dynamically hedged implementations?
```

## Key Findings

The project now generates a reproducible findings report rather than hard-coding stale performance numbers in the README. Running:

```bash
python main.py
```

creates:

```text
results/key_findings.md
results/strategy_cost_comparison.csv
results/mean_reversion_2020_stress.csv
results/factor_exposure.csv
results/derivative_pricing_comparison.csv
results/delta_hedging_frequency.csv
```

The generated findings report covers:

- momentum Sharpe ratio before and after transaction costs and slippage
- gross turnover as a fraction of average portfolio value
- mean-reversion maximum drawdown during the Feb-Mar 2020 stress window
- Fama-French alpha and p-value using official Kenneth French factors
- cost-adjusted optimal hedge frequency for a short-option delta hedge

This keeps the repository honest: numerical claims are tied to the current data pull, configuration, and generated output files.

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
│   ├── derivatives/
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
- derivative pricing and hedging research
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

The implementation uses the official Kenneth French daily factor database by default instead of constructing SMB and HML from the small technology-heavy project universe. This matters because NVDA, MSFT, AAPL, GOOGL, and META have limited size and value-style cross-sectional variation.

The regression is estimated with statsmodels OLS and Newey-West HAC standard errors:

```text
result = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
```

HAC standard errors are used because financial return residuals often exhibit heteroskedasticity and serial correlation. The report includes coefficients, robust standard errors, p-values, t-statistics, and R-squared.

### Derivative Pricing and Delta Hedging

The derivatives module adds a stochastic-process component. The stock price follows geometric Brownian motion:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

Under the risk-neutral measure:

```text
dS_t = (r - q) S_t dt + sigma S_t dW_t
```

The Black-Scholes PDE is:

```text
partial V / partial t + 0.5 sigma^2 S^2 partial^2 V / partial S^2
+ (r - q) S partial V / partial S - rV = 0
```

Implemented derivative research:

- Black-Scholes closed-form price
- five Greeks: delta, gamma, vega, theta, rho
- Cox-Ross-Rubinstein binomial tree
- Monte Carlo pricing under the risk-neutral process
- antithetic variates and control variates with reported variance reduction
- Monte Carlo convergence curve against the analytical price
- short-call delta-hedging simulation using the existing portfolio and execution models

The delta-hedging experiment reports replication error across hedge frequencies. Without transaction costs, discrete hedging error is expected to decline as hedge frequency rises. With costs, very frequent hedging can become suboptimal, so the experiment reports the cost-adjusted hedge interval.

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
data/raw/fama_french_daily_factors.csv
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
results/performance_summary_zero_cost.csv
results/strategy_cost_comparison.csv
results/key_findings.md
results/trade_history.csv
results/portfolio_value.csv
results/data_quality_report.csv
results/efficient_frontier.png
results/efficient_frontier.csv
results/fama_french_factors.csv
results/factor_exposure.csv
results/derivative_pricing_comparison.csv
results/monte_carlo_convergence.csv
results/monte_carlo_convergence.png
results/delta_hedging_frequency.csv
results/delta_hedging_frequency.png
```

## Configuration

Strategy settings, universe, date range, capital, transaction costs, slippage, Fama-French settings, and derivative experiment settings are stored in:

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
