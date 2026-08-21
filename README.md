# Quantitative Research Framework: Equity Strategies and Derivatives Pricing

A personal quantitative research framework for equity strategy evaluation, portfolio construction, risk analysis, factor modeling, derivative pricing, and hedging research.

This repository is designed as a realistic Quantitative Finance / Financial Mathematics portfolio project. It does not present itself as a hedge fund system and does not include fabricated performance numbers. Backtest results and charts are generated only after running the code on market data.

## Project Overview

This project is a Python/C++ quantitative research framework that connects three areas commonly used in quantitative finance interviews and graduate-level financial mathematics coursework:

- systematic equity strategy research
- empirical asset pricing and factor regression
- derivative pricing under stochastic processes

The equity research component studies systematic trading ideas on a diversified U.S. large-cap research universe of roughly 50 liquid names plus SPY as the benchmark. The default universe includes mega-cap technology, financials, healthcare, energy, consumer, industrial, and semiconductor names. It is broader than a five-stock winner basket, but it is still not a point-in-time index membership dataset.

Representative tickers include:

```text
AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, V, MA, UNH, JNJ, XOM,
WMT, PG, HD, COST, ABBV, MRK, KO, PEP, BAC, NFLX, ADBE, CRM,
ORCL, CSCO, AMD, QCOM, TXN, TMO, ABT, ACN, MCD, NKE, DIS,
IBM, GE, CAT, HON, UPS, LOW, GS, MS, CVX, COP, AMGN, LIN, SPY
```

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
results/parameter_sensitivity.csv
results/ml_model_comparison.csv
results/factor_exposure.csv
results/derivative_pricing_comparison.csv
results/delta_hedging_frequency.csv
```

The generated findings report covers:

- momentum Sharpe ratio before and after transaction costs and slippage
- gross turnover as a fraction of average portfolio value
- mean-reversion maximum drawdown during the Feb-Mar 2020 stress window
- mean-reversion parameter sensitivity across entry thresholds
- machine learning baseline diagnostics using chronological validation
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

### Parameter Choices and Sensitivity

The project separates research assumptions from optimized parameters.

Momentum horizons:

```text
21, 63, 126 trading days
```

These correspond approximately to 1-month, 3-month, and 6-month lookback windows. They are conventional medium-term momentum horizons rather than optimized values from a grid search.

Mean-reversion threshold:

```text
entry_z = -2.0
exit_z = 0.0
```

The default entry threshold follows the common two-standard-deviation Bollinger Band heuristic. The exit rule closes the position when the price reverts to its rolling mean.

Factor strategy weights:

```text
0.30 Value + 0.30 Momentum + 0.25 Quality + 0.15 Growth
```

These are subjective research weights, not fitted parameters. They intentionally overweight value and momentum while keeping quality and growth as secondary signals. The README does not claim these weights are optimal.

The pipeline generates a parameter sensitivity file:

```text
results/parameter_sensitivity.csv
```

The current sensitivity experiment reruns the mean-reversion strategy with:

```text
entry_z in {-1.5, -2.0, -2.5}
```

Large swings in Sharpe ratio or maximum drawdown across these thresholds should be interpreted as model instability and potential overfitting risk.

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

## Machine Learning Baselines

The machine learning module is included as a controlled prediction baseline, not as a claimed alpha engine.

Prediction target:

```text
future_21d_return
```

Feature set:

- technical indicators: returns, volatility, moving averages, RSI, MACD, Bollinger Z-score
- factor-style inputs: PE, PB, market capitalization, ROE, revenue growth

Models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Validation design:

- observations are sorted chronologically by date and ticker
- train/test split is chronological rather than random
- cross validation uses `TimeSeriesSplit`
- missing feature values are imputed inside the sklearn pipeline
- forward returns are used only as labels, never as trading inputs

Generated diagnostics:

```text
results/ml_model_comparison.csv
```

The report includes cross-validation negative MSE, test R-squared, MAE, and RMSE. Weak or negative out-of-sample R-squared is treated as an informative result rather than hidden. This is intentional because return prediction is noisy and especially vulnerable to overfitting.

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
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full research pipeline:

```bash
python main.py
```

Launch the local research dashboard:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard can open before the research pipeline is run, but the stock explorer,
backtest, risk, factor, and ML pages use the generated files from `data/processed/`
and `results/`. Run `python main.py` first when you want the full dashboard populated.

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
results/parameter_sensitivity.csv
results/ml_model_comparison.csv
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

## Research Dashboard

The Streamlit dashboard provides a Chinese browser interface for the same research framework:

- search tickers, strategies, metrics, factors, and derivative concepts
- inspect processed stock features and technical indicators
- download an external Yahoo Finance ticker on demand from the stock explorer
- run strategy backtests on the processed feature dataset
- review risk summaries, factor exposures, parameter sensitivity, and ML diagnostics
- price European options with Black-Scholes, binomial tree, and Monte Carlo methods
- inspect saved delta-hedging frequency results

The dashboard reads generated files from `data/processed/` and `results/`. It can still run the derivatives calculator, search interface, and external ticker lookup before the full research pipeline has been executed. External ticker lookup is intended for single-name exploration; portfolio backtests continue to use the processed project universe so the strategy research remains reproducible.

Local access through `http://127.0.0.1:8501` only works on the machine running Streamlit. To let other users open the dashboard directly, deploy the repository to Streamlit Community Cloud or another Python app host and set `app.py` as the entry point. On Streamlit Cloud, the public URL can be renamed through the app slug, for example:

```text
quant-research-terminal.streamlit.app
```

A custom domain can be used only after deploying the app to a hosting provider that supports domain binding.

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
- The equity universe is broader than the original five-stock prototype but is still selected ex post from currently visible large-cap U.S. equities. It is not a point-in-time historical index membership dataset and therefore still contains survivorship and selection bias.
- The execution model uses simplified close-price fills.
- Transaction cost and slippage assumptions may differ from live trading.
- Fama-French factor data uses Kenneth French official factors by default, but factor exposures are still sensitive to sample period and benchmark alignment.
- Strategy parameters are research assumptions rather than optimized values. Parameter sensitivity analysis is included to expose instability rather than hide it.
- Machine learning models are baseline prediction diagnostics. They are not used as production trading signals in this repository.

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
