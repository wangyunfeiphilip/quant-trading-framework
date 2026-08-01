# Repository Analysis

## Current Project Structure

The repository is organized as a personal quantitative research framework:

- `src/data`: market data acquisition, cleaning, validation, and feature construction
- `src/indicators`: technical indicators
- `src/strategies`: momentum, mean reversion, and factor investing strategies
- `src/backtesting`: execution model, portfolio accounting, and backtest engine
- `src/risk`: performance and benchmark-relative risk metrics
- `src/models`: machine learning prediction, factor regression, and time-series tools
- `src/portfolio`: mean-variance portfolio optimization
- `src/research`: single-stock market monitor
- `cpp_engine`: C++17 order, position, portfolio, and backtesting components
- `notebooks`: research notebook
- `research`: experiment notes and research questions
- `tests`: unit tests

## Existing Modules

The project includes data processing, technical indicators, signal generation, portfolio simulation, risk analytics, Fama-French proxy regression, machine learning prediction, portfolio optimization, and C++ simulation components.

## Existing Quantitative Models

- Cross-sectional momentum
- Bollinger Band mean reversion
- Composite multi-factor scoring
- CAPM alpha and beta
- Fama-French three-factor regression
- Mean-variance optimization
- ARIMA time-series forecasting
- Linear Regression, Random Forest, and Gradient Boosting return prediction

## Existing Problems

- yfinance fundamentals are not point-in-time historical fundamentals.
- The equity universe is small and not survivorship-bias-free.
- C++ engine is currently standalone and not bound into Python.
- Real result files are generated only after data download and execution.
- GitHub remote publishing requires either `gh` CLI or repository access through the GitHub app.

## Recommended Improvements

- Add point-in-time fundamentals from a professional data source.
- Add pybind11 bindings between Python and C++.
- Add walk-forward strategy parameter validation.
- Add CI coverage for C++ compilation.
- Add benchmark regime analysis and stress tests.
