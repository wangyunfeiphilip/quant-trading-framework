"""Command-line entry point for the quantitative research pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from backtesting.engine import BacktestEngine
from config_utils import load_config
from data.data_loader import MarketDataConfig, build_market_dataset, generate_data_quality_report
from models.factor_model import build_proxy_factors, fama_french_regression
from portfolio.optimization import annualized_covariance, annualized_mean_returns, efficient_frontier
from risk.risk_metrics import maximum_drawdown, performance_summary
from strategies.factor_strategy import generate_factor_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights


def _select_strategy(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    strategy_config = config.get("strategy", {})
    strategy_name = strategy_config.get("name", "momentum")
    top_n = int(strategy_config.get("top_n", 3))

    if strategy_name == "momentum":
        horizons = tuple(int(value) for value in strategy_config.get("momentum_horizons", [21, 63, 126]))
        return generate_momentum_weights(features, top_n=top_n, horizons=horizons)
    if strategy_name == "mean_reversion":
        return generate_mean_reversion_weights(features)
    if strategy_name == "factor":
        return generate_factor_weights(features, top_n=top_n)
    raise ValueError(f"unsupported strategy: {strategy_name}")


def _save_performance_chart(portfolio_value: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio_value["date"], portfolio_value["total_value"], label="Strategy")
    ax.set_title("Portfolio Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_drawdown_chart(portfolio_value: pd.DataFrame, output_path: Path) -> None:
    _, drawdown = maximum_drawdown(portfolio_value["daily_return"])
    dates = pd.to_datetime(portfolio_value["date"]).iloc[-len(drawdown):]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dates, drawdown, 0, alpha=0.35)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_benchmark_chart(strategy_returns: pd.Series, benchmark_returns: pd.Series, output_path: Path) -> None:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["Strategy", "SPY"]
    cumulative = (1 + aligned).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative.index, cumulative["Strategy"], label="Strategy")
    ax.plot(cumulative.index, cumulative["SPY"], label="SPY")
    ax.set_title("Benchmark Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_factor_exposure_chart(exposure: pd.DataFrame, output_path: Path) -> None:
    chart_data = exposure.drop(index=["r_squared"], errors="ignore").dropna(subset=["coefficient"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chart_data.index, chart_data["coefficient"])
    ax.set_title("Factor Exposure")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Coefficient")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_efficient_frontier(features: pd.DataFrame, config: dict, output_dir: Path) -> None:
    pivot_returns = features.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="last")
    pivot_returns = pivot_returns.drop(columns=["SPY"], errors="ignore").dropna(how="all").fillna(0)
    if pivot_returns.shape[1] < 2:
        return

    expected_returns = annualized_mean_returns(pivot_returns)
    covariance = annualized_covariance(pivot_returns)
    points = int(config.get("portfolio_optimization", {}).get("frontier_points", 25))
    frontier = efficient_frontier(expected_returns, covariance, points=points)
    frontier.to_csv(output_dir / "efficient_frontier.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier["volatility"], frontier["expected_return"], marker="o", linewidth=1)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "efficient_frontier.png", dpi=150)
    plt.close(fig)


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    data_config = config.get("data", {})
    backtest_config = config.get("backtest", {})
    tickers = tuple(config.get("universe", ["NVDA", "MSFT", "AAPL", "GOOGL", "META", "SPY"]))

    market_config = MarketDataConfig(
        tickers=tickers,
        start=data_config.get("start", "2015-01-01"),
        end=data_config.get("end", "2026-12-31"),
        raw_dir=PROJECT_ROOT / data_config.get("raw_dir", "data/raw"),
        processed_dir=PROJECT_ROOT / data_config.get("processed_dir", "data/processed"),
        abnormal_return_threshold=float(data_config.get("abnormal_return_threshold", 0.50)),
    )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    features = build_market_dataset(market_config)
    quality_report = generate_data_quality_report(features)
    quality_report.to_csv(results_dir / "data_quality_report.csv", index=False)

    weights = _select_strategy(features, config)
    engine = BacktestEngine(
        initial_capital=float(backtest_config.get("initial_capital", 100000)),
        transaction_cost_bps=float(backtest_config.get("transaction_cost_bps", 5)),
        slippage_bps=float(backtest_config.get("slippage_bps", 2)),
        signal_lag=int(backtest_config.get("signal_lag", 1)),
    )
    result = engine.run(features, weights)
    result.portfolio_value.to_csv(results_dir / "portfolio_value.csv", index=False)
    result.trades.to_csv(results_dir / "trade_history.csv", index=False)

    strategy_returns = result.portfolio_value.set_index("date")["daily_return"]
    benchmark_returns = features[features["ticker"].eq("SPY")].set_index("date")["daily_return"]
    summary = performance_summary(strategy_returns, benchmark_returns=benchmark_returns)
    summary.to_csv(results_dir / "performance_summary.csv")

    factors = build_proxy_factors(features)
    _, exposure = fama_french_regression(strategy_returns, factors)
    exposure.to_csv(results_dir / "factor_exposure.csv")

    _save_performance_chart(result.portfolio_value, results_dir / "portfolio_performance.png")
    _save_drawdown_chart(result.portfolio_value, results_dir / "drawdown_curve.png")
    _save_benchmark_chart(strategy_returns, benchmark_returns, results_dir / "benchmark_comparison.png")
    _save_factor_exposure_chart(exposure, results_dir / "factor_exposure.png")
    _save_efficient_frontier(features, config, results_dir)

    print("Research pipeline completed. Outputs written to results/.")


if __name__ == "__main__":
    main()
