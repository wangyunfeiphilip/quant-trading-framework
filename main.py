"""Command-line entry point for the quantitative research pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from backtesting.engine import BacktestEngine
from config_utils import load_config
from data.data_loader import MarketDataConfig, build_market_dataset, generate_data_quality_report
from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price
from derivatives.delta_hedging import hedging_frequency_experiment
from derivatives.numerical_methods import binomial_option_price, monte_carlo_convergence, monte_carlo_option_price
from models.factor_model import build_proxy_factors, fama_french_regression, load_kenneth_french_factors
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


def _format_percent(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _portfolio_turnover(trades: pd.DataFrame, portfolio_value: pd.DataFrame) -> float:
    """Estimate dollar turnover divided by average portfolio value."""

    if trades.empty or portfolio_value.empty:
        return np.nan
    gross_notional = pd.to_numeric(trades["notional"], errors="coerce").abs().sum()
    average_value = pd.to_numeric(portfolio_value["total_value"], errors="coerce").mean()
    return float(gross_notional / average_value) if average_value else np.nan


def _load_factor_data(features: pd.DataFrame, config: dict, market_config: MarketDataConfig) -> pd.DataFrame:
    """Load official Fama-French factors with proxy fallback for offline runs."""

    factor_config = config.get("factor_model", {})
    source = factor_config.get("source", "kenneth_french")
    if source == "kenneth_french":
        cache_path = PROJECT_ROOT / factor_config.get("cache_path", "data/raw/fama_french_daily_factors.csv")
        try:
            url = factor_config.get("url")
            return load_kenneth_french_factors(
                start=market_config.start,
                end=market_config.end,
                cache_path=cache_path,
                **({"url": url} if url else {}),
                refresh=bool(factor_config.get("refresh", False)),
            )
        except Exception as exc:
            print(f"Kenneth French factor download failed; using local proxy fallback: {exc}")
    return build_proxy_factors(features)


def _save_strategy_comparison(
    net_summary: pd.Series,
    gross_summary: pd.Series,
    turnover: float,
    output_path: Path,
) -> pd.DataFrame:
    comparison = pd.DataFrame({"net_of_costs": net_summary, "zero_cost": gross_summary})
    comparison.loc["sharpe_erosion_from_costs", "net_of_costs"] = (
        gross_summary.get("sharpe_ratio", np.nan) - net_summary.get("sharpe_ratio", np.nan)
    )
    comparison.loc["turnover_ratio", "net_of_costs"] = turnover
    comparison.to_csv(output_path)
    return comparison


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
    chart_data = exposure.drop(index=["r_squared", "hac_maxlags"], errors="ignore").dropna(subset=["coefficient"])
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


def _save_monte_carlo_convergence_chart(convergence: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(convergence["n_paths"], convergence["absolute_error"], marker="o", label="MC absolute error")
    reference = convergence["absolute_error"].iloc[0] * np.sqrt(convergence["n_paths"].iloc[0] / convergence["n_paths"])
    ax.loglog(convergence["n_paths"], reference, linestyle="--", label="1/sqrt(N) reference")
    ax.set_title("Monte Carlo Convergence")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Absolute Pricing Error")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_hedging_frequency_chart(hedging: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        hedging["hedge_interval_days"],
        hedging["rmse_replication_error"],
        marker="o",
        label="Replication RMSE",
    )
    ax1.set_xlabel("Hedge Interval (Trading Days)")
    ax1.set_ylabel("Replication RMSE")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        hedging["hedge_interval_days"],
        hedging["avg_transaction_cost"],
        color="tab:red",
        marker="s",
        label="Average Transaction Cost",
    )
    ax2.set_ylabel("Average Transaction Cost")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_derivative_research(config: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    derivative_config = config.get("derivatives", {})
    contract_config = derivative_config.get("contract", {})
    contract = OptionContract(
        spot=float(contract_config.get("spot", 100.0)),
        strike=float(contract_config.get("strike", 100.0)),
        maturity=float(contract_config.get("maturity", 1.0)),
        rate=float(contract_config.get("rate", 0.04)),
        volatility=float(contract_config.get("volatility", 0.20)),
        dividend_yield=float(contract_config.get("dividend_yield", 0.0)),
        option_type=contract_config.get("option_type", "call"),
    )

    mc_config = derivative_config.get("monte_carlo", {})
    analytic_price = black_scholes_price(contract)
    binomial_price = binomial_option_price(contract, steps=int(derivative_config.get("binomial_steps", 500)))
    plain_mc = monte_carlo_option_price(
        contract,
        n_paths=int(mc_config.get("n_paths", 50_000)),
        seed=int(mc_config.get("seed", 7)),
        antithetic=False,
        control_variate=False,
    )
    reduced_mc = monte_carlo_option_price(
        contract,
        n_paths=int(mc_config.get("n_paths", 50_000)),
        seed=int(mc_config.get("seed", 7)),
        antithetic=True,
        control_variate=True,
    )

    greeks = black_scholes_greeks(contract)
    pricing = pd.DataFrame(
        [
            {"method": "black_scholes", "price": analytic_price, "standard_error": 0.0, "variance_reduction_ratio": np.nan},
            {"method": "binomial_tree", "price": binomial_price, "standard_error": np.nan, "variance_reduction_ratio": np.nan},
            {
                "method": "monte_carlo_plain",
                "price": plain_mc.price,
                "standard_error": plain_mc.standard_error,
                "variance_reduction_ratio": plain_mc.variance_reduction_ratio,
            },
            {
                "method": "monte_carlo_antithetic_control",
                "price": reduced_mc.price,
                "standard_error": reduced_mc.standard_error,
                "variance_reduction_ratio": reduced_mc.variance_reduction_ratio,
            },
        ]
    )
    for name, value in greeks.items():
        pricing.loc[pricing["method"].eq("black_scholes"), name] = value
    pricing.to_csv(output_dir / "derivative_pricing_comparison.csv", index=False)

    path_grid = tuple(int(value) for value in mc_config.get("path_grid", [1000, 2500, 5000, 10000, 25000, 50000]))
    convergence = monte_carlo_convergence(contract, path_grid=path_grid, seed=int(mc_config.get("seed", 7)))
    convergence.to_csv(output_dir / "monte_carlo_convergence.csv", index=False)
    _save_monte_carlo_convergence_chart(convergence, output_dir / "monte_carlo_convergence.png")

    hedge_config = derivative_config.get("delta_hedging", {})
    hedge_intervals = tuple(int(value) for value in hedge_config.get("hedge_intervals", [1, 2, 5, 10, 21]))
    hedging = hedging_frequency_experiment(
        contract,
        hedge_intervals=hedge_intervals,
        n_paths=int(hedge_config.get("n_paths", 50)),
        n_steps=int(hedge_config.get("n_steps", 252)),
        transaction_cost_bps=float(hedge_config.get("transaction_cost_bps", 5.0)),
        slippage_bps=float(hedge_config.get("slippage_bps", 2.0)),
        seed=int(hedge_config.get("seed", 11)),
    )
    hedging.to_csv(output_dir / "delta_hedging_frequency.csv", index=False)
    _save_hedging_frequency_chart(hedging, output_dir / "delta_hedging_frequency.png")
    return pricing, convergence, hedging


def _save_key_findings(
    summary: pd.Series,
    gross_summary: pd.Series,
    turnover: float,
    mean_reversion_stress_drawdown: float,
    exposure: pd.DataFrame,
    hedging: pd.DataFrame,
    output_path: Path,
) -> None:
    sharpe = float(summary.get("sharpe_ratio", np.nan))
    gross_sharpe = float(gross_summary.get("sharpe_ratio", np.nan))
    sharpe_erosion = gross_sharpe - sharpe
    best_hedge = hedging.loc[hedging["rmse_plus_cost"].idxmin()] if not hedging.empty else pd.Series(dtype=float)
    alpha = float(exposure.loc["const", "coefficient"]) if "const" in exposure.index else np.nan
    alpha_p = float(exposure.loc["const", "p_value"]) if "const" in exposure.index else np.nan

    lines = [
        "# Key Findings",
        "",
        "- Momentum net Sharpe after transaction costs and slippage: "
        f"{_format_number(sharpe)} versus zero-cost Sharpe of {_format_number(gross_sharpe)} "
        f"(erosion: {_format_number(sharpe_erosion)}).",
        f"- Momentum gross turnover over the backtest: {_format_number(turnover)} times average portfolio value.",
        "- Mean-reversion maximum drawdown during the Feb-Mar 2020 stress window: "
        f"{_format_percent(mean_reversion_stress_drawdown)}.",
        "- Fama-French alpha uses official Kenneth French factors and Newey-West HAC standard errors: "
        f"alpha={_format_number(alpha)}, p-value={_format_number(alpha_p)}.",
    ]
    if not best_hedge.empty:
        lines.append(
            "- Delta-hedging cost/error trade-off: best cost-adjusted hedge interval is "
            f"{int(best_hedge['hedge_interval_days'])} trading day(s), with RMSE "
            f"{_format_number(float(best_hedge['rmse_replication_error']))} and average transaction cost "
            f"{_format_number(float(best_hedge['avg_transaction_cost']))}."
        )
    lines.extend(
        [
            "",
            "These findings are generated by `python main.py` from the current data pull and model settings.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    gross_engine = BacktestEngine(
        initial_capital=float(backtest_config.get("initial_capital", 100000)),
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        signal_lag=int(backtest_config.get("signal_lag", 1)),
    )
    gross_result = gross_engine.run(features, weights)
    gross_returns = gross_result.portfolio_value.set_index("date")["daily_return"]
    gross_summary = performance_summary(gross_returns, benchmark_returns=benchmark_returns)
    gross_summary.to_csv(results_dir / "performance_summary_zero_cost.csv")
    turnover = _portfolio_turnover(result.trades, result.portfolio_value)
    _save_strategy_comparison(summary, gross_summary, turnover, results_dir / "strategy_cost_comparison.csv")

    mean_reversion_weights = generate_mean_reversion_weights(features)
    mean_reversion_result = engine.run(features, mean_reversion_weights)
    mr_values = mean_reversion_result.portfolio_value.copy()
    mr_values["date"] = pd.to_datetime(mr_values["date"])
    stress_window = mr_values[mr_values["date"].between("2020-02-19", "2020-03-31")]
    mean_reversion_stress_drawdown, _ = maximum_drawdown(stress_window["daily_return"])
    pd.Series(
        {
            "stress_window_start": "2020-02-19",
            "stress_window_end": "2020-03-31",
            "maximum_drawdown": mean_reversion_stress_drawdown,
        }
    ).to_csv(results_dir / "mean_reversion_2020_stress.csv")

    factors = _load_factor_data(features, config, market_config)
    factors.to_csv(results_dir / "fama_french_factors.csv", index=False)
    hac_maxlags = int(config.get("factor_model", {}).get("hac_maxlags", 5))
    _, exposure = fama_french_regression(strategy_returns, factors, hac_maxlags=hac_maxlags)
    exposure.to_csv(results_dir / "factor_exposure.csv")

    _, _, hedging = _run_derivative_research(config, results_dir)
    _save_key_findings(
        summary,
        gross_summary,
        turnover,
        mean_reversion_stress_drawdown,
        exposure,
        hedging,
        results_dir / "key_findings.md",
    )

    _save_performance_chart(result.portfolio_value, results_dir / "portfolio_performance.png")
    _save_drawdown_chart(result.portfolio_value, results_dir / "drawdown_curve.png")
    _save_benchmark_chart(strategy_returns, benchmark_returns, results_dir / "benchmark_comparison.png")
    _save_factor_exposure_chart(exposure, results_dir / "factor_exposure.png")
    _save_efficient_frontier(features, config, results_dir)

    print("Research pipeline completed. Outputs written to results/.")


if __name__ == "__main__":
    main()
