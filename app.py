"""Streamlit dashboard for the quantitative research framework."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from backtesting.engine import BacktestEngine
from dashboard.search import build_search_index, search_catalog
from data.data_loader import DEFAULT_TICKERS
from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price
from derivatives.numerical_methods import binomial_option_price, monte_carlo_option_price
from risk.risk_metrics import maximum_drawdown, performance_summary
from strategies.factor_strategy import generate_factor_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


st.set_page_config(
    page_title="Quant Research Dashboard",
    page_icon="Q",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .small-note { color: #667085; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_config_tickers() -> list[str]:
    try:
        import yaml

        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text()) or {}
        tickers = config.get("universe") or list(DEFAULT_TICKERS)
        return [str(ticker).upper() for ticker in tickers]
    except Exception:
        return list(DEFAULT_TICKERS)


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    frame = read_csv(str(PROCESSED_DIR / "feature_dataset.csv"))
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_portfolio_value() -> pd.DataFrame:
    frame = read_csv(str(RESULTS_DIR / "portfolio_value.csv"))
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_key_findings() -> str:
    path = RESULTS_DIR / "key_findings.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def metric_value(series: pd.Series, key: str, precision: int = 3) -> str:
    if key not in series.index:
        return "NA"
    value = pd.to_numeric(series.loc[key], errors="coerce")
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}"


def percent_value(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.2%}"


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines", name=y))
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def render_missing_results() -> None:
    st.info("No generated research outputs found in `results/`.")


def render_search(query: str, tickers: list[str]) -> None:
    if not query.strip():
        return
    st.subheader("Search Results")
    results = search_catalog(query, build_search_index(tickers))
    if not results:
        st.write("No direct matches.")
        return
    for item in results:
        with st.container(border=True):
            st.caption(item.category)
            st.write(f"**{item.title}**")
            st.write(item.description)
            st.caption(f"Open: {item.target}")


def render_overview() -> None:
    st.header("Quantitative Research Framework")
    st.caption("Equity strategies, factor models, derivatives pricing, and hedging research")

    portfolio = load_portfolio_value()
    performance = read_csv(str(RESULTS_DIR / "performance_summary.csv"))

    if not performance.empty:
        summary = performance.iloc[:, 0] if performance.shape[1] == 1 else performance.set_index(performance.columns[0]).iloc[:, 0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cumulative Return", metric_value(summary, "cumulative_return"))
        col2.metric("Sharpe Ratio", metric_value(summary, "sharpe_ratio"))
        col3.metric("Max Drawdown", metric_value(summary, "maximum_drawdown"))
        col4.metric("Beta vs SPY", metric_value(summary, "beta_vs_benchmark"))
    else:
        render_missing_results()

    if not portfolio.empty and "total_value" in portfolio:
        st.plotly_chart(line_chart(portfolio, "date", "total_value", "Portfolio Value"), use_container_width=True)

    findings = load_key_findings()
    if findings:
        st.markdown(findings)


def render_stock_explorer(tickers: list[str]) -> None:
    st.header("Stock Explorer")
    features = load_features()
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)

    if features.empty:
        render_missing_results()
        return

    stock = features[features["ticker"].eq(ticker)].copy()
    if stock.empty:
        st.warning(f"No processed data for {ticker}.")
        return

    latest = stock.dropna(subset=["adjusted_close"]).iloc[-1]
    returns = pd.to_numeric(stock["daily_return"], errors="coerce").dropna()
    max_dd, _ = maximum_drawdown(returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Adjusted Close", f"{latest['adjusted_close']:.2f}")
    col2.metric("21D Return", percent_value(latest.get("return_21d")))
    col3.metric("126D Return", percent_value(latest.get("return_126d")))
    col4.metric("Max Drawdown", percent_value(max_dd))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock["date"], y=stock["adjusted_close"], mode="lines", name="Adjusted Close"))
    if "sma_20" in stock:
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["sma_20"], mode="lines", name="SMA 20"))
    if "sma_60" in stock:
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["sma_60"], mode="lines", name="SMA 60"))
    fig.update_layout(title=f"{ticker} Price and Moving Averages", height=440, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)

    indicator_cols = [
        "daily_return",
        "volatility_21d",
        "rsi_14",
        "macd",
        "macd_signal",
        "zscore_20",
        "return_21d",
        "return_63d",
        "return_126d",
    ]
    shown = [column for column in indicator_cols if column in stock.columns]
    st.dataframe(stock[["date", "ticker", *shown]].tail(30), use_container_width=True, hide_index=True)


def strategy_weights(features: pd.DataFrame, strategy: str, top_n: int, entry_z: float) -> pd.DataFrame:
    if strategy == "Momentum":
        return generate_momentum_weights(features, top_n=top_n)
    if strategy == "Mean Reversion":
        return generate_mean_reversion_weights(features, entry_z=entry_z)
    return generate_factor_weights(features, top_n=top_n)


def render_backtest_lab() -> None:
    st.header("Backtest Lab")
    features = load_features()
    if features.empty:
        render_missing_results()
        return

    col1, col2, col3 = st.columns(3)
    strategy = col1.selectbox("Strategy", ["Momentum", "Mean Reversion", "Factor"])
    top_n = col2.slider("Top N", 3, 20, 10)
    entry_z = col3.select_slider("Mean-Reversion Entry Z", options=[-1.5, -2.0, -2.5], value=-2.0)

    cost_col1, cost_col2, cost_col3 = st.columns(3)
    capital = cost_col1.number_input("Initial Capital", min_value=10_000, max_value=1_000_000, value=100_000, step=10_000)
    transaction_cost = cost_col2.number_input("Transaction Cost bps", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    slippage = cost_col3.number_input("Slippage bps", min_value=0.0, max_value=50.0, value=2.0, step=1.0)

    weights = strategy_weights(features, strategy, top_n=top_n, entry_z=float(entry_z))
    engine = BacktestEngine(
        initial_capital=float(capital),
        transaction_cost_bps=float(transaction_cost),
        slippage_bps=float(slippage),
        signal_lag=1,
    )
    result = engine.run(features, weights)
    returns = result.portfolio_value.set_index("date")["daily_return"]
    benchmark = features[features["ticker"].eq("SPY")].set_index("date")["daily_return"]
    summary = performance_summary(returns, benchmark_returns=benchmark)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Return", percent_value(summary.get("cumulative_return")))
    col2.metric("Annual Return", percent_value(summary.get("annualized_return")))
    col3.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', np.nan):.3f}")
    col4.metric("Max Drawdown", percent_value(summary.get("maximum_drawdown")))

    st.plotly_chart(line_chart(result.portfolio_value, "date", "total_value", f"{strategy} Backtest"), use_container_width=True)
    st.dataframe(result.trades.tail(50), use_container_width=True, hide_index=True)


def render_risk_factors() -> None:
    st.header("Risk & Factors")
    performance = read_csv(str(RESULTS_DIR / "performance_summary.csv"))
    exposure = read_csv(str(RESULTS_DIR / "factor_exposure.csv"))
    sensitivity = read_csv(str(RESULTS_DIR / "parameter_sensitivity.csv"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Summary")
        if performance.empty:
            render_missing_results()
        else:
            st.dataframe(performance, use_container_width=True)
    with col2:
        st.subheader("Factor Exposure")
        if exposure.empty:
            render_missing_results()
        else:
            st.dataframe(exposure, use_container_width=True)

    st.subheader("Parameter Sensitivity")
    if sensitivity.empty:
        render_missing_results()
    else:
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)


def render_data_ml() -> None:
    st.header("Data & ML")
    quality = read_csv(str(RESULTS_DIR / "data_quality_report.csv"))
    ml = read_csv(str(RESULTS_DIR / "ml_model_comparison.csv"))

    st.subheader("Data Quality")
    if quality.empty:
        render_missing_results()
    else:
        st.dataframe(quality, use_container_width=True, hide_index=True)

    st.subheader("Machine Learning Baselines")
    if ml.empty:
        render_missing_results()
    else:
        st.dataframe(ml, use_container_width=True, hide_index=True)


def render_derivatives_lab() -> None:
    st.header("Derivatives Lab")
    col1, col2, col3 = st.columns(3)
    spot = col1.number_input("Spot", min_value=1.0, value=100.0, step=1.0)
    strike = col2.number_input("Strike", min_value=1.0, value=100.0, step=1.0)
    maturity = col3.number_input("Maturity Years", min_value=0.01, value=1.0, step=0.05)

    col4, col5, col6 = st.columns(3)
    rate = col4.number_input("Risk-Free Rate", min_value=0.0, max_value=0.25, value=0.04, step=0.005)
    volatility = col5.number_input("Volatility", min_value=0.01, max_value=2.0, value=0.20, step=0.01)
    option_type = col6.selectbox("Option Type", ["call", "put"])

    contract = OptionContract(
        spot=float(spot),
        strike=float(strike),
        maturity=float(maturity),
        rate=float(rate),
        volatility=float(volatility),
        option_type=option_type,
    )
    bs_price = black_scholes_price(contract)
    tree_price = binomial_option_price(contract, steps=300)
    mc_paths = st.slider("Monte Carlo Paths", 1_000, 100_000, 20_000, step=1_000)
    mc = monte_carlo_option_price(contract, n_paths=int(mc_paths), seed=7, antithetic=True, control_variate=True)
    greeks = black_scholes_greeks(contract)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Black-Scholes", f"{bs_price:.4f}")
    col2.metric("Binomial Tree", f"{tree_price:.4f}")
    col3.metric("Monte Carlo", f"{mc.price:.4f}")
    col4.metric("MC Std Error", f"{mc.standard_error:.4f}")

    st.subheader("Greeks")
    greek_frame = pd.DataFrame([greeks]).T.reset_index()
    greek_frame.columns = ["Greek", "Value"]
    st.dataframe(greek_frame, use_container_width=True, hide_index=True)

    pricing = read_csv(str(RESULTS_DIR / "derivative_pricing_comparison.csv"))
    hedging = read_csv(str(RESULTS_DIR / "delta_hedging_frequency.csv"))
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Saved Pricing Comparison")
        if pricing.empty:
            render_missing_results()
        else:
            st.dataframe(pricing, use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("Saved Hedge Frequency")
        if hedging.empty:
            render_missing_results()
        else:
            st.dataframe(hedging, use_container_width=True, hide_index=True)


def main() -> None:
    tickers = load_config_tickers()
    query = st.text_input("Search ticker, strategy, metric, factor, or derivative concept", placeholder="NVDA, Sharpe, Fama-French, Black-Scholes...")
    render_search(query, tickers)

    pages = {
        "Overview": render_overview,
        "Stock Explorer": lambda: render_stock_explorer(tickers),
        "Backtest Lab": render_backtest_lab,
        "Risk & Factors": render_risk_factors,
        "Data & ML": render_data_ml,
        "Derivatives Lab": render_derivatives_lab,
    }
    selected = st.sidebar.radio("Workspace", list(pages.keys()))
    st.sidebar.caption("Quant Research Dashboard")
    pages[selected]()


if __name__ == "__main__":
    main()
