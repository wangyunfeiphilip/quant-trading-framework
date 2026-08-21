"""Streamlit dashboard for the quantitative research framework."""

from __future__ import annotations

import re
import json
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
from data.data_loader import (
    DEFAULT_TICKERS,
    clean_price_data,
    create_feature_dataset,
    download_price_data,
    load_fundamental_features,
)
from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price
from derivatives.numerical_methods import binomial_option_price, monte_carlo_option_price
from risk.risk_metrics import maximum_drawdown, performance_summary
from strategies.factor_strategy import generate_factor_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INVESTMENT_RESULTS_DIR = RESULTS_DIR / "investment_platform"
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"


st.set_page_config(
    page_title="量化研究终端",
    page_icon="Q",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f7f9; color: #20242c; }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] {
        background: #171a21;
        border-right: 1px solid #292d36;
    }
    [data-testid="stSidebar"] * { color: #f4f6f8; }
    [data-testid="stSidebar"] .stRadio label {
        border-radius: 8px;
        padding: 4px 2px;
    }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { font-size: 2.5rem; margin-bottom: 0.35rem; }
    h2 { margin-top: 1.25rem; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetric"] label { color: #667085; }
    div[data-testid="stMetricValue"] { color: #111827; font-weight: 700; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        overflow: hidden;
    }
    .terminal-title {
        border: 1px solid #e4e7ec;
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .terminal-title .eyebrow {
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .terminal-title .title {
        color: #111827;
        font-size: 1.55rem;
        font-weight: 750;
        line-height: 1.2;
    }
    .terminal-title .subtitle {
        color: #667085;
        margin-top: 6px;
        font-size: 0.98rem;
    }
    .small-note { color: #667085; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_data_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.exists():
        return file_path
    try:
        fallback = DEMO_DATA_DIR / file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        fallback = file_path
    return fallback if fallback.exists() else file_path


def file_signature(path: str | Path) -> tuple[str, float, int]:
    file_path = resolve_data_path(path)
    if not file_path.exists():
        return str(file_path), 0.0, 0
    stat = file_path.stat()
    return str(file_path), stat.st_mtime, stat.st_size


@st.cache_data(show_spinner=False)
def read_csv(signature: tuple[str, float, int]) -> pd.DataFrame:
    path, _, size = signature
    if size == 0:
        return pd.DataFrame()
    file_path = Path(path)
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_dashboard_config() -> dict:
    try:
        import yaml

        return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text()) or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_config_tickers() -> list[str]:
    config = load_dashboard_config()
    tickers = config.get("universe") or list(DEFAULT_TICKERS)
    return [str(ticker).upper() for ticker in tickers]


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    frame = read_csv(file_signature(PROCESSED_DIR / "feature_dataset.csv"))
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def normalize_ticker(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.^=-]", "", value).upper()


@st.cache_data(show_spinner=True, ttl=3600)
def load_external_ticker_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = download_price_data([ticker], start=start, end=end, raw_dir=None)
    if raw.empty:
        raise ValueError("data provider returned no price rows")

    clean = clean_price_data(raw)
    fundamentals = load_fundamental_features([ticker])
    features = create_feature_dataset(clean, fundamentals=fundamentals)
    features["date"] = pd.to_datetime(features["date"])
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_portfolio_value() -> pd.DataFrame:
    frame = read_csv(file_signature(RESULTS_DIR / "portfolio_value.csv"))
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_key_findings() -> str:
    path = resolve_data_path(RESULTS_DIR / "key_findings.md")
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data(show_spinner=False)
def load_investment_snapshot(signature: tuple[str, float, int]) -> dict:
    path, _, size = signature
    if size == 0:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_latest_investment_report(signature: tuple[str, float, int]) -> str:
    path, _, size = signature
    if size == 0:
        return ""
    return Path(path).read_text(encoding="utf-8")


def latest_investment_snapshot_path() -> Path:
    preferred = INVESTMENT_RESULTS_DIR / "latest_snapshot.json"
    if preferred.exists():
        return preferred
    candidates = sorted(INVESTMENT_RESULTS_DIR.glob("run_snapshot_*.json"))
    return candidates[-1] if candidates else preferred


def latest_investment_report_path(snapshot: dict) -> Path:
    if snapshot.get("report_path"):
        path = PROJECT_ROOT / str(snapshot["report_path"])
        if path.exists():
            return path
    candidates = sorted(INVESTMENT_RESULTS_DIR.glob("daily_report_*.md"))
    return candidates[-1] if candidates else INVESTMENT_RESULTS_DIR / "daily_report.md"


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
    st.info("尚未找到已生成的研究结果。请先运行 `python main.py`，或使用外部股票查询与衍生品实验室。")


def render_page_title(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="terminal-title">
            <div class="eyebrow">{eyebrow}</div>
            <div class="title">{title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search(query: str, tickers: list[str]) -> None:
    if not query.strip():
        return
    st.subheader("搜索结果")
    results = search_catalog(query, build_search_index(tickers))
    if not results:
        candidate = normalize_ticker(query)
        if 1 <= len(candidate) <= 12:
            st.write(
                f"没有匹配到项目内条目。若要查看 `{candidate}`，请进入「股票浏览器」并使用外部股票查询。"
            )
        else:
            st.write("没有匹配结果。")
        return
    for item in results:
        with st.container(border=True):
            st.caption(item.category)
            st.write(f"**{item.title}**")
            st.write(item.description)
            st.caption(f"打开位置：{item.target}")


def render_overview() -> None:
    render_page_title(
        "Research Overview",
        "量化研究框架总览",
        "股票策略、因子模型、衍生品定价、风险归因与动态对冲研究。",
    )

    portfolio = load_portfolio_value()
    performance = read_csv(file_signature(RESULTS_DIR / "performance_summary.csv"))

    if not performance.empty:
        summary = performance.iloc[:, 0] if performance.shape[1] == 1 else performance.set_index(performance.columns[0]).iloc[:, 0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("累计收益", metric_value(summary, "cumulative_return"))
        col2.metric("夏普比率", metric_value(summary, "sharpe_ratio"))
        col3.metric("最大回撤", metric_value(summary, "maximum_drawdown"))
        col4.metric("Beta vs SPY", metric_value(summary, "beta_vs_benchmark"))
    else:
        render_missing_results()

    if not portfolio.empty and "total_value" in portfolio:
        st.plotly_chart(line_chart(portfolio, "date", "total_value", "组合净值曲线"), width="stretch")

    findings = load_key_findings()
    if findings:
        st.markdown(findings)


def render_ai_investment_platform() -> None:
    render_page_title(
        "AI Equity Research",
        "AI 投资研究台",
        "个人股票研究、投资假设跟踪、因子评分、估值、仓位风险与交易行为反馈。",
    )

    snapshot_path = latest_investment_snapshot_path()
    snapshot = load_investment_snapshot(file_signature(snapshot_path))
    if not snapshot:
        st.info("尚未找到 AI 投资研究快照。请先运行 `python3 scripts/run_investment_platform.py --config investment_platform.json`。")
        return

    regime = snapshot.get("market_regime", {})
    weights = snapshot.get("factor_weights", {})
    scores = pd.DataFrame(snapshot.get("scores", []))
    theses = pd.DataFrame(snapshot.get("theses", []))
    valuations = pd.DataFrame(snapshot.get("valuations", []))
    sizing = pd.DataFrame(snapshot.get("sizing", []))
    prediction_summary = snapshot.get("prediction_summary", {})
    sentiment = pd.DataFrame(snapshot.get("sentiment", []))
    behaviors = snapshot.get("behaviors", [])
    integrations = pd.DataFrame(snapshot.get("integrations", []))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("市场状态", regime.get("name", "NA"))
    col2.metric("状态置信度", percent_value(float(regime.get("confidence", 0.0))))
    col3.metric("预测命中率", percent_value(float(prediction_summary.get("accuracy", 0.0))))
    col4.metric("已评估预测数", str(int(prediction_summary.get("evaluated", 0))))

    st.subheader("动态因子权重")
    weight_cols = ["growth", "quality", "momentum", "value", "risk"]
    weight_values = [float(weights.get(column, 0.0)) for column in weight_cols]
    fig = go.Figure(data=[go.Bar(x=[item.title() for item in weight_cols], y=weight_values)])
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=25, b=20), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption(weights.get("reason", ""))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("股票综合排名")
        if scores.empty:
            render_missing_results()
        else:
            shown = scores[["ticker", "total_score", "factor_scores", "explanation"]].copy()
            st.dataframe(shown, width="stretch", hide_index=True)
    with right:
        st.subheader("市场驱动因素")
        for driver in regime.get("drivers", []):
            st.write(f"- {driver}")
        cautions = regime.get("cautions", [])
        if cautions:
            st.subheader("风险提示")
            for caution in cautions:
                st.write(f"- {caution}")

    st.subheader("投资假设跟踪")
    if theses.empty:
        render_missing_results()
    else:
        st.dataframe(theses[["ticker", "thesis", "status", "triggered_conditions"]], width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("情景估值")
        if valuations.empty:
            render_missing_results()
        else:
            st.dataframe(valuations[["ticker", "weighted_fair_value", "upside_to_price"]], width="stretch", hide_index=True)
    with col_right:
        st.subheader("仓位建议")
        if sizing.empty:
            render_missing_results()
        else:
            st.dataframe(sizing[["ticker", "current_allocation", "max_allocation", "suggested_action", "reasons"]], width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("市场情绪")
        if sentiment.empty:
            render_missing_results()
        else:
            st.dataframe(sentiment[["ticker", "label", "speculation_risk", "reasons"]], width="stretch", hide_index=True)
    with col_right:
        st.subheader("个人交易行为提示")
        if behaviors:
            for item in behaviors:
                with st.container(border=True):
                    st.write(f"**{item.get('label', '')}**")
                    st.caption(item.get("severity", ""))
                    st.write(item.get("evidence", ""))
                    st.write(item.get("reminder", ""))
        else:
            st.write("暂未触发明显行为偏差提示。")

    st.subheader("开源引擎状态")
    if integrations.empty:
        render_missing_results()
    else:
        st.dataframe(integrations[["name", "role", "available"]], width="stretch", hide_index=True)

    report_path = latest_investment_report_path(snapshot)
    report = load_latest_investment_report(file_signature(report_path))
    if report:
        with st.expander("最新研究日报", expanded=False):
            st.markdown(report)


def render_stock_explorer(tickers: list[str]) -> None:
    render_page_title(
        "Single Name Explorer",
        "股票浏览器",
        "查看项目股票池或任意 Yahoo Finance 股票代码的价格、技术指标、收益率与风险特征。",
    )
    features = load_features()
    config = load_dashboard_config()
    data_config = config.get("data", {})
    start = data_config.get("start", "2015-01-01")
    end = data_config.get("end", "2026-12-31")

    if features.empty:
        render_missing_results()
        return

    col1, col2, col3 = st.columns([1, 2, 0.6])
    selected = col1.selectbox("项目股票池", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    external = normalize_ticker(
        col2.text_input(
            "外部股票查询",
            placeholder="输入任意 Yahoo Finance 代码，例如 PLTR、TSM、BABA、0700.HK",
        )
    )
    refresh_external = col3.button("刷新", disabled=not bool(external))
    if refresh_external:
        load_external_ticker_features.clear()

    ticker = external or selected
    use_external = bool(external and external not in set(tickers))

    if use_external:
        try:
            with st.spinner(f"正在从 yfinance 下载 {ticker} 并计算指标..."):
                stock = load_external_ticker_features(ticker, start=start, end=end)
        except Exception as exc:
            st.error(f"无法下载 {ticker}：{exc}")
            st.caption("Yahoo Finance 偶尔会限流或中断连接。稍等几十秒后点击「刷新」。")
            return
        source_label = "实时 yfinance 查询"
    else:
        stock = features[features["ticker"].eq(ticker)].copy()
        source_label = "项目已处理数据集"

    if stock.empty:
        st.warning(f"没有找到 {ticker} 的市场数据。请检查股票代码后重试。")
        return
    st.caption(f"数据来源：{source_label}。时间范围：{stock['date'].min().date()} 至 {stock['date'].max().date()}。")

    latest = stock.dropna(subset=["adjusted_close"]).iloc[-1]
    returns = pd.to_numeric(stock["daily_return"], errors="coerce").dropna()
    max_dd, _ = maximum_drawdown(returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新复权收盘价", f"{latest['adjusted_close']:.2f}")
    col2.metric("21 日收益", percent_value(latest.get("return_21d")))
    col3.metric("126 日收益", percent_value(latest.get("return_126d")))
    col4.metric("最大回撤", percent_value(max_dd))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock["date"], y=stock["adjusted_close"], mode="lines", name="复权收盘价"))
    if "sma_20" in stock:
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["sma_20"], mode="lines", name="SMA 20"))
    if "sma_60" in stock:
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["sma_60"], mode="lines", name="SMA 60"))
    fig.update_layout(title=f"{ticker} 价格与移动均线", height=440, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, width="stretch")

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
    st.dataframe(stock[["date", "ticker", *shown]].tail(30), width="stretch", hide_index=True)


def strategy_weights(features: pd.DataFrame, strategy: str, top_n: int, entry_z: float) -> pd.DataFrame:
    if strategy == "Momentum":
        return generate_momentum_weights(features, top_n=top_n)
    if strategy == "Mean Reversion":
        return generate_mean_reversion_weights(features, entry_z=entry_z)
    return generate_factor_weights(features, top_n=top_n)


def render_backtest_lab() -> None:
    render_page_title(
        "Strategy Backtesting",
        "回测实验室",
        "在已处理股票池上比较动量、均值回归与多因子策略，包含交易成本、滑点与信号滞后。",
    )
    features = load_features()
    if features.empty:
        render_missing_results()
        return

    col1, col2, col3 = st.columns(3)
    strategy_display = col1.selectbox("策略", ["动量", "均值回归", "多因子"])
    strategy = {"动量": "Momentum", "均值回归": "Mean Reversion", "多因子": "Factor"}[strategy_display]
    top_n = col2.slider("Top N", 3, 20, 10)
    entry_z = col3.select_slider("均值回归入场 Z-score", options=[-1.5, -2.0, -2.5], value=-2.0)

    cost_col1, cost_col2, cost_col3 = st.columns(3)
    capital = cost_col1.number_input("初始资金", min_value=10_000, max_value=1_000_000, value=100_000, step=10_000)
    transaction_cost = cost_col2.number_input("交易成本 bps", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    slippage = cost_col3.number_input("滑点 bps", min_value=0.0, max_value=50.0, value=2.0, step=1.0)

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
    col1.metric("累计收益", percent_value(summary.get("cumulative_return")))
    col2.metric("年化收益", percent_value(summary.get("annualized_return")))
    col3.metric("夏普比率", f"{summary.get('sharpe_ratio', np.nan):.3f}")
    col4.metric("最大回撤", percent_value(summary.get("maximum_drawdown")))

    st.plotly_chart(line_chart(result.portfolio_value, "date", "total_value", f"{strategy_display}策略回测"), width="stretch")
    st.dataframe(result.trades.tail(50), width="stretch", hide_index=True)


def render_risk_factors() -> None:
    render_page_title(
        "Risk Attribution",
        "风险与因子",
        "查看组合表现、Fama-French 因子暴露、参数敏感性和市场基准比较。",
    )
    performance = read_csv(file_signature(RESULTS_DIR / "performance_summary.csv"))
    exposure = read_csv(file_signature(RESULTS_DIR / "factor_exposure.csv"))
    sensitivity = read_csv(file_signature(RESULTS_DIR / "parameter_sensitivity.csv"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("表现摘要")
        if performance.empty:
            render_missing_results()
        else:
            st.dataframe(performance, width="stretch")
    with col2:
        st.subheader("因子暴露")
        if exposure.empty:
            render_missing_results()
        else:
            st.dataframe(exposure, width="stretch")

    st.subheader("参数敏感性")
    if sensitivity.empty:
        render_missing_results()
    else:
        st.dataframe(sensitivity, width="stretch", hide_index=True)


def render_data_ml() -> None:
    render_page_title(
        "Data Diagnostics",
        "数据与机器学习",
        "检查行情清洗质量，并展示收益率预测模型的时间序列验证结果。",
    )
    quality = read_csv(file_signature(RESULTS_DIR / "data_quality_report.csv"))
    ml = read_csv(file_signature(RESULTS_DIR / "ml_model_comparison.csv"))

    st.subheader("数据质量")
    if quality.empty:
        render_missing_results()
    else:
        st.dataframe(quality, width="stretch", hide_index=True)

    st.subheader("机器学习基线模型")
    if ml.empty:
        render_missing_results()
    else:
        st.dataframe(ml, width="stretch", hide_index=True)


def render_derivatives_lab() -> None:
    render_page_title(
        "Derivatives Pricing",
        "衍生品实验室",
        "使用 Black-Scholes、二叉树、Monte Carlo 和 Greeks 分析欧式期权，并查看 Delta 对冲实验。",
    )
    col1, col2, col3 = st.columns(3)
    spot = col1.number_input("标的价格", min_value=1.0, value=100.0, step=1.0)
    strike = col2.number_input("行权价", min_value=1.0, value=100.0, step=1.0)
    maturity = col3.number_input("到期年限", min_value=0.01, value=1.0, step=0.05)

    col4, col5, col6 = st.columns(3)
    rate = col4.number_input("无风险利率", min_value=0.0, max_value=0.25, value=0.04, step=0.005)
    volatility = col5.number_input("波动率", min_value=0.01, max_value=2.0, value=0.20, step=0.01)
    option_type = col6.selectbox("期权类型", ["call", "put"])

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
    mc_paths = st.slider("Monte Carlo 路径数", 1_000, 100_000, 20_000, step=1_000)
    mc = monte_carlo_option_price(contract, n_paths=int(mc_paths), seed=7, antithetic=True, control_variate=True)
    greeks = black_scholes_greeks(contract)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Black-Scholes", f"{bs_price:.4f}")
    col2.metric("Binomial Tree", f"{tree_price:.4f}")
    col3.metric("Monte Carlo", f"{mc.price:.4f}")
    col4.metric("MC 标准误", f"{mc.standard_error:.4f}")

    st.subheader("Greeks 风险敏感度")
    greek_frame = pd.DataFrame([greeks]).T.reset_index()
    greek_frame.columns = ["Greek", "Value"]
    st.dataframe(greek_frame, width="stretch", hide_index=True)

    pricing = read_csv(file_signature(RESULTS_DIR / "derivative_pricing_comparison.csv"))
    hedging = read_csv(file_signature(RESULTS_DIR / "delta_hedging_frequency.csv"))
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("已保存定价对比")
        if pricing.empty:
            render_missing_results()
        else:
            st.dataframe(pricing, width="stretch", hide_index=True)
    with col_right:
        st.subheader("已保存对冲频率实验")
        if hedging.empty:
            render_missing_results()
        else:
            st.dataframe(hedging, width="stretch", hide_index=True)


def main() -> None:
    tickers = load_config_tickers()
    st.sidebar.markdown("### 量化研究终端")
    st.sidebar.caption("Equity Strategies · Factor Models · Derivatives")

    query = st.text_input(
        "全局搜索",
        placeholder="输入股票、策略、指标或模型，例如 NVDA、Sharpe、Fama-French、Black-Scholes...",
    )
    render_search(query, tickers)

    pages = {
        "研究概览": render_overview,
        "股票浏览器": lambda: render_stock_explorer(tickers),
        "回测实验室": render_backtest_lab,
        "风险与因子": render_risk_factors,
        "数据与机器学习": render_data_ml,
        "衍生品实验室": render_derivatives_lab,
        "AI 投资研究台": render_ai_investment_platform,
    }
    selected = st.sidebar.radio("工作区", list(pages.keys()))
    pages[selected]()


if __name__ == "__main__":
    main()
