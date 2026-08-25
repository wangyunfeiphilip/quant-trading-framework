"""Streamlit dashboard for the quantitative research framework."""

from __future__ import annotations

import re
import json
from html import escape
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
from dashboard.technical_summary import generate_technical_summary
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
    page_title="Quant Research Terminal",
    page_icon="Q",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #121821;
        --muted: #657184;
        --panel: #ffffff;
        --panel-soft: #f7f9fb;
        --line: #dfe5ec;
        --line-strong: #c9d2de;
        --teal: #0b766d;
        --teal-dark: #075a54;
        --gold: #b88324;
        --blue: #315f8d;
        --red: #b33a3a;
        --sidebar: #10141b;
    }
    @keyframes liftIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background:
            linear-gradient(90deg, rgba(16, 24, 35, 0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(16, 24, 35, 0.03) 1px, transparent 1px),
            linear-gradient(180deg, #f8fafc 0%, #eef2f6 100%);
        background-size: 32px 32px, 32px 32px, auto;
        color: var(--ink);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #111720 0%, #0d1118 100%);
        border-right: 1px solid #242c36;
        box-shadow: 14px 0 30px rgba(15, 23, 32, 0.12);
    }
    [data-testid="stSidebar"] * { color: #f4f6f8; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 8px 10px;
        margin: 2px 0;
        transition: background 180ms ease, transform 180ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p {
        color: rgba(244, 246, 248, 0.68) !important;
    }
    h1, h2, h3 { letter-spacing: 0; }
    h1 { font-size: 2.35rem; margin-bottom: 0.35rem; }
    h2 { margin-top: 1.1rem; }
    .app-shell {
        border: 1px solid rgba(201, 210, 222, 0.9);
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(247, 249, 251, 0.96) 100%);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 18px 45px rgba(18, 24, 33, 0.08);
        animation: liftIn 320ms ease both;
    }
    .home-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(118, 255, 231, 0.22);
        border-radius: 14px;
        padding: 34px 34px 28px;
        margin: 0 0 18px;
        color: #f5fbff;
        background:
            linear-gradient(90deg, rgba(126, 255, 229, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, rgba(126, 255, 229, 0.06) 1px, transparent 1px),
            linear-gradient(135deg, #071014 0%, #0b141b 46%, #11120d 100%);
        background-size: 34px 34px, 34px 34px, auto;
        box-shadow:
            0 28px 80px rgba(7, 16, 20, 0.28),
            inset 0 0 0 1px rgba(255,255,255,0.04);
        animation: liftIn 420ms ease both;
    }
    .home-hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            repeating-linear-gradient(180deg, rgba(255,255,255,0.06) 0px, rgba(255,255,255,0.06) 1px, transparent 1px, transparent 7px),
            linear-gradient(120deg, transparent 0%, rgba(119, 255, 231, 0.10) 44%, transparent 68%);
        opacity: 0.34;
        pointer-events: none;
        animation: scanDrift 8s linear infinite;
    }
    @keyframes scanDrift {
        from { transform: translateY(-24px); }
        to { transform: translateY(24px); }
    }
    .hero-content {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.65fr);
        gap: 26px;
        align-items: stretch;
    }
    .hero-kicker {
        color: #7effe5;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        color: #ffffff;
        font-size: clamp(2.25rem, 6vw, 5.25rem);
        font-weight: 900;
        line-height: 0.95;
        max-width: 860px;
        letter-spacing: 0;
    }
    .hero-title span {
        color: #d6b162;
    }
    .hero-copy {
        color: rgba(245, 251, 255, 0.78);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 780px;
        margin-top: 18px;
    }
    .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 24px;
    }
    .hero-cta,
    .hero-secondary {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 11px 16px;
        font-size: 0.9rem;
        font-weight: 800;
        text-decoration: none !important;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .hero-cta {
        color: #05100f !important;
        background: #7effe5;
        box-shadow: 0 0 28px rgba(126, 255, 229, 0.36);
    }
    .hero-secondary {
        color: #f5fbff !important;
        border: 1px solid rgba(245, 251, 255, 0.22);
        background: rgba(255,255,255,0.06);
    }
    .hero-cta:hover,
    .hero-secondary:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(0,0,0,0.24), 0 0 26px rgba(126,255,229,0.22);
    }
    .hero-cta:active,
    .hero-secondary:active,
    .stButton button:active {
        transform: translateY(1px) scale(0.985);
        box-shadow: 0 0 0 7px rgba(126, 255, 229, 0.13);
    }
    .hero-panel {
        border: 1px solid rgba(245, 251, 255, 0.12);
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(12px);
    }
    .hero-terminal-line {
        display: grid;
        grid-template-columns: 92px 1fr;
        gap: 10px;
        padding: 11px 0;
        border-bottom: 1px solid rgba(245, 251, 255, 0.10);
        color: rgba(245, 251, 255, 0.76);
        font-size: 0.86rem;
    }
    .hero-terminal-line:last-child { border-bottom: 0; }
    .hero-terminal-line strong {
        color: #7effe5;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.75rem;
    }
    .hero-metrics {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-top: 26px;
    }
    .hero-metric {
        border: 1px solid rgba(245, 251, 255, 0.13);
        background: rgba(255, 255, 255, 0.055);
        border-radius: 10px;
        padding: 13px 14px;
        min-height: 86px;
    }
    .hero-metric .value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 880;
    }
    .hero-metric .label {
        color: rgba(245,251,255,0.62);
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 5px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 20px;
    }
    .feature-card {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        background:
            linear-gradient(180deg, #ffffff 0%, #f7f9fb 100%);
        border-radius: 10px;
        padding: 18px;
        min-height: 154px;
        box-shadow: 0 14px 34px rgba(18, 24, 33, 0.07);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    .feature-card:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #0b766d, #d6b162);
    }
    .feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(11, 118, 109, 0.28);
        box-shadow: 0 22px 44px rgba(18, 24, 33, 0.10);
    }
    .feature-card:active {
        transform: translateY(1px) scale(0.99);
        box-shadow: 0 0 0 6px rgba(11, 118, 109, 0.10);
    }
    .feature-index {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }
    .feature-title {
        color: var(--ink);
        font-size: 1.04rem;
        font-weight: 820;
        margin-bottom: 8px;
    }
    .feature-copy {
        color: var(--muted);
        line-height: 1.58;
        font-size: 0.92rem;
    }
    .app-shell-top {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        flex-wrap: wrap;
    }
    .brand-kicker,
    .terminal-title .eyebrow {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .brand-title {
        color: var(--ink);
        font-size: 1.75rem;
        font-weight: 780;
        line-height: 1.1;
        margin-top: 6px;
    }
    .brand-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 8px;
        max-width: 760px;
    }
    .system-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
        min-width: 260px;
    }
    .chip {
        border: 1px solid #d8e0e8;
        background: #ffffff;
        color: #354155;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 0.78rem;
        font-weight: 700;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .chip-teal { color: var(--teal-dark); border-color: rgba(11, 118, 109, 0.26); background: rgba(11, 118, 109, 0.08); }
    .chip-gold { color: #7c5619; border-color: rgba(184, 131, 36, 0.32); background: rgba(184, 131, 36, 0.09); }
    .sidebar-brand {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 14px 14px 13px;
        margin: 2px 0 18px;
        background: rgba(255, 255, 255, 0.05);
    }
    .sidebar-brand .mark {
        color: #8dd5cd;
        font-weight: 850;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        margin-bottom: 6px;
    }
    .sidebar-brand .name {
        color: #ffffff;
        font-weight: 780;
        line-height: 1.2;
    }
    .sidebar-brand .desc {
        color: rgba(244, 246, 248, 0.62);
        font-size: 0.78rem;
        margin-top: 7px;
        line-height: 1.45;
    }
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div {
        border-radius: 8px !important;
        border-color: var(--line) !important;
        background: rgba(255, 255, 255, 0.94) !important;
        min-height: 48px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: rgba(11, 118, 109, 0.65) !important;
        box-shadow: 0 0 0 3px rgba(11, 118, 109, 0.12) !important;
    }
    .stButton button {
        min-height: 48px;
        border-radius: 8px;
        border: 1px solid var(--line-strong);
        background: #ffffff;
        color: var(--ink);
        font-weight: 750;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        border-color: rgba(11, 118, 109, 0.45);
        box-shadow: 0 8px 22px rgba(18, 24, 33, 0.09);
    }
    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, #ffffff 0%, #fbfcfd 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 15px 16px;
        box-shadow: 0 12px 30px rgba(18, 24, 33, 0.06);
        animation: liftIn 280ms ease both;
    }
    div[data-testid="stMetric"] label { color: var(--muted); font-weight: 700; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(18, 24, 33, 0.05);
    }
    .terminal-title {
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.96) 100%);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px rgba(18, 24, 33, 0.06);
        animation: liftIn 300ms ease both;
    }
    .terminal-title .title {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 820;
        line-height: 1.2;
        margin-top: 5px;
    }
    .terminal-title .subtitle {
        color: var(--muted);
        margin-top: 6px;
        font-size: 0.98rem;
    }
    .signal-panel {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #ffffff;
        padding: 17px 18px;
        margin: 12px 0 18px;
        box-shadow: 0 14px 34px rgba(18, 24, 33, 0.07);
        animation: liftIn 340ms ease both;
    }
    .signal-panel.bullish { border-left: 4px solid var(--teal); }
    .signal-panel.neutral { border-left: 4px solid var(--gold); }
    .signal-panel.bearish { border-left: 4px solid var(--red); }
    .signal-head {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 8px;
        flex-wrap: wrap;
    }
    .signal-badge {
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.8rem;
        font-weight: 820;
        border: 1px solid var(--line);
    }
    .bullish .signal-badge { color: var(--teal-dark); background: rgba(11,118,109,0.09); border-color: rgba(11,118,109,0.26); }
    .neutral .signal-badge { color: #7c5619; background: rgba(184,131,36,0.10); border-color: rgba(184,131,36,0.28); }
    .bearish .signal-badge { color: var(--red); background: rgba(179,58,58,0.09); border-color: rgba(179,58,58,0.25); }
    .signal-headline { color: var(--ink); font-weight: 760; }
    .signal-panel ul {
        margin: 8px 0 0 1.1rem;
        padding: 0;
        color: #394457;
        line-height: 1.68;
    }
    .signal-note {
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 10px;
    }
    .small-note { color: var(--muted); font-size: 0.9rem; }
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px;
        background: #ffffff;
        box-shadow: 0 14px 34px rgba(18, 24, 33, 0.06);
    }
    .stAlert {
        border-radius: 8px;
    }
    @media (max-width: 760px) {
        .brand-title { font-size: 1.35rem; }
        .system-chips { justify-content: flex-start; min-width: auto; }
        .app-shell { padding: 16px; }
        .terminal-title { padding: 17px; }
        .hero-content { grid-template-columns: 1fr; }
        .hero-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .feature-grid { grid-template-columns: 1fr; }
        .home-hero { padding: 24px 20px; }
    }
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
    ticker = re.sub(r"[^A-Za-z0-9.^=-]", "", value).upper()
    if re.fullmatch(r"\d{6}", ticker):
        if ticker.startswith(("000", "001", "002", "003", "200", "300", "301")):
            return f"{ticker}.SZ"
        if ticker.startswith(("600", "601", "603", "605", "688", "689", "900")):
            return f"{ticker}.SS"
    return ticker


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


def style_figure(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#121821"), x=0.02),
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=18, r=18, t=58, b=18),
        font=dict(color="#354155", family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.72)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(101, 113, 132, 0.13)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(101, 113, 132, 0.13)", zeroline=False)
    return fig


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="lines",
            name=y,
            line=dict(color="#0b766d", width=2.4),
        )
    )
    return style_figure(fig, title=title, height=420)


def render_missing_results() -> None:
    st.info("No generated research outputs were found. Run `python main.py`, or use live ticker lookup and the derivatives lab.")


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


def render_app_header(ticker_count: int) -> None:
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-shell-top">
                <div>
                    <div class="brand-kicker">Quantitative Research Framework</div>
                    <div class="brand-title">Equity Strategies and Derivatives Pricing</div>
                    <div class="brand-subtitle">
                        An interactive research terminal for equity strategy backtesting, risk attribution,
                        Fama-French factor modeling, machine learning baselines, and derivatives pricing.
                    </div>
                </div>
                <div class="system-chips">
                    <span class="chip chip-teal">{ticker_count} research names</span>
                    <span class="chip chip-gold">Signal-lagged backtest</span>
                    <span class="chip">Streamlit Demo</span>
                    <span class="chip">Python / C++</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_hero(ticker_count: int) -> None:
    st.markdown(
        f"""
        <section class="home-hero">
            <div class="hero-content">
                <div>
                    <div class="hero-kicker">Research-grade quant lab</div>
                    <div class="hero-title">Explore markets through <span>models, risk, and motion.</span></div>
                    <div class="hero-copy">
                        A public-facing quantitative research terminal that turns raw market data into
                        strategy backtests, technical signal explanations, factor exposures, machine-learning
                        diagnostics, and option-pricing experiments.
                    </div>
                    <div class="hero-actions">
                        <a class="hero-cta" href="#stock-explorer">Open the research terminal</a>
                        <a class="hero-secondary" href="#method-stack">View capability map</a>
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-terminal-line"><strong>Input</strong><span>OHLCV, adjusted prices, technical features, factor data</span></div>
                    <div class="hero-terminal-line"><strong>Engine</strong><span>Signal-lagged portfolio simulation with costs and slippage</span></div>
                    <div class="hero-terminal-line"><strong>Models</strong><span>Fama-French OLS/HAC, ML baselines, Black-Scholes and Monte Carlo</span></div>
                    <div class="hero-terminal-line"><strong>Output</strong><span>Returns, drawdowns, factor exposure, Greeks, hedge error</span></div>
                </div>
            </div>
            <div class="hero-metrics">
                <div class="hero-metric"><div class="value">{ticker_count}</div><div class="label">research universe names in the default equity dataset</div></div>
                <div class="hero-metric"><div class="value">3</div><div class="label">strategy families: momentum, mean reversion, factor investing</div></div>
                <div class="hero-metric"><div class="value">5</div><div class="label">option Greeks with analytical and numerical pricing checks</div></div>
                <div class="hero-metric"><div class="value">HAC</div><div class="label">Newey-West robust factor regression standard errors</div></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_capability_grid() -> None:
    st.markdown(
        """
        <div id="method-stack" class="feature-grid">
            <div class="feature-card">
                <div class="feature-index">01 / data</div>
                <div class="feature-title">Market Data Pipeline</div>
                <div class="feature-copy">Download, clean, adjust, and feature-engineer public equity data for repeatable research workflows.</div>
            </div>
            <div class="feature-card">
                <div class="feature-index">02 / strategies</div>
                <div class="feature-title">Backtesting Workbench</div>
                <div class="feature-copy">Compare momentum, mean-reversion, and factor portfolios with transaction costs, slippage, and signal lag.</div>
            </div>
            <div class="feature-card">
                <div class="feature-index">03 / risk</div>
                <div class="feature-title">Risk Attribution</div>
                <div class="feature-copy">Inspect Sharpe, drawdown, beta, alpha, tracking error, factor exposures, and parameter sensitivity.</div>
            </div>
            <div class="feature-card">
                <div class="feature-index">04 / intelligence</div>
                <div class="feature-title">Signal Explanations</div>
                <div class="feature-copy">Translate RSI, MACD, moving averages, Z-scores, and momentum windows into readable research conclusions.</div>
            </div>
            <div class="feature-card">
                <div class="feature-index">05 / derivatives</div>
                <div class="feature-title">Derivatives Lab</div>
                <div class="feature-copy">Price European options with Black-Scholes, binomial trees, and Monte Carlo variance reduction, then inspect Greeks.</div>
            </div>
            <div class="feature-card">
                <div class="feature-index">06 / engineering</div>
                <div class="feature-title">Hybrid Python/C++ Design</div>
                <div class="feature-copy">Keep research, statistics, and visualization in Python while reserving C++ components for simulation experiments.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(summary) -> None:
    tone = {"Bullish": "bullish", "Bearish": "bearish"}.get(summary.stance, "neutral")
    bullets = "".join(f"<li>{escape(item)}</li>" for item in summary.bullets)
    st.markdown(
        f"""
        <div class="signal-panel {tone}">
            <div class="signal-head">
                <span class="signal-badge">Composite stance: {escape(summary.stance)}</span>
                <span class="signal-headline">{escape(summary.headline)}</span>
            </div>
            <ul>{bullets}</ul>
            <div class="signal-note">Research interpretation based on the latest technical indicators. Not investment advice.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search(query: str, tickers: list[str]) -> None:
    if not query.strip():
        return
    st.subheader("Search Results")
    results = search_catalog(query, build_search_index(tickers))
    if not results:
        candidate = normalize_ticker(query)
        if 1 <= len(candidate) <= 12:
            st.write(
                f"No internal catalog item matched. To inspect `{candidate}`, open Stock Explorer and use live ticker lookup."
            )
        else:
            st.write("No matching results.")
        return
    for item in results:
        with st.container(border=True):
            st.caption(item.category)
            st.write(f"**{item.title}**")
            st.write(item.description)
            st.caption(f"Open: {item.target}")


def render_overview() -> None:
    tickers = load_config_tickers()
    render_home_hero(len(tickers))
    render_page_title(
        "Research Overview",
        "What This Terminal Does",
        "A portfolio-grade research interface for equities, factor models, derivatives, and risk attribution.",
    )
    render_capability_grid()

    portfolio = load_portfolio_value()
    performance = read_csv(file_signature(RESULTS_DIR / "performance_summary.csv"))

    if not performance.empty:
        summary = performance.iloc[:, 0] if performance.shape[1] == 1 else performance.set_index(performance.columns[0]).iloc[:, 0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cumulative Return", metric_value(summary, "cumulative_return"))
        col2.metric("Sharpe Ratio", metric_value(summary, "sharpe_ratio"))
        col3.metric("Maximum Drawdown", metric_value(summary, "maximum_drawdown"))
        col4.metric("Beta vs SPY", metric_value(summary, "beta_vs_benchmark"))
    else:
        render_missing_results()

    if not portfolio.empty and "total_value" in portfolio:
        st.plotly_chart(line_chart(portfolio, "date", "total_value", "Portfolio Equity Curve"), width="stretch")

    findings = load_key_findings()
    if findings:
        st.markdown(findings)


def render_ai_investment_platform() -> None:
    render_page_title(
        "AI Equity Research",
        "AI Investment Research Desk",
        "Single-name research, thesis tracking, factor scoring, valuation, position risk, and behavior feedback.",
    )

    snapshot_path = latest_investment_snapshot_path()
    snapshot = load_investment_snapshot(file_signature(snapshot_path))
    if not snapshot:
        st.info("No AI investment research snapshot was found. Run `python3 scripts/run_investment_platform.py --config investment_platform.json` first.")
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
    col1.metric("Market Regime", regime.get("name", "NA"))
    col2.metric("Regime Confidence", percent_value(float(regime.get("confidence", 0.0))))
    col3.metric("Prediction Accuracy", percent_value(float(prediction_summary.get("accuracy", 0.0))))
    col4.metric("Evaluated Forecasts", str(int(prediction_summary.get("evaluated", 0))))

    st.subheader("Dynamic Factor Weights")
    weight_cols = ["growth", "quality", "momentum", "value", "risk"]
    weight_values = [float(weights.get(column, 0.0)) for column in weight_cols]
    fig = go.Figure(data=[go.Bar(x=[item.title() for item in weight_cols], y=weight_values)])
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=25, b=20), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption(weights.get("reason", ""))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Composite Stock Ranking")
        if scores.empty:
            render_missing_results()
        else:
            shown = scores[["ticker", "total_score", "factor_scores", "explanation"]].copy()
            st.dataframe(shown, width="stretch", hide_index=True)
    with right:
        st.subheader("Market Drivers")
        for driver in regime.get("drivers", []):
            st.write(f"- {driver}")
        cautions = regime.get("cautions", [])
        if cautions:
            st.subheader("Risk Alerts")
            for caution in cautions:
                st.write(f"- {caution}")

    st.subheader("Investment Thesis Tracking")
    if theses.empty:
        render_missing_results()
    else:
        st.dataframe(theses[["ticker", "thesis", "status", "triggered_conditions"]], width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Scenario Valuation")
        if valuations.empty:
            render_missing_results()
        else:
            st.dataframe(valuations[["ticker", "weighted_fair_value", "upside_to_price"]], width="stretch", hide_index=True)
    with col_right:
        st.subheader("Position Suggestions")
        if sizing.empty:
            render_missing_results()
        else:
            st.dataframe(sizing[["ticker", "current_allocation", "max_allocation", "suggested_action", "reasons"]], width="stretch", hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Market Sentiment")
        if sentiment.empty:
            render_missing_results()
        else:
            st.dataframe(sentiment[["ticker", "label", "speculation_risk", "reasons"]], width="stretch", hide_index=True)
    with col_right:
        st.subheader("Personal Trading Behavior Notes")
        if behaviors:
            for item in behaviors:
                with st.container(border=True):
                    st.write(f"**{item.get('label', '')}**")
                    st.caption(item.get("severity", ""))
                    st.write(item.get("evidence", ""))
                    st.write(item.get("reminder", ""))
        else:
            st.write("No clear behavioral bias alerts were triggered.")

    st.subheader("Open-Source Engine Status")
    if integrations.empty:
        render_missing_results()
    else:
        st.dataframe(integrations[["name", "role", "available"]], width="stretch", hide_index=True)

    report_path = latest_investment_report_path(snapshot)
    report = load_latest_investment_report(file_signature(report_path))
    if report:
        with st.expander("Latest Research Report", expanded=False):
            st.markdown(report)


def render_stock_explorer(tickers: list[str]) -> None:
    render_page_title(
        "Single Name Explorer",
        "Stock Explorer",
        "Inspect prices, technical indicators, return windows, and risk characteristics for project tickers or any Yahoo Finance symbol.",
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
    selected = col1.selectbox("Project Universe", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    external = normalize_ticker(
        col2.text_input(
            "Live Ticker Lookup",
            placeholder="Enter a Yahoo Finance symbol, e.g. PLTR, TSM, BABA, 0700.HK, 301321",
        )
    )
    refresh_external = col3.button("Refresh", disabled=not bool(external))
    if refresh_external:
        load_external_ticker_features.clear()

    ticker = external or selected
    use_external = bool(external and external not in set(tickers))

    if use_external:
        try:
            with st.spinner(f"Downloading {ticker} from yfinance and computing indicators..."):
                stock = load_external_ticker_features(ticker, start=start, end=end)
        except Exception as exc:
            st.error(f"Unable to download {ticker}: {exc}")
            st.caption(
                "Yahoo Finance can rate-limit or interrupt requests. Mainland China A-share codes are automatically "
                "expanded with `.SZ` or `.SS`; Hong Kong tickers should still use full symbols such as `0700.HK`."
            )
            return
        source_label = "Live yfinance lookup"
    else:
        stock = features[features["ticker"].eq(ticker)].copy()
        source_label = "Processed project dataset"

    if stock.empty:
        st.warning(f"No market data was found for {ticker}. Check the ticker symbol and try again.")
        return
    st.caption(f"Data source: {source_label}. Date range: {stock['date'].min().date()} to {stock['date'].max().date()}.")

    latest = stock.dropna(subset=["adjusted_close"]).iloc[-1]
    returns = pd.to_numeric(stock["daily_return"], errors="coerce").dropna()
    max_dd, _ = maximum_drawdown(returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Adjusted Close", f"{latest['adjusted_close']:.2f}")
    col2.metric("21D Return", percent_value(latest.get("return_21d")))
    col3.metric("126D Return", percent_value(latest.get("return_126d")))
    col4.metric("Maximum Drawdown", percent_value(max_dd))

    summary = generate_technical_summary(latest)
    st.subheader("Technical Signal Interpretation")
    render_signal_card(summary)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stock["date"],
            y=stock["adjusted_close"],
            mode="lines",
            name="Adjusted Close",
            line=dict(color="#121821", width=2.2),
        )
    )
    if "sma_20" in stock:
        fig.add_trace(
            go.Scatter(
                x=stock["date"],
                y=stock["sma_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="#0b766d", width=1.8),
            )
        )
    if "sma_60" in stock:
        fig.add_trace(
            go.Scatter(
                x=stock["date"],
                y=stock["sma_60"],
                mode="lines",
                name="SMA 60",
                line=dict(color="#b88324", width=1.8),
            )
        )
    fig = style_figure(fig, title=f"{ticker} Price and Moving Averages", height=440)
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
        "Backtest Lab",
        "Compare momentum, mean-reversion, and multi-factor strategies with transaction costs, slippage, and signal lag.",
    )
    features = load_features()
    if features.empty:
        render_missing_results()
        return

    col1, col2, col3 = st.columns(3)
    strategy_display = col1.selectbox("Strategy", ["Momentum", "Mean Reversion", "Multi-Factor"])
    strategy = {"Momentum": "Momentum", "Mean Reversion": "Mean Reversion", "Multi-Factor": "Factor"}[strategy_display]
    top_n = col2.slider("Top N", 3, 20, 10)
    entry_z = col3.select_slider("Mean-Reversion Entry Z-score", options=[-1.5, -2.0, -2.5], value=-2.0)

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
    col2.metric("Annualized Return", percent_value(summary.get("annualized_return")))
    col3.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', np.nan):.3f}")
    col4.metric("Maximum Drawdown", percent_value(summary.get("maximum_drawdown")))

    st.plotly_chart(line_chart(result.portfolio_value, "date", "total_value", f"{strategy_display} Strategy Backtest"), width="stretch")
    st.dataframe(result.trades.tail(50), width="stretch", hide_index=True)


def render_risk_factors() -> None:
    render_page_title(
        "Risk Attribution",
        "Risk & Factors",
        "Inspect portfolio performance, Fama-French exposures, parameter sensitivity, and benchmark comparison.",
    )
    performance = read_csv(file_signature(RESULTS_DIR / "performance_summary.csv"))
    exposure = read_csv(file_signature(RESULTS_DIR / "factor_exposure.csv"))
    sensitivity = read_csv(file_signature(RESULTS_DIR / "parameter_sensitivity.csv"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Summary")
        if performance.empty:
            render_missing_results()
        else:
            st.dataframe(performance, width="stretch")
    with col2:
        st.subheader("Factor Exposure")
        if exposure.empty:
            render_missing_results()
        else:
            st.dataframe(exposure, width="stretch")

    st.subheader("Parameter Sensitivity")
    if sensitivity.empty:
        render_missing_results()
    else:
        st.dataframe(sensitivity, width="stretch", hide_index=True)


def render_data_ml() -> None:
    render_page_title(
        "Data Diagnostics",
        "Data & Machine Learning",
        "Audit market-data cleaning quality and inspect chronological validation for return-prediction baselines.",
    )
    quality = read_csv(file_signature(RESULTS_DIR / "data_quality_report.csv"))
    ml = read_csv(file_signature(RESULTS_DIR / "ml_model_comparison.csv"))

    st.subheader("Data Quality")
    if quality.empty:
        render_missing_results()
    else:
        st.dataframe(quality, width="stretch", hide_index=True)

    st.subheader("Machine Learning Baselines")
    if ml.empty:
        render_missing_results()
    else:
        st.dataframe(ml, width="stretch", hide_index=True)


def render_derivatives_lab() -> None:
    render_page_title(
        "Derivatives Pricing",
        "Derivatives Lab",
        "Analyze European options with Black-Scholes, binomial trees, Monte Carlo, Greeks, and delta-hedging experiments.",
    )
    col1, col2, col3 = st.columns(3)
    spot = col1.number_input("Spot Price", min_value=1.0, value=100.0, step=1.0)
    strike = col2.number_input("Strike Price", min_value=1.0, value=100.0, step=1.0)
    maturity = col3.number_input("Years to Maturity", min_value=0.01, value=1.0, step=0.05)

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
    col4.metric("MC Standard Error", f"{mc.standard_error:.4f}")

    st.subheader("Greeks Risk Sensitivities")
    greek_frame = pd.DataFrame([greeks]).T.reset_index()
    greek_frame.columns = ["Greek", "Value"]
    st.dataframe(greek_frame, width="stretch", hide_index=True)

    pricing = read_csv(file_signature(RESULTS_DIR / "derivative_pricing_comparison.csv"))
    hedging = read_csv(file_signature(RESULTS_DIR / "delta_hedging_frequency.csv"))
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Saved Pricing Comparison")
        if pricing.empty:
            render_missing_results()
        else:
            st.dataframe(pricing, width="stretch", hide_index=True)
    with col_right:
        st.subheader("Saved Hedge-Frequency Experiment")
        if hedging.empty:
            render_missing_results()
        else:
            st.dataframe(hedging, width="stretch", hide_index=True)


def main() -> None:
    tickers = load_config_tickers()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="mark">QRF</div>
            <div class="name">Quant Research Terminal</div>
            <div class="desc">Equity strategies, factor models, ML baselines, derivatives pricing.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_app_header(len(tickers))

    query = st.text_input(
        "Global Search",
        placeholder="Search tickers, strategies, metrics, or models, e.g. NVDA, Sharpe, Fama-French, Black-Scholes...",
        label_visibility="collapsed",
    )
    render_search(query, tickers)

    pages = {
        "Home": render_overview,
        "Stock Explorer": lambda: render_stock_explorer(tickers),
        "Backtest Lab": render_backtest_lab,
        "Risk & Factors": render_risk_factors,
        "Data & ML": render_data_ml,
        "Derivatives Lab": render_derivatives_lab,
        "AI Research Desk": render_ai_investment_platform,
    }
    selected = st.sidebar.radio("Workspace", list(pages.keys()))
    pages[selected]()


if __name__ == "__main__":
    main()
