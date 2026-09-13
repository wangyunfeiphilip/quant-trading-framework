"""Streamlit dashboard for the quantitative research framework."""

from __future__ import annotations

import re
import json
import os
import shutil
import subprocess
import tempfile
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
    MarketDataConfig,
    build_market_dataset,
    clean_price_data,
    create_feature_dataset,
    download_price_data,
    load_fundamental_features,
)
from derivatives.black_scholes import OptionContract, black_scholes_greeks, black_scholes_price
from derivatives.numerical_methods import binomial_option_price, monte_carlo_option_price
from investment_platform.trades import (
    REQUIRED_COLUMNS as TRADE_REQUIRED_COLUMNS,
    latest_trade_file,
    normalize_trade_frame,
    private_trade_files,
    read_trade_file,
    reconstruct_from_trades,
    write_uploaded_trades,
)
from risk.risk_metrics import maximum_drawdown, performance_summary
from strategies.factor_strategy import generate_factor_weights
from strategies.mean_reversion import generate_mean_reversion_weights
from strategies.momentum import generate_momentum_weights

try:
    from data.data_loader import (
        expected_latest_market_date,
        latest_market_date,
        market_dataset_is_stale,
        next_yfinance_end_date,
    )
except ImportError:  # pragma: no cover - defensive path for stale cloud builds
    def expected_latest_market_date(today: str | pd.Timestamp | None = None) -> pd.Timestamp:
        reference = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
        if reference.weekday() == 5:
            return reference - pd.Timedelta(days=1)
        if reference.weekday() == 6:
            return reference - pd.Timedelta(days=2)
        return reference

    def next_yfinance_end_date(today: str | pd.Timestamp | None = None) -> str:
        return (expected_latest_market_date(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def latest_market_date(data: pd.DataFrame) -> pd.Timestamp | None:
        if data.empty or "date" not in data.columns:
            return None
        dates = pd.to_datetime(data["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max().normalize()

    def market_dataset_is_stale(data: pd.DataFrame, today: str | pd.Timestamp | None = None) -> bool:
        latest = latest_market_date(data)
        if latest is None:
            return True
        return latest < expected_latest_market_date(today)

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INVESTMENT_RESULTS_DIR = RESULTS_DIR / "investment_platform"
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"
AUTO_REFRESH_DATA = os.getenv("QTF_AUTO_REFRESH_DATA", "1").strip().lower() not in {"0", "false", "no"}


st.set_page_config(
    page_title="Quant Research Terminal",
    page_icon="Q",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f8f6ef;
        --body: #1b1713;
        --muted: #9d9b95;
        --panel: #24211d;
        --panel-soft: #2a2722;
        --line: rgba(255, 255, 255, 0.10);
        --line-strong: rgba(255, 255, 255, 0.18);
        --teal: #b7f8df;
        --teal-hot: #d6fff1;
        --teal-dark: #78d5be;
        --gold: #f3d687;
        --blue: #9fb7ff;
        --red: #b33a3a;
        --night: #171410;
        --night-2: #211e19;
        --cream: #f7f2e8;
        --sidebar: #10141b;
    }
    @keyframes liftIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes gridShift {
        from { background-position: 0 0, 0 0, 0 0, 0 0; }
        to { background-position: 42px 0, 0 42px, 0 0, 0 0; }
    }
    @keyframes signalSweep {
        from { transform: translateX(-58%); opacity: 0.12; }
        45% { opacity: 0.32; }
        to { transform: translateX(92%); opacity: 0.08; }
    }
    .stApp {
        background:
            radial-gradient(circle at 50% -12%, rgba(247, 242, 232, 0.09), transparent 24%),
            radial-gradient(circle at 12% 18%, rgba(183, 248, 223, 0.08), transparent 22%),
            linear-gradient(180deg, #1b1713 0%, #171410 58%, #15120f 100%);
        background-size: auto;
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        background: #1b1713 !important;
        border-bottom: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    div[data-testid="stToolbar"],
    div[data-testid="stToolbar"] > div {
        background: transparent !important;
    }
    div[data-testid="stToolbar"] button,
    div[data-testid="stToolbar"] svg,
    div[data-testid="stToolbar"] [role="button"] {
        color: rgba(247, 242, 232, 0.72) !important;
        fill: rgba(247, 242, 232, 0.72) !important;
    }
    div[data-testid="stToolbar"] button:hover,
    div[data-testid="stToolbar"] [role="button"]:hover {
        background: rgba(255, 255, 255, 0.07) !important;
    }
    .block-container {
        max-width: 1580px;
        padding-top: 1.35rem;
        padding-bottom: 3.5rem;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stSidebar"] {
        display: block !important;
        background:
            radial-gradient(circle at 80% 0%, rgba(183, 248, 223, 0.12), transparent 26%),
            linear-gradient(180deg, #111720 0%, #0d1118 100%);
        border-right: 1px solid rgba(244, 246, 248, 0.12);
        box-shadow: 14px 0 30px rgba(15, 23, 32, 0.20);
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
    .sidebar-brand {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 14px 14px 12px;
        margin: 10px 0 18px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.045)),
            rgba(13, 17, 24, 0.72);
    }
    .sidebar-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: #7effe5;
        color: #06100f !important;
        font-weight: 920;
        margin-bottom: 10px;
    }
    .sidebar-name {
        color: #f7f2e8 !important;
        font-size: 1.02rem;
        font-weight: 880;
        line-height: 1.2;
    }
    .sidebar-desc {
        color: rgba(247,242,232,0.60) !important;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 6px;
    }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    h1 { font-size: 2.35rem; margin-bottom: 0.35rem; }
    h2 { margin-top: 1.1rem; }
    .app-shell {
        border: 1px solid rgba(201, 210, 222, 0.9);
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(247, 249, 251, 0.96) 100%);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 22px 58px rgba(18, 24, 33, 0.10);
        animation: liftIn 320ms ease both;
    }
    .home-hero {
        position: relative;
        overflow: hidden;
        border: 0;
        border-radius: 0;
        padding: 58px 24px 28px;
        margin: 0 auto 8px;
        min-height: auto;
        color: var(--cream);
        background:
            radial-gradient(circle at 50% 8%, rgba(255,255,255,0.055), transparent 35%);
        box-shadow: none;
        animation: liftIn 420ms ease both;
        text-align: center;
    }
    .home-hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, transparent, rgba(255,255,255,0.035), transparent);
        opacity: 0.22;
        pointer-events: none;
        animation: signalSweep 9s ease-in-out infinite;
    }
    .home-hero:after {
        display: none;
    }
    @keyframes scanDrift {
        from { transform: translateY(-24px); }
        to { transform: translateY(24px); }
    }
    .hero-content {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1fr;
        gap: 28px;
        align-items: center;
        justify-items: center;
    }
    .hero-system-row {
        display: none;
        position: relative;
        z-index: 1;
        display: none;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 30px;
    }
    .hero-system-row span {
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.055);
        color: rgba(247, 242, 232, 0.72);
        border-radius: 999px;
        padding: 9px 14px;
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        backdrop-filter: blur(12px);
    }
    .hero-kicker {
        display: inline-flex;
        color: rgba(247, 242, 232, 0.78);
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.075);
        border-radius: 999px;
        padding: 9px 18px;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 26px;
    }
    .hero-title {
        color: var(--cream);
        font-size: 3.85rem;
        font-weight: 920;
        line-height: 0.98;
        max-width: 980px;
        letter-spacing: 0;
        margin: 0 auto;
        text-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
    }
    .hero-title span {
        color: var(--cream);
    }
    .hero-copy {
        color: rgba(247, 242, 232, 0.60);
        font-size: 1.06rem;
        line-height: 1.64;
        max-width: 760px;
        margin: 24px auto 0;
    }
    .hero-query {
        display: none;
        margin: 24px auto 0;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.055);
        color: rgba(247, 242, 232, 0.74);
        border-radius: 999px;
        padding: 11px 17px;
        max-width: 760px;
        font-size: 0.86rem;
        font-weight: 720;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.025);
    }
    .hero-query strong {
        color: var(--teal);
        margin-right: 8px;
    }
    .hero-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 24px;
    }
    .hero-cta,
    .hero-secondary,
    .stButton button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
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
    .hero-secondary:hover,
    .stButton button:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.24);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.120), rgba(255,255,255,0.070));
        box-shadow: 0 18px 44px rgba(0,0,0,0.20);
    }
    .hero-cta:active,
    .hero-secondary:active,
    .stButton button:active {
        transform: translateY(1px) scale(0.985);
        box-shadow: 0 0 0 6px rgba(247, 242, 232, 0.08);
    }
    .hero-panel {
        display: none;
        width: min(980px, 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.038)),
            rgba(36, 33, 29, 0.62);
        border-radius: 8px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        box-shadow:
            0 30px 80px rgba(0, 0, 0, 0.22),
            inset 0 0 0 1px rgba(255,255,255,0.035);
        text-align: left;
    }
    .hero-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid rgba(245, 251, 255, 0.10);
        padding-bottom: 12px;
        margin-bottom: 4px;
    }
    .hero-panel-header .label {
        color: var(--cream);
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.78rem;
    }
    .hero-panel-header .status {
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 840;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .hero-terminal-line {
        display: grid;
        grid-template-columns: 98px 1fr;
        gap: 14px;
        padding: 9px 0;
        border-bottom: 1px solid rgba(245, 251, 255, 0.10);
        color: rgba(247, 242, 232, 0.66);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .hero-terminal-line:last-child { border-bottom: 0; }
    .hero-terminal-line strong {
        color: var(--teal);
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.75rem;
    }
    .hero-metrics {
        display: none !important;
        position: relative;
        z-index: 1;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 30px auto 0;
        max-width: 1180px;
    }
    .hero-metric {
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.035));
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 72px;
        backdrop-filter: blur(10px);
        text-align: left;
    }
    .hero-metric .value {
        color: var(--cream);
        font-size: 1.45rem;
        font-weight: 900;
    }
    .hero-metric .label {
        color: rgba(247,242,232,0.54);
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
        display: block;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.035)),
            var(--panel);
        border-radius: 8px;
        padding: 30px 30px 28px;
        min-height: 250px;
        box-shadow: none;
        text-decoration: none !important;
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }
    .feature-card * {
        text-decoration: none !important;
    }
    .feature-card:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        width: 56px;
        height: 56px;
        border-radius: 8px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.10);
        left: 30px;
        top: 30px;
    }
    .feature-card:after {
        content: "";
        position: absolute;
        left: 30px;
        top: 48px;
        width: 20px;
        height: 1px;
        background: rgba(247, 242, 232, 0.78);
        box-shadow: 0 8px 0 rgba(247, 242, 232, 0.54), 0 16px 0 rgba(247, 242, 232, 0.30);
        transition: transform 180ms ease, opacity 180ms ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.18);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.04)),
            #2a2722;
        box-shadow: 0 26px 70px rgba(0, 0, 0, 0.20);
    }
    .feature-card:hover:after {
        transform: translateX(4px);
        opacity: 0.95;
    }
    .feature-card:active {
        transform: translateY(1px) scale(0.99);
        box-shadow: 0 0 0 6px rgba(11, 118, 109, 0.10);
    }
    .feature-index {
        color: rgba(247, 242, 232, 0.50);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin: 80px 0 18px;
    }
    .feature-title {
        color: var(--cream);
        font-size: 1.34rem;
        font-weight: 860;
        margin-bottom: 14px;
    }
    .feature-copy {
        color: rgba(247, 242, 232, 0.58);
        line-height: 1.62;
        font-size: 1rem;
    }
    .feature-launch-note {
        color: rgba(247, 242, 232, 0.42);
        font-size: 0.82rem;
        margin: 10px 0 12px;
        text-align: center;
    }
    .capability-header {
        text-align: center;
        max-width: 820px;
        margin: 58px auto 36px;
    }
    .capability-header .badge {
        display: inline-flex;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.07);
        color: rgba(247,242,232,0.68);
        border-radius: 999px;
        padding: 8px 18px;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 22px;
    }
    .capability-header .title {
        color: var(--cream);
        font-size: 3.4rem;
        line-height: 1.05;
        font-weight: 880;
    }
    .capability-header .copy {
        color: rgba(247,242,232,0.56);
        font-size: 1.08rem;
        line-height: 1.62;
        margin-top: 18px;
    }
    .console-nav {
        display: none;
    }
    .workspace-frame {
        border: 1px solid rgba(255, 255, 255, 0.10);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.035)),
            var(--panel);
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: none;
        margin-top: 18px;
    }
    .workspace-frame:before {
        content: "LIVE MODULE";
        display: inline-flex;
        color: rgba(247, 242, 232, 0.48);
        font-weight: 850;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        margin-bottom: 8px;
    }
    .command-strip {
        display: none;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.050), rgba(255,255,255,0.030)),
            rgba(36, 33, 29, 0.78);
        border-radius: 999px;
        padding: 12px;
        margin: 14px auto 22px;
        max-width: 1120px;
        box-shadow: none;
    }
    .command-strip .stTextInput input {
        background: rgba(255,255,255,0.06) !important;
        color: #f5fbff !important;
        border-color: rgba(126, 255, 229, 0.18) !important;
    }
    .command-strip .stTextInput input::placeholder {
        color: rgba(245,251,255,0.46) !important;
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
        color: rgba(247, 242, 232, 0.50);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .brand-title {
        color: var(--cream);
        font-size: 1.75rem;
        font-weight: 780;
        line-height: 1.1;
        margin-top: 6px;
    }
    .brand-subtitle {
        color: rgba(247, 242, 232, 0.56);
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
        border-color: rgba(255,255,255,0.12) !important;
        background: rgba(255, 255, 255, 0.06) !important;
        color: var(--cream) !important;
        min-height: 48px;
        box-shadow: none;
    }
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stNumberInput"] div[data-testid="stNumberInputContainer"],
    div[data-testid="stSelectbox"] div[role="group"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.070), rgba(255,255,255,0.045)) !important;
        border: 1px solid rgba(255,255,255,0.13) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.06) !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] input[role="combobox"],
    div[data-testid="stSelectbox"] button {
        background: transparent !important;
        border-color: rgba(255, 255, 255, 0.14) !important;
        color: var(--cream) !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stSelectbox"] input::placeholder {
        color: rgba(247, 242, 232, 0.38) !important;
    }
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] * {
        color: rgba(247, 242, 232, 0.70) !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: rgba(247, 242, 232, 0.42) !important;
        box-shadow: 0 0 0 3px rgba(247, 242, 232, 0.08) !important;
    }
    .stButton button {
        min-height: 44px;
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.13);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.080), rgba(255,255,255,0.045));
        color: rgba(247, 242, 232, 0.84);
        box-shadow: none;
        backdrop-filter: blur(12px);
    }
    div[data-testid="stButton"] > button,
    button[data-testid="stBaseButton-secondary"] {
        color: rgba(247, 242, 232, 0.88) !important;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.085), rgba(255,255,255,0.050)) !important;
        border-color: rgba(255,255,255,0.13) !important;
    }
    div[data-testid="stButton"] > button *,
    button[data-testid="stBaseButton-secondary"] * {
        color: rgba(247, 242, 232, 0.88) !important;
    }
    .stButton button p {
        color: inherit !important;
        font-weight: 760;
    }
    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.035)),
            var(--panel);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 15px 16px;
        box-shadow: none;
        animation: liftIn 280ms ease both;
    }
    div[data-testid="stMetric"] label { color: rgba(247,242,232,0.58); font-weight: 700; }
    div[data-testid="stMetricValue"] { color: var(--cream); font-weight: 800; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: none;
    }
    .terminal-title {
        border: 1px solid rgba(255,255,255,0.10);
        border-left: 4px solid rgba(247,242,232,0.40);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.035)),
            var(--panel);
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: none;
        animation: liftIn 300ms ease both;
    }
    .terminal-title .title {
        color: var(--cream);
        font-size: 1.65rem;
        font-weight: 820;
        line-height: 1.2;
        margin-top: 5px;
    }
    .terminal-title .subtitle {
        color: rgba(247, 242, 232, 0.58);
        margin-top: 6px;
        font-size: 0.98rem;
    }
    .signal-panel {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.035)),
            var(--panel);
        padding: 17px 18px;
        margin: 12px 0 18px;
        box-shadow: none;
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
    .signal-headline { color: var(--cream); font-weight: 760; }
    .signal-panel ul {
        margin: 8px 0 0 1.1rem;
        padding: 0;
        color: rgba(247, 242, 232, 0.66);
        line-height: 1.68;
    }
    .signal-note {
        color: rgba(247, 242, 232, 0.48);
        font-size: 0.84rem;
        margin-top: 10px;
    }
    .small-note { color: rgba(247, 242, 232, 0.50); font-size: 0.9rem; }
    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 8px;
        background: #ffffff;
        box-shadow: none;
    }
    .stAlert {
        border-radius: 8px;
    }
    @media (max-width: 1200px) {
        .hero-title { font-size: 3.45rem; }
        .home-hero { min-height: auto; }
        .hero-content { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
        .brand-title { font-size: 1.35rem; }
        .system-chips { justify-content: flex-start; min-width: auto; }
        .app-shell { padding: 16px; }
        .terminal-title { padding: 17px; }
        .hero-content { grid-template-columns: 1fr; }
        .hero-title { font-size: 2.45rem; }
        .hero-copy { font-size: 0.98rem; }
        .hero-system-row { margin-bottom: 30px; }
        .hero-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .feature-grid { grid-template-columns: 1fr; }
        .home-hero { padding: 24px 20px; }
        .console-nav { position: static; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Editorial fintech presentation layer. This override intentionally targets
# Streamlit containers and the existing presentation classes only; all data,
# callbacks, state, and research calculations remain unchanged.
st.markdown(
    """
    <style>
    :root {
        --qtf-bg: #f4f3ee;
        --qtf-surface: #ffffff;
        --qtf-soft: #ecebe5;
        --qtf-ink: #151515;
        --qtf-muted: #777770;
        --qtf-line: rgba(21, 21, 21, 0.12);
        --qtf-dark: #090909;
        --qtf-dark-muted: rgba(255, 255, 255, 0.62);
        --qtf-accent: #d9f06f;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--qtf-bg) !important;
        color: var(--qtf-ink) !important;
    }
    .stApp {
        background:
            radial-gradient(circle at 86% -12%, rgba(217, 240, 111, 0.24), transparent 22%),
            var(--qtf-bg) !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(244, 243, 238, 0.92) !important;
        border-bottom: 1px solid var(--qtf-line) !important;
        box-shadow: none !important;
    }
    div[data-testid="stDecoration"] { display: none !important; }
    .block-container {
        max-width: 1460px !important;
        padding: 1.1rem 2.25rem 4.5rem !important;
    }
    .main .block-container > div { animation: qtfReveal 360ms ease both; }
    @keyframes qtfReveal {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stSidebar"] {
        background: #f7f6f1 !important;
        border-right: 1px solid var(--qtf-line) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] * { color: var(--qtf-ink) !important; }
    [data-testid="stSidebar"] .sidebar-brand {
        border: 0 !important;
        border-bottom: 1px solid var(--qtf-line) !important;
        border-radius: 0 !important;
        padding: 11px 0 20px !important;
        margin: 0 0 20px !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .sidebar-mark {
        width: 30px !important;
        height: 30px !important;
        border-radius: 5px !important;
        background: var(--qtf-dark) !important;
        color: #fff !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.06em;
    }
    [data-testid="stSidebar"] .sidebar-name {
        color: var(--qtf-ink) !important;
        font-size: 0.98rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .sidebar-desc,
    [data-testid="stSidebar"] .stCaption {
        color: var(--qtf-muted) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 4px !important;
        padding: 8px 9px !important;
        margin: 2px 0 !important;
        transition: background 160ms ease, transform 160ms ease !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(21, 21, 21, 0.06) !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: var(--qtf-dark) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] * { color: #fff !important; }

    h1, h2, h3, h4, p, li, label, .stMarkdown, .stCaption {
        color: var(--qtf-ink);
    }
    h1, h2, h3, h4 { letter-spacing: -0.025em !important; font-weight: 560 !important; }
    h1 { font-size: clamp(2.35rem, 4vw, 4.4rem) !important; line-height: 0.98 !important; }
    h2 { font-size: clamp(1.55rem, 2.5vw, 2.4rem) !important; }
    h3 { font-size: 1.32rem !important; }
    p, li { line-height: 1.65; }

    .terminal-title {
        border: 0 !important;
        border-bottom: 1px solid var(--qtf-line) !important;
        border-radius: 0 !important;
        background: transparent !important;
        padding: 38px 0 24px !important;
        margin: 0 0 28px !important;
        box-shadow: none !important;
    }
    .terminal-title .eyebrow,
    .brand-kicker {
        color: var(--qtf-muted) !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.16em !important;
    }
    .terminal-title .title {
        color: var(--qtf-ink) !important;
        font-size: clamp(2.15rem, 4vw, 4.8rem) !important;
        font-weight: 480 !important;
        letter-spacing: -0.055em !important;
        line-height: 0.98 !important;
        margin-top: 10px !important;
    }
    .terminal-title .subtitle {
        color: var(--qtf-muted) !important;
        font-size: 1rem !important;
        max-width: 760px;
        margin-top: 16px !important;
    }

    .home-hero {
        min-height: 66vh !important;
        padding: 82px 0 42px !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: var(--qtf-ink) !important;
        text-align: left !important;
        box-shadow: none !important;
    }
    .home-hero:before { opacity: 0.10 !important; }
    .hero-content {
        grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr) !important;
        gap: 8vw !important;
        justify-items: stretch !important;
    }
    .hero-system-row { display: flex !important; justify-content: flex-start !important; margin-bottom: 30px !important; }
    .hero-system-row span, .hero-kicker {
        border: 0 !important;
        border-radius: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        color: var(--qtf-muted) !important;
        letter-spacing: 0.17em !important;
        font-size: 0.7rem !important;
    }
    .hero-title {
        color: var(--qtf-ink) !important;
        max-width: 800px !important;
        font-size: clamp(3.3rem, 7.2vw, 7.2rem) !important;
        font-weight: 470 !important;
        letter-spacing: -0.075em !important;
        line-height: 0.91 !important;
        text-shadow: none !important;
    }
    .hero-title span { color: var(--qtf-ink) !important; }
    .hero-copy { color: var(--qtf-muted) !important; max-width: 660px !important; margin: 30px 0 0 !important; font-size: 1.08rem !important; }
    .hero-panel {
        display: block !important;
        align-self: stretch;
        width: auto !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 5px !important;
        background: var(--qtf-dark) !important;
        color: #fff !important;
        padding: 28px !important;
        box-shadow: 0 22px 50px rgba(0,0,0,0.12) !important;
    }
    .hero-panel-header { border-bottom-color: rgba(255,255,255,0.16) !important; }
    .hero-panel-header .label, .hero-terminal-line strong { color: var(--qtf-accent) !important; }
    .hero-panel-header .status { color: rgba(255,255,255,0.62) !important; }
    .hero-terminal-line { color: var(--qtf-dark-muted) !important; border-bottom-color: rgba(255,255,255,0.12) !important; padding: 18px 0 !important; }
    .hero-metrics { display: grid !important; margin: 54px 0 0 !important; max-width: none !important; grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
    .hero-metric { border: 0 !important; border-top: 1px solid var(--qtf-line) !important; border-radius: 0 !important; background: transparent !important; padding: 16px 0 !important; }
    .hero-metric .value { color: var(--qtf-ink) !important; font-weight: 550 !important; }
    .hero-metric .label { color: var(--qtf-muted) !important; }

    .feature-grid { gap: 0 !important; margin: 36px 0 20px !important; }
    .feature-card {
        border: 0 !important;
        border-top: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 0 !important;
        background: var(--qtf-dark) !important;
        min-height: 270px !important;
        padding: 30px 24px !important;
        box-shadow: none !important;
    }
    .feature-card:first-child { border-left: 1px solid rgba(255,255,255,0.16) !important; }
    .feature-card:before { display: none !important; }
    .feature-card:after { left: 24px !important; top: auto !important; bottom: 27px !important; background: var(--qtf-accent) !important; box-shadow: 7px 0 0 var(--qtf-accent), 14px 0 0 var(--qtf-accent) !important; }
    .feature-card:hover { transform: translateY(-3px) !important; background: #171717 !important; box-shadow: none !important; }
    .feature-index { color: rgba(255,255,255,0.45) !important; margin: 0 0 66px !important; font-weight: 650 !important; }
    .feature-title { color: #fff !important; font-size: 1.55rem !important; font-weight: 500 !important; }
    .feature-copy { color: rgba(255,255,255,0.62) !important; font-size: 0.96rem !important; }
    .feature-launch-note { color: var(--qtf-muted) !important; }

    .workspace-frame {
        border: 0 !important;
        border-radius: 5px !important;
        background: var(--qtf-dark) !important;
        color: #fff !important;
        padding: 24px 26px !important;
        box-shadow: none !important;
    }
    .workspace-frame:before { color: rgba(255,255,255,0.5) !important; }
    .brand-title, .brand-subtitle { color: #fff !important; }
    .brand-subtitle { color: var(--qtf-dark-muted) !important; }

    .stButton button,
    div[data-testid="stButton"] > button,
    button[data-testid="stBaseButton-secondary"] {
        min-height: 42px !important;
        border: 1px solid var(--qtf-ink) !important;
        border-radius: 4px !important;
        background: var(--qtf-dark) !important;
        color: #fff !important;
        box-shadow: none !important;
        font-weight: 600 !important;
        letter-spacing: 0 !important;
    }
    .stButton button:hover,
    div[data-testid="stButton"] > button:hover { transform: translateY(-2px); background: #2a2a2a !important; box-shadow: 0 10px 22px rgba(0,0,0,0.12) !important; }
    .stButton button:active { transform: translateY(0) scale(0.99) !important; }

    .stTextInput input, .stNumberInput input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stTextInputRootElement"],
    div[data-testid="stNumberInput"] div[data-testid="stNumberInputContainer"] {
        min-height: 44px !important;
        border: 1px solid var(--qtf-line) !important;
        border-radius: 4px !important;
        background: var(--qtf-surface) !important;
        color: var(--qtf-ink) !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] button { background: transparent !important; color: var(--qtf-ink) !important; }
    div[data-testid="stTextInput"] input::placeholder { color: #aaa9a2 !important; }
    label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] *, div[data-testid="stWidgetLabel"] * { color: var(--qtf-muted) !important; }

    div[data-testid="stMetric"] {
        border: 0 !important;
        border-top: 1px solid var(--qtf-line) !important;
        border-radius: 0 !important;
        background: transparent !important;
        padding: 16px 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] { color: var(--qtf-ink) !important; }
    div[data-testid="stMetric"] label { color: var(--qtf-muted) !important; font-size: 0.78rem !important; }
    div[data-testid="stMetricValue"] { font-weight: 520 !important; letter-spacing: -0.04em; }

    div[data-testid="stDataFrame"], div[data-testid="stTable"], [data-testid="stPlotlyChart"] {
        border: 1px solid var(--qtf-line) !important;
        border-radius: 4px !important;
        background: var(--qtf-surface) !important;
        box-shadow: none !important;
    }
    [data-testid="stPlotlyChart"] { padding: 7px !important; }
    [data-testid="stExpander"] { border: 1px solid var(--qtf-line) !important; border-radius: 4px !important; background: transparent !important; }
    .signal-panel { border-radius: 4px !important; background: var(--qtf-surface) !important; border-color: var(--qtf-line) !important; box-shadow: none !important; }
    .signal-headline, .signal-panel ul, .signal-note { color: var(--qtf-ink) !important; }
    .signal-note { color: var(--qtf-muted) !important; }
    .stAlert { border-radius: 4px !important; }
    .stMarkdown code { background: var(--qtf-soft) !important; color: var(--qtf-ink) !important; }

    @media (max-width: 900px) {
        .block-container { padding: 0.8rem 1rem 3rem !important; }
        .hero-content { grid-template-columns: 1fr !important; gap: 36px !important; }
        .hero-title { font-size: clamp(3rem, 12vw, 5rem) !important; }
        .hero-panel { min-height: auto; }
        .hero-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
        .feature-grid { grid-template-columns: 1fr !important; }
        .feature-card:first-child { border-left: 0 !important; }
    }
    @media (max-width: 560px) {
        .terminal-title { padding-top: 24px !important; }
        .terminal-title .title { font-size: 2.45rem !important; }
        .hero-title { font-size: 3.05rem !important; }
        .hero-copy { font-size: 0.98rem !important; }
        .hero-metrics { grid-template-columns: 1fr !important; }
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


def clear_project_data_cache() -> None:
    """Clear cached dashboard reads after the processed dataset is rebuilt."""

    read_csv.clear()
    load_features.clear()


def refresh_project_market_data(tickers: list[str], start: str) -> pd.DataFrame:
    """Rebuild the local project universe dataset from yfinance."""

    with tempfile.TemporaryDirectory(prefix="qtf_market_refresh_") as tmp:
        tmp_root = Path(tmp)
        config = MarketDataConfig(
            tickers=tuple(tickers),
            start=start,
            end=next_yfinance_end_date(),
            raw_dir=tmp_root / "raw",
            processed_dir=tmp_root / "processed",
        )
        refreshed = build_market_dataset(config)

        ticker_count = refreshed["ticker"].nunique() if "ticker" in refreshed.columns else 0
        minimum_coverage = max(1, int(len(tickers) * 0.90))
        if ticker_count < minimum_coverage:
            raise ValueError(f"downloaded {ticker_count}/{len(tickers)} tickers; keeping the existing dataset")

        if market_dataset_is_stale(refreshed):
            latest = latest_market_date(refreshed)
            raise ValueError(f"latest downloaded date is {latest.date() if latest is not None else 'missing'}")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for csv_file in (tmp_root / "processed").glob("*.csv"):
            shutil.copy2(csv_file, PROCESSED_DIR / csv_file.name)

        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in (tmp_root / "raw").glob("*.csv"):
            shutil.copy2(csv_file, raw_dir / csv_file.name)

    clear_project_data_cache()
    return refreshed


def maybe_auto_refresh_project_data(tickers: list[str], start: str, features: pd.DataFrame) -> pd.DataFrame:
    """Refresh stale project data once per browser session."""

    if not AUTO_REFRESH_DATA or not market_dataset_is_stale(features):
        return features

    latest = latest_market_date(features)
    expected = expected_latest_market_date()
    refresh_key = f"{latest.date() if latest is not None else 'missing'}->{expected.date()}"
    if st.session_state.get("project_data_refresh_attempt") == refresh_key:
        return features

    st.session_state["project_data_refresh_attempt"] = refresh_key
    with st.spinner("Refreshing project market data from yfinance..."):
        try:
            refreshed = refresh_project_market_data(tickers, start)
        except Exception as exc:
            st.session_state["project_data_refresh_error"] = str(exc)
            return features

    st.session_state["project_data_refresh_error"] = ""
    st.session_state["project_data_refresh_success"] = latest_market_date(refreshed)
    st.toast("Project market data refreshed.")
    return load_features()


def render_project_data_status(features: pd.DataFrame, tickers: list[str], start: str) -> pd.DataFrame:
    """Show project data freshness and expose a manual rebuild button."""

    latest = latest_market_date(features)
    expected = expected_latest_market_date()
    stale = market_dataset_is_stale(features)

    status_text = (
        f"Project dataset latest date: {latest.date() if latest is not None else 'missing'}; "
        f"expected latest closed business date: {expected.date()}."
    )
    if stale:
        st.warning(f"{status_text} Local data may be stale.")
    else:
        st.caption(f"{status_text} Data is current enough for daily research.")

    if st.session_state.get("project_data_refresh_error"):
        st.caption(f"Last automatic refresh failed: {st.session_state['project_data_refresh_error']}")

    if st.button("Refresh project dataset from yfinance", key="refresh_project_dataset"):
        with st.spinner("Rebuilding project universe data, fundamentals, and technical features..."):
            try:
                refreshed = refresh_project_market_data(tickers, start)
            except Exception as exc:
                st.error(f"Project dataset refresh failed: {exc}")
                return features
        st.success(f"Project dataset refreshed through {latest_market_date(refreshed).date()}.")
        return load_features()

    return features


def normalize_ticker(value: str) -> str:
    ticker = re.sub(r"[^A-Za-z0-9.^=-]", "", value).upper()
    if ticker.startswith("US."):
        ticker = ticker[3:]
    if re.fullmatch(r"SZ\.\d{6}", ticker):
        return f"{ticker[-6:]}.SZ"
    if re.fullmatch(r"SH\.\d{6}", ticker):
        return f"{ticker[-6:]}.SS"
    if re.fullmatch(r"HK\.\d{1,5}", ticker):
        return f"{ticker.split('.')[-1].zfill(4)}.HK"
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
def load_backtest_manifest(signature: tuple[str, float, int]) -> dict:
    path, _, size = signature
    if size == 0:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_reproducibility_panel() -> None:
    manifest = load_backtest_manifest(file_signature(RESULTS_DIR / "backtest_manifest.json"))
    if not manifest:
        return
    data = manifest.get("data", {})
    backtest = manifest.get("backtest", {})
    validation = manifest.get("validation", {})
    with st.expander("回测可复现性与数据审计", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("数据截止日", data.get("end_date") or "NA")
        c2.metric("股票数量", str(data.get("ticker_count", "NA")))
        c3.metric("数据行数", f"{int(data.get('rows', 0)):,}")
        c4.metric("信号滞后", f"{backtest.get('signal_lag_sessions', 'NA')} 个交易日")
        st.write(
            f"数据源：{data.get('source', 'NA')}；价格口径：{data.get('price_field', 'NA')}；"
            f"交易成本：{backtest.get('transaction_cost_bps', 'NA')} bps；"
            f"滑点：{backtest.get('slippage_bps', 'NA')} bps。"
        )
        if validation.get("chronological_signal_lag") and not validation.get("future_price_backfill"):
            st.success("已启用至少 1 个交易日信号滞后，历史缺失价格不会用未来值回填。")
        warnings = validation.get("warnings", [])
        if warnings:
            st.warning("研究限制：" + "；".join(str(item) for item in warnings))
        st.caption(f"配置指纹：{data.get('configuration_sha256', 'NA')}")


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


def render_home_hero(ticker_count: int) -> None:
    st.markdown(
        f"""
        <section class="home-hero">
            <div class="hero-system-row">
                <span>Public AI Quant Demo</span>
                <span>Equity Strategies</span>
                <span>Derivatives Pricing</span>
                <span>Risk Attribution</span>
            </div>
            <div class="hero-content">
                <div>
                    <div class="hero-kicker">Features</div>
                    <div class="hero-title">Quant research OS for <span>models, risk, and execution.</span></div>
                    <div class="hero-copy">
                        Explore equities, backtests, risk factors, machine-learning diagnostics,
                        and derivatives pricing through one public-facing research interface.
                    </div>
                    <div class="hero-query">
                        <strong>TRY</strong> NVDA momentum signal / SPY factor exposure / Black-Scholes Greeks / hedge-frequency error
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-header">
                        <span class="label">Research Pipeline</span>
                        <span class="status">Online</span>
                    </div>
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


def render_hero_controls() -> None:
    col1, col2, col3 = st.columns([1.25, 1.25, 3.5])
    with col1:
        st.button("Enter Research Terminal", key="hero_open_terminal", on_click=set_active_module, args=("stock",))
    with col2:
        st.button("View Capability Map", key="hero_view_map", on_click=set_active_module, args=("home",))
    with col3:
        st.markdown(
            '<div class="feature-launch-note">Module changes happen inside this page. No URL jump, no separate workspace.</div>',
            unsafe_allow_html=True,
        )


def render_capability_grid() -> None:
    st.markdown('<div id="module-dock"></div>', unsafe_allow_html=True)
    for row_start in range(0, len(FEATURE_MODULES), 3):
        columns = st.columns(3)
        for column, feature in zip(columns, FEATURE_MODULES[row_start : row_start + 3]):
            key, index, title, copy = feature
            with column:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-index">{escape(index)}</div>
                        <div class="feature-title">{escape(title)}</div>
                        <div class="feature-copy">{escape(copy)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(f"Launch {title}", key=f"feature_launch_{key}", on_click=set_active_module, args=(key,))


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


MODULE_NAV = {
    "personal": ("总览", "每日投研、市场状态和顶部行动建议"),
    "trade": ("买卖建议", "结合基本面、估值、技术面、消息面和仓位约束"),
    "portfolio": ("持仓分析", "IBKR持仓、成本、现价、盈亏和风险贡献"),
    "ranking": ("股票排名", "多因子评分、解释和估值空间"),
    "opportunity": ("潜力股雷达", "每日新挖掘机会、历史推荐和证据门槛"),
    "watchlist": ("自选回测", "Moomoo自选股组合动态回测"),
    "strategy": ("策略实验室", "动量、均值回归和多因子策略回测"),
    "stock": ("股票智能", "任意股票日K、MACD、RSI、估值和买卖点"),
    "risk_center": ("风险中心", "组合波动、回撤、Sharpe、Beta和集中度"),
    "factor": ("因子暴露", "Fama-French暴露、Alpha和市场敏感度"),
    "derivatives": ("衍生品实验室", "期权定价、Greeks和对冲实验"),
    "earnings": ("财报研究", "财报日历、已发布财报解析和投资价值判断"),
    "data_security": ("数据安全", "本地私有持仓、交易流水和接入状态"),
}


FEATURE_MODULES = [
    (
        "personal",
        "00 / portfolio",
        "个人AI投研平台",
        "读取你的持仓、自选股和交易流水，动态查看股票研究、排名、雷达、财报、回测和风险。",
    ),
    (
        "stock",
        "01 / explore",
        "股票浏览器",
        "查询项目股票或实时 Yahoo Finance 代码，查看技术指标和中文结论。",
    ),
    (
        "backtest",
        "02 / strategies",
        "回测工作台",
        "比较动量、均值回归和多因子策略，包含交易成本、滑点和信号滞后。",
    ),
    (
        "risk",
        "03 / risk",
        "风险归因",
        "查看 Sharpe、回撤、Beta、Alpha、Tracking Error、因子暴露和参数敏感性。",
    ),
    (
        "data",
        "04 / intelligence",
        "数据与模型诊断",
        "检查数据质量和无未来函数的收益预测基线。",
    ),
    (
        "derivatives",
        "05 / derivatives",
        "衍生品实验室",
        "使用 Black-Scholes、二叉树和 Monte Carlo 做期权定价，并查看 Greeks。",
    ),
    (
        "ai",
        "06 / research desk",
        "AI投研台",
        "追踪投资逻辑、因子评分、情景估值、风险笔记和交易行为反馈。",
    ),
]


def initialize_module_state() -> None:
    if "active_module" not in st.session_state:
        st.session_state["active_module"] = "personal"


def set_active_module(module: str) -> None:
    st.session_state["active_module"] = module if module in MODULE_NAV else "personal"


def active_module() -> str:
    initialize_module_state()
    module = st.session_state.get("active_module", "personal")
    return module if module in MODULE_NAV else "personal"


def render_sidebar_nav(current: str) -> None:
    keys = list(MODULE_NAV)
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-mark">AQ</div>
                <div class="sidebar-name">个人 AI 投研终端</div>
                <div class="sidebar-desc">持仓、自选股、交易流水、财报、估值、回测、风险与买卖建议。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = st.radio(
            "目录",
            keys,
            index=keys.index(current) if current in keys else 0,
            format_func=lambda key: MODULE_NAV[key][0],
            label_visibility="collapsed",
        )
        st.caption("Research tool only. Not investment advice.")
    if selected != current:
        set_active_module(selected)
        st.rerun()


def render_console_nav(current: str) -> None:
    st.markdown('<nav class="console-nav">', unsafe_allow_html=True)
    columns = st.columns(len(MODULE_NAV))
    for column, (key, (label, _)) in zip(columns, MODULE_NAV.items()):
        button_label = label
        with column:
            st.button(button_label, key=f"module_nav_{key}", on_click=set_active_module, args=(key,))
    st.markdown("</nav>", unsafe_allow_html=True)


def render_command_search(tickers: list[str]) -> None:
    st.markdown('<div class="command-strip">', unsafe_allow_html=True)
    query = st.text_input(
        "Global Search",
        placeholder="搜索：NVDA、财报、潜力股、Sharpe、Fama-French、Black-Scholes...",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    render_search(query, tickers)


def render_workspace_frame(module: str) -> None:
    label, subtitle = MODULE_NAV[module]
    st.markdown(
        f"""
        <div id="workspace" class="workspace-frame">
            <div class="brand-kicker">{escape(label)}</div>
            <div class="brand-title">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
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


def render_personal_quant_platform() -> None:
    render_page_title(
        "PERSONAL AI QUANT",
        "个人 AI 投研总览",
        "动态接入你的 IBKR 持仓、Moomoo 自选股、交易流水、财报、股票排名、潜力股雷达、回测和风险研究。",
    )
    snapshot = load_investment_snapshot(file_signature(latest_investment_snapshot_path()))
    features = load_features()
    if not snapshot:
        st.warning("还没有 investment_platform snapshot。请先运行每日投研刷新。")
        return
    if features.empty:
        st.warning("还没有 feature_dataset，回测和技术图表暂不可用。")

    st.caption("This is a research tool, not investment advice. 本页面只读研究，不会提交、修改或取消任何 IBKR 订单。")
    render_personal_refresh_controls(features)
    render_personal_briefing(snapshot)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("股票排名快照")
            scores = pd.DataFrame(snapshot.get("scores", []))
            if scores.empty:
                render_missing_results()
            else:
                shown = scores.sort_values("total_score", ascending=False).head(8).copy()
                columns = [column for column in ["ticker", "total_score", "explanation"] if column in shown]
                st.dataframe(shown[columns], width="stretch", hide_index=True)
    with right:
        with st.container(border=True):
            st.subheader("潜力股雷达快照")
            themes = pd.DataFrame(snapshot.get("theme_candidates", []))
            if themes.empty:
                render_missing_results()
            else:
                columns = [column for column in ["ticker", "theme", "potential_score", "priority", "history_tag"] if column in themes]
                st.dataframe(themes[columns].head(8), width="stretch", hide_index=True)

    with st.container(border=True):
        st.subheader("财报与消息更新")
        render_personal_earnings_digest(snapshot)


def load_personal_context() -> tuple[dict, pd.DataFrame]:
    snapshot = load_investment_snapshot(file_signature(latest_investment_snapshot_path()))
    features = load_features()
    if not snapshot:
        st.warning("还没有 investment_platform snapshot。请先运行每日投研刷新。")
    return snapshot, features


def render_personal_earnings_digest(snapshot: dict) -> None:
    calendar = pd.DataFrame(snapshot.get("earnings_calendar", []))
    reviews = pd.DataFrame(snapshot.get("earnings_reviews", []))
    columns = st.columns(2)
    with columns[0]:
        st.write("**近期财报日历**")
        if calendar.empty:
            st.info("暂无财报日历。")
        else:
            shown = calendar.copy()
            keep = [column for column in ["ticker", "company", "date", "time", "status", "verdict", "next_step"] if column in shown]
            st.dataframe(shown[keep].head(12), width="stretch", hide_index=True)
    with columns[1]:
        st.write("**已解析财报**")
        if reviews.empty:
            st.info("暂无已解析财报。")
        else:
            keep = [column for column in ["ticker", "date", "verdict", "quality_score", "thesis_impact", "action"] if column in reviews]
            st.dataframe(reviews[keep].head(12), width="stretch", hide_index=True)


def moomoo_watchlist_tickers(snapshot: dict) -> list[str]:
    tickers: set[str] = set()
    for row in snapshot.get("moomoo_watchlist_rows", []) or []:
        ticker = normalize_ticker(str(row.get("ticker") or row.get("code") or ""))
        if ticker:
            tickers.add(ticker)
    for group in snapshot.get("watchlists", []) or []:
        if "moomoo" not in str(group.get("source", "")).lower() and "moomoo" not in str(group.get("name", "")).lower():
            continue
        tickers.update(normalize_ticker(str(ticker)) for ticker in group.get("tickers", []) if ticker)
    return sorted(ticker for ticker in tickers if ticker)


def render_moomoo_watchlist_status(snapshot: dict, features: pd.DataFrame) -> None:
    rows = snapshot.get("moomoo_watchlist_rows", []) or []
    watchlists = [group for group in snapshot.get("watchlists", []) or [] if "moomoo" in str(group.get("source", "")).lower() or "moomoo" in str(group.get("name", "")).lower()]
    moomoo_tickers = moomoo_watchlist_tickers(snapshot)
    available = set(features["ticker"].astype(str).str.upper()) if not features.empty and "ticker" in features else set()
    with_data = [ticker for ticker in moomoo_tickers if ticker in available]
    missing = [ticker for ticker in moomoo_tickers if ticker not in available]

    st.subheader("Moomoo 自选股接入状态")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moomoo导入股票", str(len(moomoo_tickers)))
    c2.metric("已有本地日K", str(len(with_data)))
    c3.metric("缺少本地日K", str(len(missing)))
    source_paths = sorted({str(row.get("source_path", "")) for row in rows if row.get("source_path")})
    c4.metric("导出源文件", str(len(source_paths)))
    st.caption("当前是只读接入：优先读取 Moomoo 导出文件和本地采集文件；缺日K的自选股可以在回测页勾选实时补齐，使用 Yahoo Finance 日K降级补齐。不会提交、修改或取消任何订单。")

    if source_paths:
        with st.expander("查看 Moomoo 导出/缓存来源", expanded=False):
            for path in source_paths:
                st.write(f"- {path}")
    if watchlists:
        with st.expander("查看完整 Moomoo 自选股分组", expanded=False):
            summary = pd.DataFrame(
                [
                    {
                        "分组": group.get("name"),
                        "来源": group.get("source"),
                        "股票数": len(group.get("tickers", [])),
                        "股票": ", ".join(group.get("tickers", [])[:120]),
                    }
                    for group in watchlists
                ]
            )
            st.dataframe(summary, width="stretch", hide_index=True)
    if missing:
        with st.expander("缺少本地日K、需实时补齐的自选股", expanded=False):
            st.write(", ".join(missing))


def render_trade_advice_page() -> None:
    render_page_title(
        "ACTION DESK",
        "买卖建议",
        "不是机械按仓位加减仓，而是结合基本面、估值、技术面、消息面、现金和组合波动给出行动建议。",
    )
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_personal_briefing(snapshot)


def render_portfolio_analysis_page() -> None:
    render_page_title("PORTFOLIO", "持仓分析", "读取 IBKR 持仓和本地交易流水，显示成本、现价、盈亏、昨日涨跌和风险贡献。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_interactive_portfolio_tab(snapshot, features)
    render_trade_ledger_tab(snapshot, features)


def render_ranking_page() -> None:
    render_page_title("STOCK RANKING", "股票排名", "多因子评分结合成长、质量、动量、估值和风险，并解释为什么排名高或低。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_personal_ranking_tab(snapshot)


def render_opportunity_page() -> None:
    render_page_title("OPPORTUNITY RADAR", "潜力股雷达", "不限于已有持仓，跟踪 AI、半导体、医疗创新、能源基础设施等新机会，并保留历史推荐记录。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_personal_opportunity_radar_tab(snapshot)


def render_watchlist_backtest_page() -> None:
    render_page_title("WATCHLIST BACKTEST", "自选回测", "读取 Moomoo 自选股，支持动态选择股票、权重、区间，并和 SPY 比较。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_moomoo_watchlist_status(snapshot, features)
    render_dynamic_watchlist_backtest_tab(snapshot, features)
    render_reproducibility_panel()


def render_strategy_lab_page() -> None:
    render_page_title("BACKTEST LAB", "策略实验室", "复用量化框架里的动量、均值回归、多因子策略，支持股票池、交易成本、滑点和信号滞后。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_interactive_strategy_tab(snapshot, features)
    render_reproducibility_panel()


def render_stock_intelligence_page() -> None:
    render_page_title("STOCK INTELLIGENCE", "股票智能", "查询任意股票，动态生成日K、MACD、RSI、Bollinger、估值、支撑压力、买卖区和单股回测。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_personal_refresh_controls(features)
    render_interactive_stock_tab(snapshot, features)


def render_risk_center_page() -> None:
    render_page_title("RISK CENTER", "风险中心", "分析组合波动、回撤、Sharpe、Sortino、Beta、Tracking Error、集中度和风险贡献。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_interactive_risk_factor_tab(snapshot, features)
    st.caption("中文解释：风险中心优先使用完整交易流水重建真实净值；没有流水时使用当前持仓快照和已有因子结果作为降级分析。")


def render_factor_exposure_page() -> None:
    render_page_title("FACTOR EXPOSURE", "因子暴露", "查看 Fama-French 因子、Alpha、Beta、SMB、HML、p-value 和 R-squared。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    exposure = read_csv(file_signature(RESULTS_DIR / "factor_exposure.csv"))
    if exposure.empty:
        st.info("暂无 Fama-French 因子暴露结果。请先运行策略/风险模块生成因子文件。")
    else:
        st.dataframe(exposure, width="stretch", hide_index=True)
        st.markdown(
            """
            **怎么读：** Market beta 越高，组合越像大盘高弹性资产；SMB 为正表示偏小盘，为负表示偏大盘；
            HML 为正表示偏价值，为负表示偏成长；Alpha 是扣除因子后的超额收益估计，p-value 越低越有统计意义。
            """
        )
    render_interactive_risk_factor_tab(snapshot, features)


def render_earnings_research_page() -> None:
    render_page_title("EARNINGS RESEARCH", "财报研究", "跟踪自选股和持仓股财报日历，并展示已发布财报的详细解析、电话会要点和投资价值判断。")
    snapshot, _features = load_personal_context()
    if not snapshot:
        return
    render_personal_earnings_detail_tab(snapshot)


def render_data_security_page() -> None:
    render_page_title("DATA SAFETY", "数据安全与接入", "真实 IBKR / Moomoo / API key / 交易流水只保留在本地，不提交到 GitHub。")
    snapshot, features = load_personal_context()
    if not snapshot:
        return
    render_implementation_connection_tab(snapshot, features)


def render_personal_briefing(snapshot: dict) -> None:
    regime = snapshot.get("market_regime", {})
    allocation = snapshot.get("allocation_plan", {})
    daily = snapshot.get("portfolio_daily_review", {})
    weights = snapshot.get("factor_weights", {})
    top = pd.DataFrame(snapshot.get("top_recommendations", []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场状态", regime.get("name", "NA"), f"置信度 {percent_value(regime.get('confidence'))}")
    c2.metric("估算账户日波动", percent_value(allocation.get("estimated_daily_volatility")), "目标 2%-3%")
    c3.metric("现金权重", percent_value(allocation.get("cash_weight")), f"最低 {percent_value(allocation.get('minimum_cash_weight'))}")
    c4.metric("昨日持仓盈亏", _money_value(daily.get("total_daily_pnl")), percent_value(daily.get("total_daily_return")))

    st.subheader("市场状态解释")
    for driver in regime.get("drivers", []):
        st.write(f"- {_driver_cn(str(driver))}")
    st.caption(_reason_cn(str(weights.get("reason", ""))))

    factor_frame = pd.DataFrame(
        [
            {"因子": "成长", "权重": weights.get("growth", 0), "解释": "AI增长环境下仍是主因子，但要受风险预算约束。"},
            {"因子": "质量", "权重": weights.get("quality", 0), "解释": "盈利质量和现金流越稳定，越适合作为核心仓。"},
            {"因子": "动量", "权重": weights.get("momentum", 0), "解释": "趋势确认有用，但不能替代买点和止损。"},
            {"因子": "估值", "权重": weights.get("value", 0), "解释": "衡量安全边际，防止好公司用坏价格买。"},
            {"因子": "风险", "权重": weights.get("risk", 0), "解释": "按你的要求单独提高，用来压制高波动和高相关仓位。"},
        ]
    )
    fig = go.Figure(data=[go.Bar(x=factor_frame["因子"], y=factor_frame["权重"], marker_color="#0b766d")])
    st.plotly_chart(style_figure(fig, "动态因子权重", height=320), width="stretch")
    st.dataframe(factor_frame.assign(权重=lambda x: x["权重"].map(percent_value)), width="stretch", hide_index=True)

    if not top.empty:
        st.subheader("顶部买卖建议")
        columns = ["ticker", "decision", "problem_area", "current_allocation", "target_allocation", "suggested_amount", "buy_zone", "trim_zone", "stop_loss", "reason"]
        shown = top[[column for column in columns if column in top.columns]].copy()
        shown = shown.rename(
            columns={
                "ticker": "股票",
                "decision": "结论",
                "problem_area": "问题来源",
                "current_allocation": "当前仓位",
                "target_allocation": "目标仓位",
                "suggested_amount": "建议金额",
                "buy_zone": "买入区",
                "trim_zone": "减仓区",
                "stop_loss": "止损/失效",
                "reason": "原因",
            }
        )
        for column in ["当前仓位", "目标仓位"]:
            if column in shown:
                shown[column] = shown[column].map(percent_value)
        if "建议金额" in shown:
            shown["建议金额"] = shown["建议金额"].map(_money_value)
        st.dataframe(shown, width="stretch", hide_index=True)


def render_personal_refresh_controls(features: pd.DataFrame) -> None:
    latest = latest_market_date(features)
    expected = expected_latest_market_date()
    status = "已到最新交易日" if latest is not None and latest >= expected else "需要刷新"
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    c1.metric("本地市场数据", status, f"{latest.date() if latest is not None else 'missing'} / 应到 {expected.date()}")
    if c2.button("刷新市场日K", key="personal_refresh_market_data"):
        data_config = load_dashboard_config().get("data", {})
        try:
            with st.spinner("正在从 yfinance 刷新市场日K和技术指标..."):
                refresh_project_market_data(load_config_tickers(), data_config.get("start", "2015-01-01"))
        except Exception as exc:
            st.error(f"市场数据刷新失败：{exc}")
        else:
            st.success("市场日K已刷新。")
            st.cache_data.clear()
            st.rerun()
    if c3.button("全量刷新投研", key="personal_refresh_research"):
        result = _run_investment_module("all")
        if result.returncode == 0:
            st.success("全量投研刷新完成，已更新 snapshot / 日报 / 网站数据。")
            st.code(result.stdout[-1200:] or "OK")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("全量投研刷新失败。")
            st.code((result.stderr or result.stdout)[-2000:])


def _run_investment_module(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_investment_module.py"), "--module", module, "--config", "investment_platform.json"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def render_implementation_connection_tab(snapshot: dict, features: pd.DataFrame) -> None:
    st.subheader("实施连接状态")
    snapshot_path = latest_investment_snapshot_path()
    latest_trade = latest_trade_file(PROJECT_ROOT)
    trade_files = private_trade_files(PROJECT_ROOT)
    market_latest = latest_market_date(features)
    account = snapshot.get("account_summary", {}) or {}
    portfolio = pd.DataFrame(snapshot.get("portfolio", []))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("持仓快照", "已接入" if not portfolio.empty else "缺失", snapshot_path.name if snapshot_path.exists() else "NA")
    c2.metric("完整交易流水", "已接入" if latest_trade else "待上传", latest_trade.name if latest_trade else "data/private")
    c3.metric("市场数据日期", str(market_latest.date()) if market_latest is not None else "NA")
    c4.metric("账户净值", _money_value(account.get("net_liquidation") or account.get("equity_with_loan")))

    connection_rows = [
        {
            "连接项": "IBKR 持仓/账户快照",
            "状态": "已接入" if not portfolio.empty else "缺失",
            "用途": "读取当前持仓、成本、现价、市值、现金和未实现盈亏",
            "位置/来源": str(snapshot_path),
        },
        {
            "连接项": "IBKR 完整交易流水",
            "状态": "已接入" if latest_trade else "待上传",
            "用途": "重建真实历史净值、真实成本、交易贡献、风险和行为分析",
            "位置/来源": str(latest_trade or (PROJECT_ROOT / "data" / "private" / "trades.csv")),
        },
        {
            "连接项": "Moomoo 自选股与点位",
            "状态": "已接入" if snapshot.get("watchlists") or snapshot.get("stock_details") else "待刷新",
            "用途": "自选股股票池、支撑压力、短线观察和财报跟踪",
            "位置/来源": "每日投研 snapshot / moomoo 导出",
        },
        {
            "连接项": "量化研究框架",
            "状态": "已嵌入",
            "用途": "组合净值、策略回测、技术指标、风险指标、因子暴露、衍生品实验",
            "位置/来源": "src/backtesting, src/risk, src/strategies, src/derivatives",
        },
    ]
    st.dataframe(pd.DataFrame(connection_rows), width="stretch", hide_index=True)

    st.subheader("完整交易流水导入模板")
    st.caption("可以直接上传 IBKR Activity Statement / Executions CSV；如果手工整理，使用下面最小字段。文件只保存在本地 data/private，不会进入 Git。")
    template = "date,ticker,side,quantity,price,commission,fees\n2026-08-21,NVDA,BUY,5,180.25,1.00,0.00\n"
    st.download_button("下载 trades.csv 模板", data=template, file_name="trades_template.csv", mime="text/csv")
    if st.button("重新扫描本地私有流水文件"):
        st.cache_data.clear()
        st.rerun()

    if trade_files:
        st.subheader("已发现的私有交易流水")
        rows = [
            {
                "文件": path.name,
                "修改时间": pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "大小": f"{path.stat().st_size / 1024:.1f} KB",
                "路径": str(path),
            }
            for path in trade_files
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_trade_ledger_tab(snapshot: dict, features: pd.DataFrame) -> None:
    st.subheader("完整交易流水接入")
    st.caption(f"支持字段：{', '.join(TRADE_REQUIRED_COLUMNS)}。兼容 symbol/qty/exec_price/action/trade_date 等常见别名。")
    template = "date,ticker,side,quantity,price,commission,fees\n2026-08-21,NVDA,BUY,5,180.25,1.00,0.00\n"
    st.download_button("下载 trades.csv 模板", data=template, file_name="trades_template.csv", mime="text/csv", key="trade_template_download")
    uploaded = st.file_uploader("上传 IBKR activity statement / executions CSV / trades.csv", type=["csv"], key="private_trade_upload")
    if uploaded is not None:
        target = write_uploaded_trades(PROJECT_ROOT, uploaded.getvalue(), uploaded.name)
        st.success(f"已保存到本地私有目录：{target}")
        st.cache_data.clear()

    files = private_trade_files(PROJECT_ROOT)
    if not files:
        st.warning("还没有检测到完整交易流水。当前只可展示最新持仓快照；真实历史净值需要上传 IBKR trades/activity statement。")
        st.code("quant-trading-framework/data/private/trades.csv")
        return

    labels = [f"{path.name} · {pd.Timestamp(path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}" for path in files]
    selected_idx = st.selectbox("选择交易流水文件", range(len(files)), format_func=lambda index: labels[index], index=len(files) - 1)
    selected_path = files[int(selected_idx)]
    try:
        trades = read_trade_file(selected_path)
    except Exception as exc:
        st.error(f"交易流水解析失败：{exc}")
        return
    st.success(f"已读取 {len(trades)} 条交易：{selected_path}")

    initial_cash = st.number_input("初始现金/本金（留空逻辑：按流水推断最低所需现金）", min_value=0.0, value=0.0, step=1000.0)
    cash_value = None if initial_cash == 0 else float(initial_cash)
    result = reconstruct_from_trades(features, trades, initial_cash=cash_value)
    for warning in result.warnings:
        st.warning(warning)

    if result.portfolio_value.empty:
        st.info("交易流水已读取，但缺少可匹配的历史价格，暂时不能重建净值。")
        st.dataframe(trades, width="stretch", hide_index=True)
        return

    summary = result.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", percent_value(summary.get("cumulative_return")))
    c2.metric("年化波动", percent_value(summary.get("annualized_volatility")))
    c3.metric("Sharpe", _ratio_value(summary.get("sharpe_ratio")))
    c4.metric("最大回撤", percent_value(summary.get("maximum_drawdown")))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.portfolio_value["date"], y=result.portfolio_value["total_value"], mode="lines", name="真实交易流水净值"))
    fig.add_trace(go.Scatter(x=result.portfolio_value["date"], y=result.portfolio_value["cash"], mode="lines", name="现金"))
    fig.add_trace(go.Scatter(x=result.portfolio_value["date"], y=result.portfolio_value["holdings"], mode="lines", name="持仓市值"))
    st.plotly_chart(style_figure(fig, "交易流水重建组合净值", height=430), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("当前持仓贡献")
        st.dataframe(_format_money_percent_frame(result.contribution), width="stretch", hide_index=True)
    with right:
        st.subheader("交易流水预览")
        st.dataframe(trades.tail(50), width="stretch", hide_index=True)

    reconciliation = _ledger_snapshot_reconciliation(result.contribution, pd.DataFrame(snapshot.get("portfolio", [])))
    if not reconciliation.empty:
        st.subheader("交易流水 vs 当前 IBKR 持仓对账")
        st.caption("如果数量或市值差异明显，通常说明流水文件不完整、拆股/分红尚未调整，或 IBKR 快照比本地价格更新。")
        st.dataframe(_format_money_percent_frame(reconciliation), width="stretch", hide_index=True)


def render_interactive_portfolio_tab(snapshot: dict, features: pd.DataFrame) -> None:
    portfolio = pd.DataFrame(snapshot.get("portfolio", []))
    if portfolio.empty:
        st.info("没有持仓快照。")
        return
    st.subheader("当前 IBKR 持仓快照")
    shown = portfolio.copy()
    for column in ["allocation"]:
        if column in shown:
            shown[column] = shown[column].map(percent_value)
    for column in ["avg_cost", "ibkr_price", "market_value", "unrealized_pnl", "daily_pnl"]:
        if column in shown:
            shown[column] = shown[column].map(_money_value)
    st.dataframe(shown, width="stretch", hide_index=True)

    if "allocation" in portfolio and "sector" in portfolio:
        sector = portfolio.groupby("sector", dropna=False)["allocation"].sum().sort_values(ascending=False)
        fig = go.Figure(data=[go.Pie(labels=sector.index, values=sector.values, hole=0.45)])
        st.plotly_chart(style_figure(fig, "行业/主题仓位分布", height=360), width="stretch")

    daily = pd.DataFrame(snapshot.get("portfolio_daily_review", {}).get("rows", []))
    if not daily.empty:
        st.subheader("昨日持仓涨跌回顾")
        daily_shown = daily[["ticker", "latest_date", "daily_return", "daily_pnl", "unrealized_pnl", "unrealized_pnl_pct", "allocation"]].copy()
        for column in ["daily_return", "unrealized_pnl_pct", "allocation"]:
            daily_shown[column] = daily_shown[column].map(percent_value)
        for column in ["daily_pnl", "unrealized_pnl"]:
            daily_shown[column] = daily_shown[column].map(_money_value)
        st.dataframe(daily_shown, width="stretch", hide_index=True)


def render_interactive_stock_tab(snapshot: dict, features: pd.DataFrame) -> None:
    stock_details = snapshot.get("stock_details", [])
    tickers = _all_research_tickers(snapshot, features)
    if not tickers and features.empty:
        render_missing_results()
        return

    config = load_dashboard_config()
    data_config = config.get("data", {})
    start = data_config.get("start", "2015-01-01")
    end = next_yfinance_end_date()

    st.subheader("动态股票查询")
    st.caption("可以输入美股、ETF、港股或A股 Yahoo Finance 代码。例：NVDA、AVGO、IREN、MRVL、0700.HK、301321。")
    col1, col2, col3 = st.columns([1.4, 2.0, 0.7])
    selected = col1.selectbox("已有股票池", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    query = normalize_ticker(col2.text_input("直接查询任意股票", value="", placeholder="输入后点击刷新，例如 301321 或 PLTR"))
    if col3.button("实时刷新", key="personal_stock_refresh"):
        load_external_ticker_features.clear()
        st.cache_data.clear()

    ticker = query or selected
    detail = _detail_for_ticker(snapshot, ticker)
    try:
        stock, source_label = _stock_frame_for_ticker(ticker, features, start, end)
    except Exception as exc:
        st.error(f"无法下载 {ticker}：{exc}")
        st.caption("如果是A股，系统会自动尝试 .SZ/.SS；如果是港股请使用 0700.HK；如果数据商限流，请稍后再点实时刷新。")
        return
    if stock.empty:
        st.error(f"无法取得 {ticker} 的日K数据。A股会自动尝试 .SZ/.SS；港股请用 0700.HK 这类格式。")
        return

    latest = stock.dropna(subset=["adjusted_close"]).iloc[-1]
    returns = pd.to_numeric(stock["daily_return"], errors="coerce").dropna()
    max_dd, _ = maximum_drawdown(returns)
    price = _first_number(detail.get("price"), latest.get("close"), latest.get("adjusted_close"))
    valuation = detail.get("valuation") if isinstance(detail.get("valuation"), dict) else {}
    portfolio = detail.get("portfolio") if isinstance(detail.get("portfolio"), dict) else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("现价", _money_value(price))
    c2.metric("综合评分", _ratio_value(detail.get("score")))
    c3.metric("21日涨跌", percent_value(latest.get("return_21d")))
    c4.metric("年化波动", percent_value(latest.get("volatility_21d")))
    c5.metric("最大回撤", percent_value(max_dd))
    st.caption(f"数据源：{source_label}；日期：{stock['date'].min().date()} 至 {stock['date'].max().date()}。")

    if portfolio:
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("持仓数量", _ratio_value(portfolio.get("quantity")))
        pc2.metric("IBKR成本", _money_value(portfolio.get("avg_cost")))
        pc3.metric("持仓市值", _money_value(portfolio.get("market_value")))
        pc4.metric("浮动盈亏", _money_value(portfolio.get("unrealized_pnl")), percent_value(portfolio.get("unrealized_pnl_pct")))

    render_stock_action_panel(ticker, detail, stock)
    render_stock_charts(ticker, stock)

    cols = st.columns(3)
    with cols[0]:
        st.subheader("基本面")
        st.dataframe(_fundamental_table(latest), width="stretch", hide_index=True)
        render_factor_scores(detail)
    with cols[1]:
        st.subheader("2027E 情景估值")
        render_valuation_block(valuation)
    with cols[2]:
        st.subheader("支撑压力")
        render_levels_block(detail, stock)

    st.subheader("单股历史回测 vs SPY")
    render_single_stock_backtest(ticker, stock, features)

    with st.expander("投资逻辑 / 财报 / 消息面", expanded=False):
        render_stock_research_notes(snapshot, ticker, detail)


def _all_research_tickers(snapshot: dict, features: pd.DataFrame) -> list[str]:
    tickers: set[str] = set()
    for key in ["portfolio", "scores", "stock_details", "theme_candidates", "earnings_calendar", "earnings_reviews"]:
        for item in snapshot.get(key, []) or []:
            ticker = str(item.get("ticker", "")).upper().strip()
            if ticker:
                tickers.add(ticker)
    for group in snapshot.get("watchlists", []) or []:
        tickers.update(str(ticker).upper().strip() for ticker in group.get("tickers", []) if ticker)
    if not features.empty and "ticker" in features:
        tickers.update(str(ticker).upper() for ticker in features["ticker"].dropna().unique())
    return sorted(tickers)


def _detail_for_ticker(snapshot: dict, ticker: str) -> dict:
    normalized = normalize_ticker(ticker)
    aliases = {normalized, normalized.replace(".SZ", ""), normalized.replace(".SS", ""), normalized.replace(".HK", "")}
    for item in snapshot.get("stock_details", []) or []:
        item_ticker = str(item.get("ticker", "")).upper()
        item_code = str(item.get("code", "")).upper().replace("US.", "")
        if item_ticker in aliases or item_code in aliases:
            return item
    return {"ticker": normalized, "name": normalized}


def _stock_frame_for_ticker(ticker: str, features: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, str]:
    normalized = normalize_ticker(ticker)
    if not features.empty and "ticker" in features:
        aliases = {normalized, normalized.replace(".SZ", ""), normalized.replace(".SS", ""), normalized.replace(".HK", "")}
        local = features[features["ticker"].astype(str).str.upper().isin(aliases)].copy()
        if not local.empty:
            return local.sort_values("date").reset_index(drop=True), "本地处理数据集"
    try:
        external = load_external_ticker_features(normalized, start=start, end=end)
    except Exception:
        if re.fullmatch(r"\d{6}", normalized):
            suffix = ".SS" if normalized.endswith(".SZ") else ".SZ"
            external = load_external_ticker_features(normalized[:6] + suffix, start=start, end=end)
        else:
            raise
    return external.sort_values("date").reset_index(drop=True), "实时下载 yfinance 日K"


def render_stock_action_panel(ticker: str, detail: dict, stock: pd.DataFrame) -> None:
    latest = stock.dropna(subset=["adjusted_close"]).iloc[-1]
    price = _first_number(detail.get("price"), latest.get("close"), latest.get("adjusted_close"))
    valuation = detail.get("valuation") if isinstance(detail.get("valuation"), dict) else {}
    levels = _levels_from_detail_or_stock(detail, stock)
    decision, problem, reason = _stock_decision(price, valuation, latest, levels, detail)
    buy_zone = _format_level(levels.get("buy_zone"))
    sell_zone = _format_level(levels.get("sell_zone"))
    stop_zone = _format_level(levels.get("stop_zone"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新操作结论", decision)
    c2.metric("主要问题来源", problem)
    c3.metric("合理买入区", buy_zone)
    c4.metric("减仓/卖出观察区", sell_zone)
    st.info(f"{ticker}：{reason} 失效/止损观察：{stop_zone}。")


def render_stock_charts(ticker: str, stock: pd.DataFrame) -> None:
    stock = stock.sort_values("date").tail(260).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock["date"], open=stock["open"], high=stock["high"], low=stock["low"], close=stock["close"], name="日K"))
    for column, name, color in [("sma_20", "SMA20", "#0b766d"), ("sma_60", "SMA60", "#b88324"), ("ema_20", "EMA20", "#315bdc")]:
        if column in stock:
            fig.add_trace(go.Scatter(x=stock["date"], y=stock[column], mode="lines", name=name, line=dict(color=color, width=1.5)))
    if {"bb_upper", "bb_lower"}.issubset(stock.columns):
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["bb_upper"], mode="lines", name="Bollinger上轨", line=dict(color="#9ca3af", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=stock["date"], y=stock["bb_lower"], mode="lines", name="Bollinger下轨", line=dict(color="#9ca3af", width=1, dash="dot")))
    st.plotly_chart(style_figure(fig, f"{ticker} 日K / 均线 / Bollinger", height=520), width="stretch")

    ind_fig = go.Figure()
    for column, name in [("rsi_14", "RSI14"), ("macd", "MACD"), ("macd_signal", "MACD Signal"), ("macd_histogram", "MACD柱"), ("zscore_20", "Bollinger Z-score")]:
        if column in stock:
            ind_fig.add_trace(go.Scatter(x=stock["date"], y=stock[column], mode="lines", name=name))
    st.plotly_chart(style_figure(ind_fig, f"{ticker} MACD / RSI / Z-score", height=360), width="stretch")
    render_signal_card_cn(generate_technical_summary(stock.dropna(subset=["adjusted_close"]).iloc[-1]))


def render_signal_card_cn(summary) -> None:
    stance_map = {"Bullish": "偏强", "Bearish": "偏弱", "Neutral": "中性"}
    bullet_map = {
        "Trend:": "趋势：",
        "Momentum:": "动量：",
        "RSI:": "RSI：",
        "MACD:": "MACD：",
        "Bollinger/Z-score:": "Bollinger/Z-score：",
        "Risk:": "风险：",
    }
    st.subheader("技术面自然语言结论")
    st.write(f"**综合判断：{stance_map.get(summary.stance, summary.stance)}**")
    for bullet in summary.bullets:
        text = bullet
        for source, target in bullet_map.items():
            text = text.replace(source, target)
        text = (
            text.replace("price is above SMA20 and SMA60, with a bullish short-to-medium-term moving-average stack.", "价格位于SMA20和SMA60上方，短中期均线结构偏强。")
            .replace("price is below SMA20 and SMA60, with a bearish short-to-medium-term moving-average stack.", "价格位于SMA20和SMA60下方，短中期均线结构偏弱。")
            .replace("the fast line is above the signal line, confirming positive short-term momentum.", "快线在信号线上方，短线动量偏正面。")
            .replace("the fast line is below the signal line, confirming negative short-term momentum.", "快线在信号线下方，短线动量偏负面。")
            .replace("in overbought territory; chasing strength carries higher short-term risk.", "处于超买区，追高风险上升。")
            .replace("in oversold territory; this can be a mean-reversion watch point.", "处于超卖区，可以作为均值回归观察点。")
            .replace("position sizing should match the volatility regime.", "仓位需要和当前波动状态匹配。")
        )
        st.write(f"- {text}")


def _fundamental_table(latest: pd.Series) -> pd.DataFrame:
    rows = [
        ("PE", latest.get("pe_ratio"), "估值倍数，越高越需要成长兑现支撑。"),
        ("PB", latest.get("pb_ratio"), "市净率，适合辅助判断资产定价。"),
        ("ROE", latest.get("roe"), "净资产收益率，反映盈利质量。"),
        ("Revenue Growth", latest.get("revenue_growth"), "收入增速，衡量成长兑现。"),
        ("Market Cap", latest.get("market_cap"), "市值规模，影响波动、流动性和因子暴露。"),
    ]
    frame = pd.DataFrame(rows, columns=["指标", "数值", "中文解释"])
    frame["数值"] = frame.apply(lambda row: _money_value(row["数值"]) if row["指标"] in {"Market Cap"} else _ratio_or_number(row["数值"]), axis=1)
    return frame


def render_factor_scores(detail: dict) -> None:
    scores = detail.get("factor_scores") if isinstance(detail.get("factor_scores"), dict) else {}
    if not scores:
        st.caption("该股票暂无本地多因子评分。")
        return
    frame = pd.DataFrame([{"因子": key, "得分": value} for key, value in scores.items()])
    st.dataframe(frame, width="stretch", hide_index=True)


def render_valuation_block(valuation: dict) -> None:
    if not valuation:
        st.warning("暂无可核验 2027E EPS/FCF 或情景估值输入，暂不输出合理价。")
        return
    st.metric("概率加权合理价", _money_value(valuation.get("weighted_fair_value")), percent_value(valuation.get("upside_to_price")))
    scenarios = pd.DataFrame(valuation.get("scenarios", []))
    if not scenarios.empty:
        shown = scenarios.copy()
        if "fair_value" in shown:
            shown["fair_value"] = shown["fair_value"].map(_money_value)
        if "probability" in shown:
            shown["probability"] = shown["probability"].map(percent_value)
        st.dataframe(shown.rename(columns={"name": "情景", "fair_value": "合理价", "probability": "概率", "driver": "驱动假设"}), width="stretch", hide_index=True)


def render_levels_block(detail: dict, stock: pd.DataFrame) -> None:
    levels = _levels_from_detail_or_stock(detail, stock)
    st.write(f"**买入观察区：** {_format_level(levels.get('buy_zone'))}")
    st.write(f"**减仓观察区：** {_format_level(levels.get('sell_zone'))}")
    st.write(f"**止损/失效区：** {_format_level(levels.get('stop_zone'))}")
    supports = pd.DataFrame(levels.get("supports", []))
    resistances = pd.DataFrame(levels.get("resistances", []))
    if not supports.empty:
        st.caption("支撑位")
        st.dataframe(supports, width="stretch", hide_index=True)
    if not resistances.empty:
        st.caption("压力位")
        st.dataframe(resistances, width="stretch", hide_index=True)


def render_single_stock_backtest(ticker: str, stock: pd.DataFrame, features: pd.DataFrame) -> None:
    frame = stock[["date", "daily_return"]].dropna().copy()
    if frame.empty:
        st.info("该股票暂缺日收益率，无法回测。")
        return
    benchmark = features[features["ticker"].eq("SPY")][["date", "daily_return"]].dropna().copy() if not features.empty else pd.DataFrame()
    benchmark_returns = benchmark.set_index("date")["daily_return"] if not benchmark.empty else pd.Series(dtype=float)
    returns = frame.set_index("date")["daily_return"]
    summary = performance_summary(returns, benchmark_returns=benchmark_returns.reindex(returns.index))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("买入持有收益", percent_value(summary.get("cumulative_return")))
    c2.metric("年化收益", percent_value(summary.get("annualized_return")))
    c3.metric("Sharpe", _ratio_value(summary.get("sharpe_ratio")))
    c4.metric("最大回撤", percent_value(summary.get("maximum_drawdown")))

    chart = pd.DataFrame({"date": returns.index, ticker: (1 + returns).cumprod()})
    if not benchmark_returns.empty:
        aligned = benchmark_returns.reindex(returns.index).fillna(0.0)
        chart["SPY"] = (1 + aligned).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart["date"], y=chart[ticker], mode="lines", name=ticker))
    if "SPY" in chart:
        fig.add_trace(go.Scatter(x=chart["date"], y=chart["SPY"], mode="lines", name="SPY"))
    st.plotly_chart(style_figure(fig, f"{ticker} 买入持有回测 vs SPY", height=380), width="stretch")
    st.caption("单股回测为日K买入持有路径，用于理解这只股票的收益、回撤和波动特征，不代表未来表现。")


def render_stock_research_notes(snapshot: dict, ticker: str, detail: dict) -> None:
    thesis = detail.get("thesis") if isinstance(detail.get("thesis"), dict) else {}
    if thesis:
        st.subheader("投资逻辑")
        st.write(thesis.get("thesis", ""))
        for label, key in [("关注原因", "why_follow"), ("支持证据", "supporting_evidence"), ("关键风险", "key_risks"), ("触发失效条件", "triggered_conditions")]:
            values = thesis.get(key, [])
            if values:
                st.write(f"**{label}**")
                for value in values:
                    st.write(f"- {value}")
    for key, label in [("catalysts", "催化剂"), ("smart_money", "聪明钱/机构行为"), ("sentiment", "消息与情绪")]:
        value = detail.get(key)
        if value:
            st.write(f"**{label}**")
            st.json(value, expanded=False)

    earnings = [item for item in snapshot.get("earnings_reviews", []) or [] if str(item.get("ticker", "")).upper() == ticker.upper()]
    if earnings:
        st.subheader("财报解析")
        for item in earnings:
            st.write(f"**{item.get('date', '')}：{item.get('verdict', '')}**")
            st.write(item.get("investment_value", ""))
            for point in item.get("key_points", [])[:6]:
                st.write(f"- {point}")


def _levels_from_detail_or_stock(detail: dict, stock: pd.DataFrame) -> dict:
    supports = detail.get("supports") if isinstance(detail.get("supports"), list) else []
    resistances = detail.get("resistances") if isinstance(detail.get("resistances"), list) else []
    latest = stock.dropna(subset=["close"]).iloc[-1]
    price = _first_number(detail.get("price"), latest.get("close"))
    if not supports or not resistances:
        tail = stock.sort_values("date").tail(120)
        support_candidates = [
            ("20日低", tail["low"].tail(20).min()),
            ("60日低", tail["low"].tail(60).min()),
            ("SMA20", latest.get("sma_20")),
            ("SMA60", latest.get("sma_60")),
            ("Bollinger下轨", latest.get("bb_lower")),
        ]
        resistance_candidates = [
            ("20日高", tail["high"].tail(20).max()),
            ("60日高", tail["high"].tail(60).max()),
            ("SMA20", latest.get("sma_20")),
            ("SMA60", latest.get("sma_60")),
            ("Bollinger上轨", latest.get("bb_upper")),
        ]
        supports = _level_rows(price, support_candidates, below=True)
        resistances = _level_rows(price, resistance_candidates, below=False)
    buy = _nearest_level(price, supports, prefer_below=True)
    sell = _nearest_level(price, resistances, prefer_below=False)
    stop = None
    if buy:
        stop = {"low": float(buy["low"]) * 0.95, "high": float(buy["low"]) * 0.98, "labels": ["跌破主要支撑后观察"]}
    return {"supports": supports, "resistances": resistances, "buy_zone": buy, "sell_zone": sell, "stop_zone": stop}


def _level_rows(price: float | None, candidates: list[tuple[str, object]], *, below: bool) -> list[dict]:
    rows = []
    for label, value in candidates:
        number = _first_number(value)
        if number is None or price is None:
            continue
        if below and number > price * 1.03:
            continue
        if not below and number < price * 0.97:
            continue
        rows.append({"low": round(number * 0.99, 2), "high": round(number * 1.01, 2), "center": round(number, 2), "strength": "技术位", "labels": [label]})
    return sorted(rows, key=lambda row: abs(float(row["center"]) - float(price or row["center"])))[:5]


def _nearest_level(price: float | None, levels: list[dict], *, prefer_below: bool) -> dict | None:
    if price is None or not levels:
        return levels[0] if levels else None
    candidates = []
    for level in levels:
        center = _first_number(level.get("center"), level.get("low"), level.get("high"))
        if center is None:
            continue
        if prefer_below and center <= price * 1.02:
            candidates.append((abs(center - price), level))
        elif not prefer_below and center >= price * 0.98:
            candidates.append((abs(center - price), level))
    if not candidates:
        return levels[0]
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _stock_decision(price: float | None, valuation: dict, latest: pd.Series, levels: dict, detail: dict) -> tuple[str, str, str]:
    upside = _first_number(valuation.get("upside_to_price")) if valuation else None
    rsi = _first_number(latest.get("rsi_14"))
    zscore = _first_number(latest.get("zscore_20"))
    vol = _first_number(latest.get("volatility_21d"))
    score = _first_number(detail.get("score"))
    problem = "综合"
    if rsi is not None and rsi >= 70:
        problem = "技术面"
        return "不追高/等回落", problem, f"RSI {rsi:.1f} 已偏热，短线先等买点靠近支撑。"
    if zscore is not None and zscore >= 2:
        problem = "技术面"
        return "偏离过大/等回踩", problem, f"价格相对20日均值偏离较大，追涨性价比下降。"
    if upside is not None and upside < -0.10:
        problem = "估值面"
        return "估值偏贵/反弹减仓", problem, f"2027E 概率加权合理价低于现价约 {percent_value(upside)}，需要更高业绩兑现才能支撑。"
    if upside is not None and upside > 0.20 and (score is None or score >= 65):
        problem = "买点/仓位"
        return "可研究低吸", problem, f"估值空间为 {percent_value(upside)}，基本面/因子证据较好，但仍应贴近支撑分批。"
    if vol is not None and vol > 0.65:
        problem = "风险面"
        return "小仓观察", problem, f"年化波动约 {percent_value(vol)}，即使逻辑不错也应控制仓位。"
    return "持有/继续观察", problem, "没有出现明确估值失效或技术破位，重点等更好的买卖点。"


def _format_level(level: dict | None) -> str:
    if not level:
        return "NA"
    low = _first_number(level.get("low"), level.get("center"))
    high = _first_number(level.get("high"), level.get("center"))
    if low is None or high is None:
        return "NA"
    return f"${low:,.2f}-${high:,.2f}"


def _first_number(*values: object) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _ratio_or_number(value: object) -> str:
    number = _first_number(value)
    if number is None:
        return "NA"
    if abs(number) <= 1:
        return percent_value(number)
    return f"{number:,.2f}"


def render_personal_ranking_tab(snapshot: dict) -> None:
    st.subheader("股票综合排名")
    scores = pd.DataFrame(snapshot.get("scores", []))
    details = {str(item.get("ticker", "")).upper(): item for item in snapshot.get("stock_details", []) or []}
    moomoo_rows = {normalize_ticker(str(item.get("ticker") or item.get("code") or "")): item for item in snapshot.get("moomoo_watchlist_rows", []) or []}
    score_by_ticker = {str(item.get("ticker", "")).upper(): item for item in snapshot.get("scores", []) or []}
    all_tickers = sorted(set(score_by_ticker) | set(details) | set(moomoo_rows) | set(moomoo_watchlist_tickers(snapshot)))
    if not all_tickers:
        st.info("暂无排名或自选股数据。请先运行每日投研刷新，或检查 Moomoo 自选股导出文件。")
        return

    query = st.text_input("筛选股票/解释", placeholder="例如 NVDA、半导体、growth、估值", key="ranking_filter")
    rows = []
    for ticker in all_tickers:
        item = score_by_ticker.get(ticker, {})
        detail = details.get(ticker, {})
        moomoo = moomoo_rows.get(ticker, {})
        factor_scores = item.get("factor_scores", {})
        if not isinstance(factor_scores, dict):
            factor_scores = {}
        total_score = item.get("total_score")
        rows.append(
            {
                "排名": np.nan,
                "股票": ticker,
                "公司": detail.get("name") or moomoo.get("name", ""),
                "数据状态": "已评分" if total_score is not None else "待评分/缺因子",
                "总分": total_score,
                "成长": factor_scores.get("growth"),
                "质量": factor_scores.get("quality"),
                "动量": factor_scores.get("momentum"),
                "估值": factor_scores.get("value"),
                "风险": factor_scores.get("risk"),
                "现价": detail.get("price") or moomoo.get("last_price"),
                "2027E合理价": (detail.get("valuation") or {}).get("weighted_fair_value") if isinstance(detail.get("valuation"), dict) else np.nan,
                "估值空间": (detail.get("valuation") or {}).get("upside_to_price") if isinstance(detail.get("valuation"), dict) else np.nan,
                "Moomoo分组": ", ".join(moomoo.get("groups", [])) if isinstance(moomoo.get("groups"), list) else "",
                "解释": "；".join(item.get("explanation", [])) if isinstance(item.get("explanation"), list) else item.get("explanation", "") or str(moomoo.get("action", "")),
            }
        )
    shown = pd.DataFrame(rows)
    shown = shown.sort_values(["总分", "股票"], ascending=[False, True], na_position="last").reset_index(drop=True)
    shown["排名"] = np.where(shown["总分"].notna(), np.arange(1, len(shown) + 1), np.nan)
    if query:
        mask = shown.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False, na=False)).any(axis=1)
        shown = shown[mask]
    for column in ["估值空间"]:
        shown[column] = shown[column].map(percent_value)
    for column in ["现价", "2027E合理价"]:
        shown[column] = shown[column].map(_money_value)
    st.dataframe(shown, width="stretch", hide_index=True)
    st.caption("表格已合并 Moomoo 自选股、股票详情和多因子评分。有些自选股显示“待评分/缺因子”，代表已在自选池里，但还缺本地日K、财务因子或估值输入。")


def render_personal_opportunity_radar_tab(snapshot: dict) -> None:
    st.subheader("潜力股雷达")
    themes = pd.DataFrame(snapshot.get("theme_candidates", []))
    if themes.empty:
        st.info("暂无潜力股雷达数据。")
        return
    query = st.text_input("筛选主题/股票/证据", placeholder="例如 AI Cloud、癌症疫苗、HBM、机器人", key="opportunity_filter")
    shown = themes.copy()
    if query:
        mask = shown.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False, na=False)).any(axis=1)
        shown = shown[mask]
    columns = ["ticker", "company", "theme", "potential_score", "priority", "history_tag", "first_seen", "prior_score", "next_check", "evidence"]
    shown = shown[[column for column in columns if column in shown.columns]].rename(
        columns={
            "ticker": "股票",
            "company": "公司",
            "theme": "主题",
            "potential_score": "潜力分",
            "priority": "优先级",
            "history_tag": "历史记录",
            "first_seen": "首次发现",
            "prior_score": "此前评分",
            "next_check": "下一步核验",
            "evidence": "证据",
        }
    )
    st.dataframe(shown, width="stretch", hide_index=True)

    if shown.empty or "股票" not in shown:
        st.info("没有匹配的潜力股。")
        return
    selected = st.selectbox("查看雷达详情", shown["股票"].tolist())
    item = next((row for row in snapshot.get("theme_candidates", []) or [] if str(row.get("ticker", "")).upper() == selected), None)
    if item:
        left, right = st.columns(2)
        with left:
            st.write("**为什么进入雷达**")
            st.write(item.get("evidence", ""))
            st.write("**搜索路径**")
            for lane in item.get("search_lanes", []):
                st.write(f"- {lane}")
        with right:
            st.write("**证据门槛**")
            for gate in item.get("evidence_gates", []):
                st.write(f"- {gate}")
            st.write("**淘汰规则**")
            for rule in item.get("reject_rules", []):
                st.write(f"- {rule}")


def render_personal_earnings_detail_tab(snapshot: dict) -> None:
    st.subheader("财报日历与详细解析")
    calendar = pd.DataFrame(snapshot.get("earnings_calendar", []))
    reviews = snapshot.get("earnings_reviews", []) or []
    query = st.text_input("筛选财报股票", placeholder="例如 IREN、MRVL、NVDA", key="earnings_filter")

    if not calendar.empty:
        cal = calendar.copy()
        if query:
            mask = cal.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False, na=False)).any(axis=1)
            cal = cal[mask]
        st.write("**近期财报日历**")
        st.dataframe(cal, width="stretch", hide_index=True)
    else:
        st.info("暂无财报日历。")

    filtered_reviews = reviews
    if query:
        filtered_reviews = [item for item in reviews if query.upper() in str(item.get("ticker", "")).upper() or query.lower() in str(item).lower()]
    if not filtered_reviews:
        st.info("暂无匹配的已解析财报。")
        return

    st.write("**已解析财报详情**")
    for item in filtered_reviews:
        ticker = str(item.get("ticker", "")).upper()
        title = f"{ticker} · {item.get('date', '')} · {item.get('verdict', '')}"
        with st.expander(title, expanded=ticker in {"IREN", "MRVL"}):
            c1, c2, c3 = st.columns(3)
            c1.metric("状态", str(item.get("status", "NA")))
            c2.metric("质量分", _ratio_value(item.get("quality_score")))
            c3.metric("证据等级", str(item.get("evidence_grade", "NA")))
            st.write("**投资价值**")
            st.write(item.get("investment_value", ""))
            st.write("**操作建议**")
            st.write(item.get("action", ""))
            metrics = pd.DataFrame(item.get("reported_metrics", []))
            if not metrics.empty:
                st.write("**核心财报数据**")
                st.dataframe(metrics, width="stretch", hide_index=True)
            for label, key in [("关键结论", "key_points"), ("风险点", "red_flags"), ("电话会/管理层表述", "call_highlights"), ("下一步跟踪", "next_steps")]:
                values = item.get(key, [])
                if values:
                    st.write(f"**{label}**")
                    for value in values:
                        st.write(f"- {value}")
            docs = item.get("source_documents", [])
            if docs:
                st.write("**来源文件/链接**")
                for doc in docs:
                    st.write(f"- {doc}")


def render_dynamic_watchlist_backtest_tab(snapshot: dict, features: pd.DataFrame) -> None:
    st.subheader("自选股组合动态回测")
    if features.empty:
        render_missing_results()
        return
    watchlists = snapshot.get("watchlists", []) or []
    group_names = [str(group.get("name", f"自选组{idx + 1}")) for idx, group in enumerate(watchlists)]
    group_choice = st.selectbox("选择自选股分组", ["全部自选股", *group_names], key="watchlist_group")
    if group_choice == "全部自选股":
        base_tickers = sorted({normalize_ticker(str(t)) for group in watchlists for t in group.get("tickers", [])} | set(moomoo_watchlist_tickers(snapshot)))
    else:
        group = watchlists[group_names.index(group_choice)] if group_choice in group_names else {}
        base_tickers = sorted({normalize_ticker(str(t)) for t in group.get("tickers", [])})
    available = sorted(features["ticker"].astype(str).str.upper().dropna().unique())
    extra_text = st.text_input("额外加入股票", placeholder="例如 PLTR, TSM, 301321", key="watch_extra_tickers")
    extra_tickers = [normalize_ticker(item) for item in re.split(r"[,，\\s]+", extra_text) if item.strip()]
    base_tickers = sorted(set(base_tickers) | set(extra_tickers))
    option_tickers = sorted(set(base_tickers) | set(available))
    local_defaults = [ticker for ticker in base_tickers if ticker in set(available)]
    defaults = local_defaults[: min(24, len(local_defaults))] or [ticker for ticker in ["NVDA", "AVGO", "MSFT", "GOOGL", "AMZN"] if ticker in available]
    selected = st.multiselect("选择回测股票", option_tickers, default=defaults)
    if not selected:
        st.warning("请至少选择一只股票。")
        return
    selected_missing = [ticker for ticker in selected if ticker not in set(available)]
    live_fill = st.checkbox("所选股票缺少本地日K时，尝试实时下载后纳入回测", value=True, key="watch_live_fill")
    working_features = features
    if selected_missing and not live_fill:
        st.warning(f"这些已选择股票缺少本地日K，当前不会进入回测：{', '.join(selected_missing[:30])}")
    if live_fill and selected_missing:
        live_frames = []
        with st.spinner(f"正在补齐 {min(len(selected_missing), 10)} 只缺失自选股的日K..."):
            for ticker in selected_missing[:10]:
                try:
                    live, _ = _stock_frame_for_ticker(ticker, features, "2015-01-01", next_yfinance_end_date())
                    if not live.empty:
                        live_frames.append(live)
                except Exception as exc:
                    st.caption(f"{ticker} 实时补齐失败：{exc}")
        if live_frames:
            working_features = pd.concat([features, *live_frames], ignore_index=True)
            available = sorted(working_features["ticker"].astype(str).str.upper().dropna().unique())

    c1, c2 = st.columns([1.2, 1.0])
    range_choice = c1.selectbox("回测区间", ["最近1年", "最近6个月", "最近3个月", "最近1个月", "最近2年", "全部", "自定义"], key="watch_range_choice")
    mode = c2.radio("权重方式", ["等权", "自定义"], horizontal=True, key="watch_weight_mode")
    data_min = pd.to_datetime(working_features["date"]).min().normalize()
    data_max = pd.to_datetime(working_features["date"]).max().normalize()
    if range_choice == "自定义":
        c3, c4 = st.columns(2)
        default_start = max(data_min, data_max - pd.DateOffset(years=1))
        start_date = c3.date_input("开始日期", value=default_start.date(), min_value=data_min.date(), max_value=data_max.date(), key="watch_backtest_start")
        end_date = c4.date_input("结束日期", value=data_max.date(), min_value=data_min.date(), max_value=data_max.date(), key="watch_backtest_end")
    else:
        offsets = {
            "最近1个月": pd.DateOffset(months=1),
            "最近3个月": pd.DateOffset(months=3),
            "最近6个月": pd.DateOffset(months=6),
            "最近1年": pd.DateOffset(years=1),
            "最近2年": pd.DateOffset(years=2),
        }
        start_ts = data_min if range_choice == "全部" else max(data_min, data_max - offsets[range_choice])
        start_date = start_ts.date()
        end_date = data_max.date()
        st.caption(f"当前回测区间：{start_date} 至 {end_date}。要选更短或任意日期，请把区间切到“自定义”。")
    weights = {}
    if mode == "自定义":
        st.caption("自定义权重会自动归一化，不需要刚好加总到100%。")
        columns = st.columns(min(4, len(selected)))
        for idx, ticker in enumerate(selected):
            weights[ticker] = columns[idx % len(columns)].number_input(f"{ticker} 权重", min_value=0.0, max_value=100.0, value=round(100 / len(selected), 1), step=1.0, key=f"watch_weight_{ticker}")
    else:
        weights = {ticker: 1.0 for ticker in selected}

    result = _buy_hold_portfolio_backtest(working_features, selected, weights, pd.Timestamp(start_date), pd.Timestamp(end_date))
    if result.empty:
        st.warning("所选股票在该区间没有足够数据。")
        return
    benchmark = working_features[working_features["ticker"].eq("SPY")][["date", "daily_return"]].dropna().copy()
    returns = result.set_index("date")["daily_return"]
    benchmark_returns = benchmark.set_index("date")["daily_return"].reindex(returns.index) if not benchmark.empty else pd.Series(dtype=float)
    summary = performance_summary(returns, benchmark_returns=benchmark_returns)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("累计收益", percent_value(summary.get("cumulative_return")))
    c2.metric("年化收益", percent_value(summary.get("annualized_return")))
    c3.metric("年化波动", percent_value(summary.get("annualized_volatility")))
    c4.metric("Sharpe", _ratio_value(summary.get("sharpe_ratio")))
    c5.metric("最大回撤", percent_value(summary.get("maximum_drawdown")))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["date"], y=result["portfolio"], mode="lines", name="自选组合"))
    if "SPY" in result:
        fig.add_trace(go.Scatter(x=result["date"], y=result["SPY"], mode="lines", name="SPY"))
    st.plotly_chart(style_figure(fig, "自选股组合净值 vs SPY", height=420), width="stretch")
    st.dataframe(_contribution_table(working_features, selected, weights, pd.Timestamp(start_date), pd.Timestamp(end_date)), width="stretch", hide_index=True)
    missing = sorted(set(selected) - set(available))
    if missing:
        st.warning(f"这些已选自选股仍缺少可用日K，未进入回测：{', '.join(missing[:20])}")


def _buy_hold_portfolio_backtest(features: pd.DataFrame, tickers: list[str], weights: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = features[features["ticker"].isin(tickers)].copy()
    data = data[(data["date"] >= start) & (data["date"] <= end)]
    if data.empty:
        return pd.DataFrame()
    weight_series = pd.Series({ticker: float(weights.get(ticker, 0.0)) for ticker in tickers}, dtype=float)
    if weight_series.sum() <= 0:
        weight_series[:] = 1.0
    weight_series = weight_series / weight_series.sum()
    returns = data.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="last").sort_index().fillna(0.0)
    returns = returns[[ticker for ticker in weight_series.index if ticker in returns.columns]]
    if returns.empty:
        return pd.DataFrame()
    aligned_weights = weight_series.reindex(returns.columns).fillna(0.0)
    portfolio_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
    result = pd.DataFrame({"date": portfolio_returns.index, "daily_return": portfolio_returns, "portfolio": (1 + portfolio_returns).cumprod()})
    spy = features[features["ticker"].eq("SPY")][["date", "daily_return"]].dropna().copy()
    if not spy.empty:
        spy_returns = spy.set_index("date")["daily_return"].reindex(portfolio_returns.index).fillna(0.0)
        result["SPY"] = (1 + spy_returns).cumprod()
    return result.reset_index(drop=True)


def _contribution_table(features: pd.DataFrame, tickers: list[str], weights: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    weight_series = pd.Series({ticker: float(weights.get(ticker, 0.0)) for ticker in tickers}, dtype=float)
    if weight_series.sum() <= 0:
        weight_series[:] = 1.0
    weight_series = weight_series / weight_series.sum()
    for ticker in tickers:
        data = features[(features["ticker"].eq(ticker)) & (features["date"] >= start) & (features["date"] <= end)].sort_values("date")
        if data.empty:
            continue
        returns = pd.to_numeric(data["daily_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "股票": ticker,
                "权重": percent_value(weight_series.get(ticker, 0.0)),
                "区间收益": percent_value((1 + returns).prod() - 1),
                "年化波动": percent_value(returns.std(ddof=0) * np.sqrt(252)),
                "贡献解释": "高收益且权重高则贡献大；高波动且收益差则拖累组合。",
            }
        )
    return pd.DataFrame(rows)


def render_interactive_strategy_tab(snapshot: dict, features: pd.DataFrame) -> None:
    if features.empty:
        render_missing_results()
        return
    portfolio_tickers = sorted({str(row.get("ticker", "")).upper() for row in snapshot.get("portfolio", [])})
    watchlist_tickers = sorted({str(t).upper() for group in snapshot.get("watchlists", []) for t in group.get("tickers", [])})
    available = sorted(features["ticker"].dropna().unique())
    universe_choice = st.radio("股票池", ["我的持仓", "Moomoo自选", "默认股票池", "手动选择"], horizontal=True)
    if universe_choice == "我的持仓":
        universe = [ticker for ticker in portfolio_tickers if ticker in available]
    elif universe_choice == "Moomoo自选":
        universe = [ticker for ticker in watchlist_tickers if ticker in available][:40]
    elif universe_choice == "手动选择":
        universe = st.multiselect("选择股票", available, default=[ticker for ticker in ["NVDA", "AVGO", "MSFT", "GOOGL", "AMZN"] if ticker in available])
    else:
        universe = [ticker for ticker in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "AMD", "ORCL", "CSCO", "SPY"] if ticker in available]

    c1, c2, c3 = st.columns(3)
    strategy_display = c1.selectbox("策略", ["Momentum", "Mean Reversion", "Multi-Factor"])
    top_n = c2.slider("Top N", 1, max(1, min(20, len(universe) or 1)), min(5, max(1, len(universe) or 1)))
    entry_z = c3.select_slider("均值回归 Z-score", options=[-1.5, -2.0, -2.5], value=-2.0)
    c4, c5, c6 = st.columns(3)
    capital = c4.number_input("初始资金", min_value=1_000.0, max_value=1_000_000.0, value=100_000.0, step=5_000.0)
    cost = c5.number_input("交易成本 bps", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    slippage = c6.number_input("滑点 bps", min_value=0.0, max_value=100.0, value=2.0, step=1.0)

    data = features[features["ticker"].isin(universe)].copy()
    if data.empty:
        st.warning("所选股票池没有可用历史数据。")
        return
    weights = strategy_weights(data, strategy_display, top_n=top_n, entry_z=float(entry_z))
    result = BacktestEngine(initial_capital=capital, transaction_cost_bps=cost, slippage_bps=slippage, signal_lag=1).run(data, weights)
    benchmark = features[features["ticker"].eq("SPY")].set_index("date")["daily_return"]
    summary = performance_summary(result.portfolio_value.set_index("date")["daily_return"], benchmark_returns=benchmark)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", percent_value(summary.get("cumulative_return")))
    c2.metric("年化收益", percent_value(summary.get("annualized_return")))
    c3.metric("Sharpe", _ratio_value(summary.get("sharpe_ratio")))
    c4.metric("最大回撤", percent_value(summary.get("maximum_drawdown")))
    st.caption("回测使用 signal_lag=1，信号至少滞后一日成交，避免 look-ahead bias。")
    st.plotly_chart(line_chart(result.portfolio_value, "date", "total_value", f"{strategy_display} 回测净值"), width="stretch")
    st.dataframe(result.trades.tail(100), width="stretch", hide_index=True)


def render_interactive_risk_factor_tab(snapshot: dict, features: pd.DataFrame) -> None:
    latest = latest_trade_file(PROJECT_ROOT)
    if latest and not features.empty:
        result = reconstruct_from_trades(features, read_trade_file(latest))
        summary = result.summary
        st.subheader("基于交易流水的风险指标")
    else:
        st.subheader("当前持仓快照风险提示")
        summary = pd.Series(dtype=float)
        st.info("没有交易流水时，风险页优先展示已有因子文件和当前持仓集中度。")

    if not summary.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年化波动", percent_value(summary.get("annualized_volatility")))
        c2.metric("Sortino", _ratio_value(summary.get("sortino_ratio")))
        c3.metric("Beta vs SPY", _ratio_value(summary.get("beta_vs_benchmark")))
        c4.metric("Tracking Error", percent_value(summary.get("tracking_error")))
        if latest and not result.contribution.empty:
            st.dataframe(_format_money_percent_frame(result.contribution), width="stretch", hide_index=True)

    exposure = read_csv(file_signature(RESULTS_DIR / "factor_exposure.csv"))
    if not exposure.empty:
        st.subheader("Fama-French 因子暴露")
        st.dataframe(exposure, width="stretch", hide_index=True)

    portfolio = pd.DataFrame(snapshot.get("portfolio", []))
    if not portfolio.empty and {"ticker", "allocation"}.issubset(portfolio.columns):
        st.subheader("集中度")
        st.dataframe(portfolio[["ticker", "sector", "allocation"]].assign(allocation=lambda x: x["allocation"].map(percent_value)), width="stretch", hide_index=True)


def render_personal_derivatives_tab() -> None:
    render_derivatives_lab()


def render_interactive_earnings_opportunity_tab(snapshot: dict) -> None:
    earnings = pd.DataFrame(snapshot.get("earnings_reviews", []))
    calendar = pd.DataFrame(snapshot.get("earnings_calendar", []))
    themes = pd.DataFrame(snapshot.get("theme_candidates", []))

    left, right = st.columns(2)
    with left:
        st.subheader("已解析财报")
        if earnings.empty:
            render_missing_results()
        else:
            columns = ["ticker", "date", "status", "verdict", "quality_score", "thesis_impact", "investment_value", "action"]
            st.dataframe(earnings[[column for column in columns if column in earnings.columns]], width="stretch", hide_index=True)
    with right:
        st.subheader("财报日历")
        if calendar.empty:
            render_missing_results()
        else:
            st.dataframe(calendar, width="stretch", hide_index=True)

    st.subheader("潜力股与主题机会")
    if themes.empty:
        render_missing_results()
    else:
        columns = ["ticker", "company", "theme", "potential_score", "priority", "history_tag", "next_check", "evidence"]
        st.dataframe(themes[[column for column in columns if column in themes.columns]], width="stretch", hide_index=True)


def _format_money_percent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    for column in ["weight", "daily_return", "unrealized_pnl_pct", "allocation"]:
        if column in shown:
            shown[column] = shown[column].map(percent_value)
    for column in [
        "avg_cost",
        "price",
        "market_value",
        "unrealized_pnl",
        "realized_pnl",
        "total_pnl",
        "ledger_market_value",
        "snapshot_market_value",
        "market_value_diff",
        "ledger_avg_cost",
        "ledger_price",
        "snapshot_avg_cost",
        "snapshot_price",
    ]:
        if column in shown:
            shown[column] = shown[column].map(_money_value)
    return shown


def _ledger_snapshot_reconciliation(ledger: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or portfolio.empty or "ticker" not in ledger or "ticker" not in portfolio:
        return pd.DataFrame()
    ledger_frame = ledger[["ticker", "quantity", "market_value", "avg_cost", "price"]].copy()
    ledger_frame = ledger_frame.rename(
        columns={
            "quantity": "ledger_quantity",
            "market_value": "ledger_market_value",
            "avg_cost": "ledger_avg_cost",
            "price": "ledger_price",
        }
    )
    snapshot_columns = [column for column in ["ticker", "quantity", "market_value", "avg_cost", "ibkr_price", "allocation"] if column in portfolio.columns]
    snapshot_frame = portfolio[snapshot_columns].copy()
    snapshot_frame = snapshot_frame.rename(
        columns={
            "quantity": "snapshot_quantity",
            "market_value": "snapshot_market_value",
            "avg_cost": "snapshot_avg_cost",
            "ibkr_price": "snapshot_price",
        }
    )
    merged = pd.merge(ledger_frame, snapshot_frame, on="ticker", how="outer")
    for column in ["ledger_quantity", "snapshot_quantity", "ledger_market_value", "snapshot_market_value"]:
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["quantity_diff"] = merged.get("ledger_quantity", 0.0).fillna(0.0) - merged.get("snapshot_quantity", 0.0).fillna(0.0)
    merged["market_value_diff"] = merged.get("ledger_market_value", 0.0).fillna(0.0) - merged.get("snapshot_market_value", 0.0).fillna(0.0)
    merged["status"] = np.where(
        merged["quantity_diff"].abs().le(1e-6),
        "数量一致",
        "需要核对",
    )
    return merged.sort_values(["status", "ticker"]).reset_index(drop=True)


def _money_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}"


def _ratio_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.3f}" if np.isfinite(number) else "NA"


def _driver_cn(text: str) -> str:
    mapping = {
        "VIX is calm": "VIX较低，市场暂时没有强烈避险定价",
        "Nasdaq is above its long-term trend": "纳指仍在长期趋势线上方，成长股环境未被破坏",
        "Market breadth is supportive": "市场宽度较好，上涨不是只靠极少数龙头",
        "Credit stress is contained": "信用压力可控，流动性风险暂未扩散",
        "Rates are not pressuring duration assets": "利率暂未明显压制高久期成长股估值",
        "No major caution triggered": "暂无重大警报",
    }
    return mapping.get(text, text)


def _reason_cn(text: str) -> str:
    mapping = {
        "Risk-aware AI growth tape: growth and momentum still lead, but risk gets a dedicated 10% weight": "AI主线仍占优，但风险因子被单独提高到10%，成长和动量不能无约束压过波动、回撤和集中度。",
    }
    return mapping.get(text, text)


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
    features = render_project_data_status(features, tickers, start)

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
    render_reproducibility_panel()


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
    render_reproducibility_panel()


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
    module = active_module()
    data_config = load_dashboard_config().get("data", {})
    maybe_auto_refresh_project_data(tickers, data_config.get("start", "2015-01-01"), load_features())
    render_sidebar_nav(module)
    module = active_module()

    pages = {
        "personal": render_personal_quant_platform,
        "trade": render_trade_advice_page,
        "portfolio": render_portfolio_analysis_page,
        "ranking": render_ranking_page,
        "opportunity": render_opportunity_page,
        "watchlist": render_watchlist_backtest_page,
        "strategy": render_strategy_lab_page,
        "stock": render_stock_intelligence_page,
        "risk_center": render_risk_center_page,
        "factor": render_factor_exposure_page,
        "derivatives": render_derivatives_lab,
        "earnings": render_earnings_research_page,
        "data_security": render_data_security_page,
    }
    pages.get(module, render_personal_quant_platform)()


if __name__ == "__main__":
    main()
