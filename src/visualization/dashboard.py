"""Matplotlib and Plotly charts for strategy research reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from risk.risk_metrics import maximum_drawdown


def plot_cumulative_returns(portfolio_value: pd.DataFrame):
    """Plot cumulative return curve with Matplotlib."""

    frame = portfolio_value.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["date"], frame["cumulative_return"], label="Strategy")
    ax.set_title("Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_drawdown(portfolio_value: pd.DataFrame):
    """Plot drawdown curve with Matplotlib."""

    frame = portfolio_value.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    _, drawdown = maximum_drawdown(frame["daily_return"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(frame["date"].iloc[-len(drawdown):], drawdown, 0, alpha=0.35)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    return fig


def plot_performance_comparison(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> go.Figure:
    """Build an interactive strategy versus benchmark cumulative-return chart."""

    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["Strategy", "Benchmark"]
    cumulative = (1 + aligned).cumprod() - 1
    fig = go.Figure()
    for column in cumulative.columns:
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[column], mode="lines", name=column))
    fig.update_layout(title="Strategy vs Benchmark", xaxis_title="Date", yaxis_title="Cumulative Return")
    return fig


def plot_factor_exposures(exposure: pd.DataFrame) -> go.Figure:
    """Build an interactive factor exposure bar chart."""

    frame = exposure.drop(index=["const", "r_squared"], errors="ignore").dropna(subset=["coefficient"])
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame.index,
                y=frame["coefficient"],
                error_y={"type": "data", "array": frame.get("p_value", pd.Series(0, index=frame.index))},
            )
        ]
    )
    fig.update_layout(title="Factor Exposures", xaxis_title="Factor", yaxis_title="Coefficient")
    return fig
