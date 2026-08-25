"""Search catalog for dashboard navigation."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SearchItem:
    """One searchable dashboard item."""

    title: str
    category: str
    description: str
    target: str
    keywords: tuple[str, ...] = ()


def build_search_index(tickers: tuple[str, ...] | list[str] = ()) -> list[SearchItem]:
    """Build a lightweight index for tickers, concepts, outputs, and modules."""

    items = [
        SearchItem(
            "Momentum Strategy",
            "Strategy",
            "Cross-sectional 1/3/6-month momentum ranking with monthly rebalancing.",
            "Backtest Lab",
            ("return_21d", "return_63d", "return_126d", "rebalance"),
        ),
        SearchItem(
            "Mean Reversion Strategy",
            "Strategy",
            "Bollinger Band and Z-score strategy with parameter sensitivity checks.",
            "Backtest Lab",
            ("bollinger", "zscore", "-1.5", "-2.0", "-2.5"),
        ),
        SearchItem(
            "Fama-French Factor Model",
            "Factor Model",
            "Official Kenneth French factors with Newey-West HAC standard errors.",
            "Risk & Factors",
            ("mkt_rf", "smb", "hml", "alpha", "newey-west", "hac"),
        ),
        SearchItem(
            "Sharpe Ratio",
            "Risk Metric",
            "Mean excess return divided by volatility, annualized with sqrt(252).",
            "Risk & Factors",
            ("risk", "return", "volatility"),
        ),
        SearchItem(
            "Maximum Drawdown",
            "Risk Metric",
            "Worst peak-to-trough portfolio loss on the compounded return path.",
            "Risk & Factors",
            ("mdd", "drawdown", "risk"),
        ),
        SearchItem(
            "Black-Scholes",
            "Derivatives",
            "Closed-form European option pricing under geometric Brownian motion.",
            "Derivatives Lab",
            ("option", "call", "put", "sde", "gbm"),
        ),
        SearchItem(
            "Greeks",
            "Derivatives",
            "Delta, gamma, vega, theta, and rho for option risk sensitivities.",
            "Derivatives Lab",
            ("delta", "gamma", "vega", "theta", "rho"),
        ),
        SearchItem(
            "Monte Carlo Variance Reduction",
            "Derivatives",
            "Antithetic and control variates with convergence diagnostics.",
            "Derivatives Lab",
            ("simulation", "antithetic", "control variate", "convergence"),
        ),
        SearchItem(
            "Delta Hedging",
            "Derivatives",
            "Short option replication experiment with transaction costs and hedge-frequency analysis.",
            "Derivatives Lab",
            ("replication", "hedge", "transaction cost"),
        ),
        SearchItem(
            "Machine Learning Baselines",
            "Prediction",
            "Linear regression, random forest, and gradient boosting with chronological validation.",
            "Data & ML",
            ("future_21d_return", "timeseriessplit", "rmse", "r2"),
        ),
        SearchItem(
            "Key Findings",
            "Report",
            "Generated research summary from the current data pull and model settings.",
            "Overview",
            ("results/key_findings.md", "summary"),
        ),
    ]

    for ticker in tickers:
        symbol = str(ticker).upper()
        items.append(
            SearchItem(
                symbol,
                "Ticker",
                f"Open the stock explorer for {symbol}.",
                "Stock Explorer",
                (symbol.lower(),),
            )
        )
    return items


def _tokens(value: str) -> set[str]:
    normalized = value.lower().replace("-", " ")
    return set(re.findall(r"[a-z0-9_.+]+", normalized))


def search_catalog(query: str, catalog: list[SearchItem], limit: int = 12) -> list[SearchItem]:
    """Return ranked search matches for a free-text query."""

    normalized = query.strip().lower()
    if not normalized:
        return catalog[:limit]

    query_tokens = _tokens(normalized)
    scored: list[tuple[int, SearchItem]] = []
    for item in catalog:
        haystack = " ".join((item.title, item.category, item.description, " ".join(item.keywords))).lower()
        score = 0
        if normalized in item.title.lower():
            score += 8
        if normalized in haystack:
            score += 4
        score += 2 * len(query_tokens.intersection(_tokens(haystack)))
        if score:
            scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[0], pair[1].category, pair[1].title))
    return [item for _, item in scored[:limit]]
