"""Private trade-ledger ingestion and portfolio reconstruction.

This module is intentionally data-source agnostic. It can read an exported IBKR
activity statement CSV or a simple local trades.csv, normalize common column
names, and rebuild a daily mark-to-market portfolio from executed trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from risk.risk_metrics import performance_summary


REQUIRED_COLUMNS = ["date", "ticker", "side", "quantity", "price"]
PRIVATE_DATA_DIR = Path("data") / "private"


@dataclass(frozen=True)
class TradeLedgerResult:
    trades: pd.DataFrame
    portfolio_value: pd.DataFrame
    positions: pd.DataFrame
    contribution: pd.DataFrame
    summary: pd.Series
    warnings: list[str]
    source: str


def private_trade_files(project_root: Path) -> list[Path]:
    """Return candidate private trade-ledger files without touching Git data."""

    private_dir = project_root / PRIVATE_DATA_DIR
    if not private_dir.exists():
        return []
    patterns = ["trades*.csv", "activity_statement*.csv", "ibkr_trades*.csv", "executions*.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(private_dir.glob(pattern))
    return sorted(set(files), key=lambda path: path.stat().st_mtime)


def latest_trade_file(project_root: Path) -> Path | None:
    files = private_trade_files(project_root)
    return files[-1] if files else None


def normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common IBKR/simple trade CSV formats to REQUIRED_COLUMNS."""

    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    normalized = frame.copy()
    normalized.columns = [_clean_column(column) for column in normalized.columns]
    aliases = {
        "symbol": "ticker",
        "underlying": "ticker",
        "asset": "ticker",
        "local_symbol": "ticker",
        "code": "ticker",
        "trade_date": "date",
        "transaction_date": "date",
        "datetime": "date",
        "date_time": "date",
        "trade_time": "date",
        "trade_date_time": "date",
        "time": "date",
        "buy_sell": "side",
        "action": "side",
        "transaction_type": "side",
        "type": "side",
        "qty": "quantity",
        "shares": "quantity",
        "amount": "quantity",
        "quantity": "quantity",
        "exec_price": "price",
        "trade_price": "price",
        "fill_price": "price",
        "t_price": "price",
        "transaction_price": "price",
        "price": "price",
        "proceeds": "net_amount",
        "commission": "commission",
        "commissions": "commission",
        "comm_fee": "commission",
        "commissions_fees": "commission",
        "fees": "fees",
    }
    normalized = normalized.rename(columns={old: new for old, new in aliases.items() if old in normalized.columns})
    normalized = _filter_ibkr_trade_rows(normalized)

    if "ticker" not in normalized.columns and "description" in normalized.columns:
        normalized["ticker"] = normalized["description"].astype(str).str.extract(r"\b([A-Z]{1,5}(?:\.[A-Z])?)\b", expand=False)
    if "side" not in normalized.columns and "quantity" in normalized.columns:
        normalized["side"] = np.where(pd.to_numeric(normalized["quantity"], errors="coerce") < 0, "SELL", "BUY")

    missing = [column for column in ["date", "ticker", "quantity", "price"] if column not in normalized.columns]
    if missing:
        raise ValueError(f"trade ledger missing required columns: {', '.join(missing)}")

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.replace("US.", "", regex=False).str.strip()
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["side"] = normalized.get("side", "BUY")
    normalized["side"] = normalized["side"].astype(str).str.upper().map(_normalize_side)
    normalized["quantity"] = normalized["quantity"].abs()

    for optional in ["commission", "fees", "net_amount"]:
        if optional in normalized.columns:
            normalized[optional] = pd.to_numeric(normalized[optional], errors="coerce").fillna(0.0)
        else:
            normalized[optional] = 0.0
    normalized["commission"] = normalized["commission"].abs()
    normalized["fees"] = normalized["fees"].abs()

    normalized = normalized.dropna(subset=["date", "ticker", "side", "quantity", "price"])
    normalized = normalized[(normalized["ticker"] != "") & (normalized["quantity"] > 0) & (normalized["price"] > 0)]
    columns = [*REQUIRED_COLUMNS, "commission", "fees", "net_amount"]
    return normalized[columns].sort_values(["date", "ticker", "side"]).reset_index(drop=True)


def read_trade_file(path: Path) -> pd.DataFrame:
    """Read and normalize a trade file."""

    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    return normalize_trade_frame(frame)


def write_uploaded_trades(project_root: Path, data: bytes, filename: str = "trades_uploaded.csv") -> Path:
    """Persist uploaded trades into the ignored private data directory."""

    private_dir = project_root / PRIVATE_DATA_DIR
    private_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".csv"):
        safe_name = "trades_uploaded.csv"
    target = private_dir / safe_name
    target.write_bytes(data)
    return target


def reconstruct_from_trades(
    features: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
    initial_cash: float | None = None,
) -> TradeLedgerResult:
    """Rebuild a daily portfolio value series from normalized executed trades."""

    normalized = normalize_trade_frame(trades)
    warnings: list[str] = []
    if normalized.empty:
        return TradeLedgerResult(normalized, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), ["没有可用交易流水。"], "empty")

    prices = _price_matrix(features, "close")
    if prices.empty:
        return TradeLedgerResult(normalized, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), ["没有可用价格数据。"], "trade_ledger")

    trade_tickers = sorted(normalized["ticker"].unique())
    missing_tickers = sorted(set(trade_tickers) - set(prices.columns))
    if missing_tickers:
        warnings.append(f"以下交易标的缺少历史价格，已从净值重建中排除：{', '.join(missing_tickers)}")
        normalized = normalized[~normalized["ticker"].isin(missing_tickers)].copy()
    if normalized.empty:
        return TradeLedgerResult(normalized, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), warnings, "trade_ledger")

    start = normalized["date"].min()
    end = prices.index.max()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end), sorted(set(normalized["ticker"]) & set(prices.columns))]
    prices = prices.ffill().dropna(how="all")

    positions = pd.Series(0.0, index=prices.columns)
    avg_cost = pd.Series(0.0, index=prices.columns)
    realized = pd.Series(0.0, index=prices.columns)
    cash = 0.0 if initial_cash is None else float(initial_cash)
    min_cash = cash
    value_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []

    grouped = {date: group for date, group in normalized.groupby("date")}
    for date, price_row in prices.iterrows():
        for _, trade in grouped.get(date.normalize(), pd.DataFrame()).iterrows():
            ticker = trade["ticker"]
            qty = float(trade["quantity"])
            price = float(trade["price"])
            fees = float(trade.get("commission", 0.0)) + float(trade.get("fees", 0.0))
            if ticker not in positions.index:
                continue
            if trade["side"] == "BUY":
                old_qty = positions[ticker]
                new_qty = old_qty + qty
                avg_cost[ticker] = ((avg_cost[ticker] * old_qty) + (price * qty) + fees) / new_qty if new_qty else 0.0
                positions[ticker] = new_qty
                cash -= price * qty + fees
            else:
                sell_qty = min(qty, positions[ticker])
                if sell_qty <= 0:
                    continue
                realized[ticker] += (price - avg_cost[ticker]) * sell_qty - fees
                positions[ticker] -= sell_qty
                cash += price * sell_qty - fees
                if positions[ticker] <= 1e-9:
                    positions[ticker] = 0.0
                    avg_cost[ticker] = 0.0
            min_cash = min(min_cash, cash)

        holdings_value = float((positions * price_row.reindex(positions.index).fillna(0.0)).sum())
        total_value = cash + holdings_value
        value_rows.append({"date": date, "cash": cash, "holdings": holdings_value, "total_value": total_value})
        for ticker in positions.index:
            qty = float(positions[ticker])
            if qty <= 0:
                continue
            price = float(price_row.get(ticker, np.nan))
            if not np.isfinite(price):
                continue
            position_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "quantity": qty,
                    "avg_cost": float(avg_cost[ticker]),
                    "price": price,
                    "market_value": qty * price,
                    "unrealized_pnl": (price - float(avg_cost[ticker])) * qty,
                    "realized_pnl": float(realized[ticker]),
                }
            )

    portfolio_value = pd.DataFrame(value_rows)
    if portfolio_value.empty:
        return TradeLedgerResult(normalized, portfolio_value, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), warnings, "trade_ledger")
    if initial_cash is None and min_cash < 0:
        inferred = abs(min_cash)
        portfolio_value[["cash", "total_value"]] = portfolio_value[["cash", "total_value"]] + inferred
        warnings.append(f"未提供初始现金，系统按交易流水推断最低所需初始现金约 ${inferred:,.2f}。")
    portfolio_value["daily_return"] = portfolio_value["total_value"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    portfolio_value["cumulative_return"] = portfolio_value["total_value"].div(portfolio_value["total_value"].iloc[0]).sub(1.0)
    positions_frame = pd.DataFrame(position_rows)
    contribution = _latest_contribution(positions_frame, portfolio_value)
    benchmark = _returns_for(features, benchmark_ticker).reindex(pd.to_datetime(portfolio_value["date"]))
    summary = performance_summary(portfolio_value.set_index("date")["daily_return"], benchmark_returns=benchmark)
    return TradeLedgerResult(normalized, portfolio_value, positions_frame, contribution, summary, warnings, "trade_ledger")


def _latest_contribution(positions: pd.DataFrame, portfolio_value: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or portfolio_value.empty:
        return pd.DataFrame()
    latest_date = positions["date"].max()
    latest = positions[positions["date"].eq(latest_date)].copy()
    total_value = float(portfolio_value["total_value"].iloc[-1])
    latest["weight"] = latest["market_value"] / total_value if total_value else np.nan
    latest["total_pnl"] = latest["unrealized_pnl"] + latest["realized_pnl"]
    return latest.sort_values("market_value", ascending=False).reset_index(drop=True)


def _price_matrix(features: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    column = price_col if price_col in features.columns else "adjusted_close"
    frame = features[["date", "ticker", column]].dropna().copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame.pivot_table(index="date", columns="ticker", values=column, aggfunc="last").sort_index()


def _returns_for(features: pd.DataFrame, ticker: str) -> pd.Series:
    if features.empty or "daily_return" not in features.columns:
        return pd.Series(dtype=float)
    frame = features[features["ticker"].eq(ticker)][["date", "daily_return"]].dropna().copy()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.set_index("date")["daily_return"].sort_index()


def _clean_column(column: Any) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalize_side(value: str) -> str:
    text = str(value).upper()
    if any(token in text for token in ["SELL", "SLD", "出售", "卖"]):
        return "SELL"
    return "BUY"


def _filter_ibkr_trade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep execution rows when an IBKR sectioned statement is uploaded."""

    filtered = frame.copy()
    if "data_discriminator" in filtered.columns:
        mask = filtered["data_discriminator"].astype(str).str.contains("order|trade", case=False, na=False)
        if mask.any():
            filtered = filtered[mask].copy()
    if "asset_category" in filtered.columns:
        mask = filtered["asset_category"].astype(str).str.contains("stocks|stock|equity|equities", case=False, na=False)
        if mask.any():
            filtered = filtered[mask].copy()
    if "currency" in filtered.columns:
        mask = filtered["currency"].astype(str).str.upper().isin(["USD", ""])
        if mask.any():
            filtered = filtered[mask].copy()
    return filtered
