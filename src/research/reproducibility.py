"""Reproducibility metadata for research runs and backtests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 fingerprint for a research artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_date(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def git_state(project_root: str | Path) -> dict[str, Any]:
    """Return the commit and dirty-state used to generate a run."""

    root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit or None, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_backtest_manifest(
    *,
    config: dict[str, Any],
    features: pd.DataFrame,
    artifact_paths: Iterable[str | Path] = (),
    project_root: str | Path = ".",
    generated_at: datetime | None = None,
    git_commit: str | None = None,
    backtest_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable description of the data and assumptions used.

    The manifest deliberately records limitations that affect interpretation:
    the current yfinance fundamentals are snapshots, and the default universe
    is not point-in-time index membership.
    """

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    tickers = sorted(frame["ticker"].dropna().astype(str).str.upper().unique().tolist()) if "ticker" in frame else []
    backtest = config.get("backtest", {})
    strategy = config.get("strategy", {})
    root = Path(project_root).resolve()
    repository_state = git_state(root)

    artifacts: dict[str, dict[str, Any]] = {}
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            relative = str(path.resolve().relative_to(root))
        except ValueError:
            relative = str(path)
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    canonical_config = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return {
        "schema_version": 1,
        "generated_at_utc": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "git_commit": git_commit or repository_state["commit"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_worktree_dirty": repository_state["dirty"],
        },
        "data": {
            "source": "Yahoo Finance via yfinance",
            "price_field": "adjusted_close",
            "corporate_actions": "auto_adjust=False with adjusted_close used for return calculations",
            "start_date": _iso_date(frame["date"].min()) if "date" in frame else None,
            "end_date": _iso_date(frame["date"].max()) if "date" in frame else None,
            "rows": int(len(frame)),
            "tickers": tickers,
            "ticker_count": len(tickers),
            "configuration_sha256": hashlib.sha256(canonical_config).hexdigest(),
        },
        "backtest": {
            "strategy": strategy.get("name", "unknown"),
            "top_n": strategy.get("top_n"),
            "momentum_horizons": strategy.get("momentum_horizons"),
            "initial_capital": backtest.get("initial_capital"),
            "transaction_cost_bps": backtest.get("transaction_cost_bps"),
            "slippage_bps": backtest.get("slippage_bps"),
            "signal_lag_sessions": backtest.get("signal_lag"),
            "rebalance_frequency": strategy.get("rebalance_frequency"),
            "summary": backtest_summary or {},
        },
        "validation": {
            "chronological_signal_lag": int(backtest.get("signal_lag", 1)) >= 1,
            "future_price_backfill": False,
            "fundamentals_point_in_time": False,
            "universe_point_in_time": False,
            "warnings": [
                "yfinance fundamentals are current snapshots, not point-in-time historical fundamentals.",
                "The visible large-cap universe is selected ex post and can contain survivorship bias.",
            ],
        },
        "artifacts": artifacts,
    }


def write_backtest_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write a stable, human-readable manifest beside the result files."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path
