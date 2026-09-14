"""Fail-closed validation for published research artifacts."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_RESULT_FILES = (
    "backtest_manifest.json",
    "performance_summary.csv",
    "portfolio_value.csv",
    "trade_history.csv",
    "data_quality_report.csv",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> list[str]:
    missing = sorted(columns.difference(frame.columns))
    return [f"{label}: missing columns {missing}"] if missing else []


def validate_research_outputs(root: str | Path) -> dict[str, Any]:
    """Validate the minimum evidence required for a public research run."""

    root_path = Path(root)
    results_path = root_path / "results"
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [name for name in REQUIRED_RESULT_FILES if not (results_path / name).exists()]
    errors.extend(f"results/{name}: required artifact is missing" for name in missing_files)

    manifest: dict[str, Any] = {}
    manifest_path = results_path / "backtest_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"results/backtest_manifest.json: invalid JSON ({exc})")

    frames: dict[str, pd.DataFrame] = {}
    for name in ("portfolio_value.csv", "trade_history.csv", "data_quality_report.csv", "performance_summary.csv"):
        path = results_path / name
        if not path.exists():
            continue
        try:
            frames[name] = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - parser/provider edge
            errors.append(f"results/{name}: cannot read CSV ({exc})")

    portfolio = frames.get("portfolio_value.csv", pd.DataFrame())
    trades = frames.get("trade_history.csv", pd.DataFrame())
    quality = frames.get("data_quality_report.csv", pd.DataFrame())
    performance = frames.get("performance_summary.csv", pd.DataFrame())

    errors.extend(_require_columns(portfolio, {"date", "total_value", "daily_return"}, "portfolio_value.csv"))
    if not portfolio.empty:
        dates = pd.to_datetime(portfolio["date"], errors="coerce")
        if dates.isna().any():
            errors.append("portfolio_value.csv: contains invalid dates")
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            errors.append("portfolio_value.csv: dates must be increasing and unique")
        values = pd.to_numeric(portfolio["total_value"], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            errors.append("portfolio_value.csv: total_value must be finite and positive")

    errors.extend(_require_columns(trades, {"date", "ticker", "side", "quantity", "executed_price", "notional"}, "trade_history.csv"))
    if not trades.empty:
        quantities = pd.to_numeric(trades["quantity"], errors="coerce")
        prices = pd.to_numeric(trades["executed_price"], errors="coerce")
        if quantities.isna().any() or (quantities <= 0).any():
            errors.append("trade_history.csv: quantity must be positive")
        if prices.isna().any() or (prices <= 0).any():
            errors.append("trade_history.csv: executed_price must be positive")

    if not quality.empty:
        for column in ("duplicate_records", "missing_price_values", "invalid_dates", "non_positive_prices"):
            if column in quality and pd.to_numeric(quality[column], errors="coerce").fillna(0).gt(0).any():
                errors.append(f"data_quality_report.csv: {column} contains non-zero values")

    if manifest:
        validation = manifest.get("validation", {})
        if validation.get("chronological_signal_lag") is not True:
            errors.append("backtest_manifest.json: chronological signal lag is not verified")
        if validation.get("future_price_backfill") is not False:
            errors.append("backtest_manifest.json: future price backfill is not explicitly disabled")
        cutoff = manifest.get("data", {}).get("end_date")
        if cutoff:
            cutoff_date = pd.to_datetime(cutoff, errors="coerce")
            if pd.notna(cutoff_date) and cutoff_date.date() > date.today():
                errors.append(f"backtest_manifest.json: data cutoff {cutoff} is in the future")
        warnings.extend(str(item) for item in validation.get("warnings", []))

    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "validated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "root": str(root_path),
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "portfolio_rows": int(len(portfolio)),
            "trade_rows": int(len(trades)),
            "quality_rows": int(len(quality)),
            "performance_rows": int(len(performance)),
            "data_cutoff": manifest.get("data", {}).get("end_date"),
        },
    }


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a validation report beside the result artifacts."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path
