"""Refresh and validate the public Project Universe snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_utils import load_config
from data.data_loader import expected_latest_market_date
from investment_platform.market_data_refresh import refresh_project_market_data


PUBLIC_FEATURE_FILE = Path("data/processed/feature_dataset.csv")
PUBLIC_QUALITY_FILE = Path("data/processed/data_quality_report.csv")
PUBLIC_README_FILE = Path("README.md")
PUBLIC_FILES = (
    Path("demo_data/README.md"),
    Path("demo_data/data/processed/data_quality_report.csv"),
    Path("demo_data/data/processed/feature_dataset.csv"),
)
SENSITIVE_TEXT = re.compile(
    r"(?i)(?:/Users/|/var/folders|TemporaryItems|account(?:\s*(?:number|no\.?|#))?|"
    r"holdings?|brokerage|balances?|transactions?|IBKR|Moomoo|api[_ -]?key|"
    r"access[_ -]?token|refresh[_ -]?token|password|secret|bearer)"
)


class RefreshValidationError(ValueError):
    """Raised when a candidate public refresh is not safe to publish."""


@dataclass(frozen=True)
class RefreshMetrics:
    """Validated metrics for one candidate public snapshot."""

    earliest_date: date
    latest_date: date
    ticker_count: int
    row_count: int
    missing_tickers: tuple[str, ...]
    duplicate_rows: int
    data_changed: bool = False


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype("string").str.upper().str.strip()
    return out


def _read_previous_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker"])
    return _normalise(pd.read_csv(path))


def _canonicalise_public_frame(
    frame: pd.DataFrame,
    sort_columns: list[str],
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize public CSV semantics before comparing or serializing them."""

    out = _normalise(frame)
    order = list(column_order or out.columns)
    if set(order) != set(out.columns):
        order = list(out.columns)

    for column in order:
        if column == "date" or column.endswith("_date"):
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        elif pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("boolean")

    return out.loc[:, order].sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def public_csv_changed(candidate_path: Path, existing_path: Path, sort_columns: list[str]) -> bool:
    """Return whether two public CSVs differ semantically, ignoring formatting noise."""

    if not existing_path.exists():
        return True
    candidate = pd.read_csv(candidate_path)
    existing = pd.read_csv(existing_path)
    if set(candidate.columns) != set(existing.columns):
        return True
    order = list(existing.columns)
    return not _canonicalise_public_frame(candidate, sort_columns, order).equals(
        _canonicalise_public_frame(existing, sort_columns, order)
    )


def _write_canonical_csv(
    frame: pd.DataFrame,
    destination: Path,
    sort_columns: list[str],
    column_order: list[str],
) -> None:
    """Write a stable UTF-8, LF-terminated, index-free public CSV atomically."""

    canonical = _canonicalise_public_frame(frame, sort_columns, column_order)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            canonical.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                float_format="%.17g",
                date_format="%Y-%m-%d",
                na_rep="",
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def publish_csv_if_changed(
    candidate_path: Path,
    existing_path: Path,
    sort_columns: list[str],
) -> bool:
    """Publish a candidate only when its canonical semantic content changed."""

    if not public_csv_changed(candidate_path, existing_path, sort_columns):
        return False
    candidate = pd.read_csv(candidate_path)
    if existing_path.exists():
        existing_columns = list(pd.read_csv(existing_path, nrows=0).columns)
        column_order = existing_columns if set(existing_columns) == set(candidate.columns) else list(candidate.columns)
    else:
        column_order = list(candidate.columns)
    _write_canonical_csv(candidate, existing_path, sort_columns, column_order)
    return True


def _assert_public_text(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if SENSITIVE_TEXT.search(text):
            raise RefreshValidationError(f"sensitive text detected in public artifact: {path}")


def _updated_public_readme(source: Path, destination: Path, latest: date) -> None:
    """Stage an accurate public coverage note without touching the old copy."""

    text = source.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"(The equity feature snapshot covers \d{4}-\d{2}-\d{2} through )\d{4}-\d{2}-\d{2}(;)",
        rf"\g<1>{latest.isoformat()}\g<2>",
        text,
        count=1,
    )
    if replacements != 1:
        raise RefreshValidationError("public README coverage statement was not found")
    destination.write_text(updated, encoding="utf-8")


def validate_refresh_outputs(
    refreshed: pd.DataFrame,
    clean: pd.DataFrame,
    quality: pd.DataFrame,
    expected_tickers: set[str],
    expected_latest: pd.Timestamp,
    previous: pd.DataFrame | None = None,
) -> RefreshMetrics:
    """Validate all pipeline outputs before any public file is replaced."""

    feature = _normalise(refreshed)
    clean_frame = _normalise(clean)
    quality_frame = _normalise(quality)
    required_feature = {"date", "ticker"}
    required_clean = {"date", "ticker", "close", "adjusted_close"}
    if feature.empty:
        raise RefreshValidationError("candidate feature dataset is empty")
    if not required_feature.issubset(feature.columns):
        raise RefreshValidationError("candidate feature dataset is missing date/ticker columns")
    if not required_clean.issubset(clean_frame.columns) or clean_frame.empty:
        raise RefreshValidationError("candidate clean dataset is missing required price columns")

    missing = tuple(sorted(expected_tickers - set(feature["ticker"].dropna().unique())))
    extra = sorted(set(feature["ticker"].dropna().unique()) - expected_tickers)
    if missing or extra:
        raise RefreshValidationError(f"ticker coverage invalid; missing={missing}, extra={extra}")

    duplicate_rows = int(feature.duplicated(["date", "ticker"]).sum())
    clean_duplicates = int(clean_frame.duplicated(["date", "ticker"]).sum())
    if duplicate_rows or clean_duplicates:
        raise RefreshValidationError(
            f"duplicate date/ticker rows detected; feature={duplicate_rows}, clean={clean_duplicates}"
        )

    if feature[["date", "ticker"]].isna().any(axis=1).any():
        raise RefreshValidationError("candidate feature dataset contains invalid date/ticker values")
    if clean_frame[["date", "ticker", "close", "adjusted_close"]].isna().any(axis=1).any():
        raise RefreshValidationError("candidate clean dataset contains missing core price values")
    if (clean_frame[["close", "adjusted_close"]] <= 0).any(axis=1).any():
        raise RefreshValidationError("candidate clean dataset contains non-positive prices")

    latest = feature["date"].max().normalize()
    earliest = feature["date"].min().normalize()
    if latest < expected_latest.normalize():
        raise RefreshValidationError(
            f"candidate latest date {latest.date()} is older than expected {expected_latest.date()}"
        )

    per_ticker_latest = feature.groupby("ticker")["date"].max().dt.normalize()
    stale_tickers = sorted(ticker for ticker in expected_tickers if per_ticker_latest[ticker] < expected_latest.normalize())
    if stale_tickers:
        raise RefreshValidationError(f"tickers are stale: {stale_tickers}")

    if quality_frame.empty or not {"ticker", "rows", "end_date", "duplicate_records", "invalid_dates"}.issubset(
        quality_frame.columns
    ):
        raise RefreshValidationError("candidate data quality report is incomplete")
    if set(quality_frame["ticker"]) != expected_tickers:
        raise RefreshValidationError("data quality report ticker coverage does not match the universe")
    if int(quality_frame["duplicate_records"].fillna(0).sum()) or int(quality_frame["invalid_dates"].fillna(0).sum()):
        raise RefreshValidationError("data quality report contains duplicate or invalid records")

    feature_counts = feature.groupby("ticker").size()
    previous_counts = previous.groupby("ticker").size() if previous is not None and not previous.empty else pd.Series(dtype="int64")
    suspicious = [
        ticker
        for ticker in expected_tickers
        if ticker in previous_counts and feature_counts.get(ticker, 0) < previous_counts[ticker] * 0.90
    ]
    if suspicious:
        raise RefreshValidationError(f"candidate row counts are suspiciously low for: {sorted(suspicious)}")

    report_ends = quality_frame.set_index("ticker")["end_date"].astype(str).str[:10]
    actual_ends = per_ticker_latest.astype(str).str[:10]
    if not report_ends.equals(actual_ends.sort_index().reindex(report_ends.index).sort_index()):
        # Compare by ticker explicitly so report row order cannot hide a mismatch.
        for ticker in expected_tickers:
            if report_ends.get(ticker) != actual_ends.get(ticker):
                raise RefreshValidationError(f"quality report end date mismatch for {ticker}")

    return RefreshMetrics(
        earliest_date=earliest.date(),
        latest_date=latest.date(),
        ticker_count=int(feature["ticker"].nunique()),
        row_count=int(len(feature)),
        missing_tickers=missing,
        duplicate_rows=duplicate_rows,
    )


def update_public_snapshot(project_root: Path, start: str | None = None) -> RefreshMetrics:
    """Build in a temporary directory and atomically publish only public outputs."""

    data_root = project_root / "demo_data"
    feature_path = data_root / PUBLIC_FEATURE_FILE
    quality_path = data_root / PUBLIC_QUALITY_FILE
    previous = _read_previous_snapshot(feature_path)
    config = load_config(project_root / "config.yaml")
    tickers = {str(ticker).upper().strip() for ticker in config.get("universe", [])}
    if not tickers:
        raise RefreshValidationError("public ticker universe is empty")
    public_start = start or (
        previous["date"].min().normalize().strftime("%Y-%m-%d") if not previous.empty else str(config["data"]["start"])
    )
    expected_latest = expected_latest_market_date().normalize()

    with tempfile.TemporaryDirectory(prefix="qtf_public_project_refresh_") as tmp:
        temporary_root = Path(tmp)
        refresh_project_market_data(temporary_root, sorted(tickers), public_start)
        candidate_feature = temporary_root / "data" / "processed" / "feature_dataset.csv"
        candidate_clean = temporary_root / "data" / "processed" / "clean_stock_data.csv"
        candidate_quality = temporary_root / "data" / "processed" / "data_quality_report.csv"
        candidate_readme = temporary_root / "README.md"
        candidate_feature_frame = pd.read_csv(candidate_feature)
        candidate_clean_frame = pd.read_csv(candidate_clean)
        candidate_quality_frame = pd.read_csv(candidate_quality)
        metrics = validate_refresh_outputs(
            candidate_feature_frame,
            candidate_clean_frame,
            candidate_quality_frame,
            tickers,
            expected_latest,
            previous,
        )
        _updated_public_readme(data_root / PUBLIC_README_FILE, candidate_readme, metrics.latest_date)
        _assert_public_text((candidate_feature, candidate_quality, candidate_readme))
        readme_path = data_root / PUBLIC_README_FILE
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        feature_changed = publish_csv_if_changed(candidate_feature, feature_path, ["date", "ticker"])
        quality_changed = publish_csv_if_changed(candidate_quality, quality_path, ["ticker"])
        readme_changed = not readme_path.exists() or readme_path.read_bytes() != candidate_readme.read_bytes()
        if readme_changed:
            temporary_readme = readme_path.with_name(f".{readme_path.name}.tmp")
            shutil.copy2(candidate_readme, temporary_readme)
            os.replace(temporary_readme, readme_path)
        metrics = RefreshMetrics(
            earliest_date=metrics.earliest_date,
            latest_date=metrics.latest_date,
            ticker_count=metrics.ticker_count,
            row_count=metrics.row_count,
            missing_tickers=metrics.missing_tickers,
            duplicate_rows=metrics.duplicate_rows,
            data_changed=feature_changed or quality_changed or readme_changed,
        )

    print(f"Expected latest market date: {expected_latest.date()}")
    print(f"Dataset date before: {previous['date'].max().date() if not previous.empty else 'missing'}")
    print(f"Dataset date after: {metrics.latest_date}")
    print(f"Ticker count: {metrics.ticker_count}/{len(tickers)}")
    print(f"Row count: {metrics.row_count}")
    print(f"Data changed: {'yes' if metrics.data_changed else 'no'}")
    print("Public files refreshed: demo_data/README.md, demo_data/data/processed/feature_dataset.csv, demo_data/data/processed/data_quality_report.csv")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--start", default=None, help="Optional public snapshot start date")
    args = parser.parse_args()
    try:
        update_public_snapshot(args.project_root.resolve(), args.start)
    except (OSError, KeyError, ValueError, RefreshValidationError) as exc:
        print(f"Public project refresh rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
