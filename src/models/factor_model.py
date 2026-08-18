"""Fama-French factor loading estimation with robust standard errors."""

from __future__ import annotations

from pathlib import Path
import io
import urllib.request
import zipfile

import pandas as pd

KENNETH_FRENCH_DAILY_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def _read_factor_source(source: bytes | str | Path) -> str:
    """Return Kenneth French factor text from bytes, a path, or raw text."""

    if isinstance(source, bytes):
        data = source
    else:
        if "\n" in str(source):
            return str(source)
        path = Path(source)
        if path.exists():
            data = path.read_bytes()
        else:
            return str(source)

    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("Kenneth French archive did not contain a CSV file")
            return archive.read(csv_names[0]).decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


def parse_kenneth_french_daily_factors(source: bytes | str | Path) -> pd.DataFrame:
    """Parse official Kenneth French daily three-factor data.

    The source file stores returns in percentage points. This function converts
    `mkt_rf`, `smb`, `hml`, and `rf` to decimal daily returns.
    """

    text = _read_factor_source(source)
    rows: list[str] = []
    header: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower().replace(" ", "")
        if lower.startswith(",mkt-rf,smb,hml,rf") or lower.startswith("date,mkt-rf,smb,hml,rf"):
            header = stripped
            continue
        first = stripped.split(",", 1)[0].strip()
        if len(first) == 8 and first.isdigit():
            rows.append(stripped)
        elif rows:
            break

    if not rows:
        raise ValueError("could not locate daily factor rows in Kenneth French data")

    csv_text = (header or ",Mkt-RF,SMB,HML,RF") + "\n" + "\n".join(rows)
    frame = pd.read_csv(io.StringIO(csv_text))
    date_col = frame.columns[0]
    frame = frame.rename(
        columns={
            date_col: "date",
            "Mkt-RF": "mkt_rf",
            "Mkt-RF ": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RF": "rf",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d")
    for column in ("mkt_rf", "smb", "hml", "rf"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame[["date", "mkt_rf", "smb", "hml", "rf"]].dropna().sort_values("date").reset_index(drop=True)


def load_kenneth_french_factors(
    start: str | None = None,
    end: str | None = None,
    cache_path: str | Path | None = None,
    url: str = KENNETH_FRENCH_DAILY_FACTORS_URL,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load official Kenneth French daily factors from cache or Dartmouth.

    Parameters
    ----------
    start, end:
        Optional date filters.
    cache_path:
        Local CSV cache used to avoid repeated network calls.
    url:
        Kenneth French daily factor ZIP URL.
    refresh:
        If true, download even when the cache exists.
    """

    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists() and not refresh:
        factors = pd.read_csv(cache, parse_dates=["date"])
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            factors = parse_kenneth_french_daily_factors(response.read())
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            factors.to_csv(cache, index=False)

    if start is not None:
        factors = factors[factors["date"].ge(pd.to_datetime(start))]
    if end is not None:
        factors = factors[factors["date"].le(pd.to_datetime(end))]
    return factors.reset_index(drop=True)


def build_proxy_factors(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Build local factor proxies for offline tests and fallback runs.

    The production research path should use Kenneth French factors. These
    proxies are intentionally retained only for unit tests and environments
    without network access.
    """

    frame = feature_data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "daily_return" not in frame.columns:
        frame["daily_return"] = frame.groupby("ticker")["adjusted_close"].pct_change()

    rows = []
    for date, group in frame.dropna(subset=["daily_return"]).groupby("date"):
        market = group.loc[group["ticker"].eq("SPY"), "daily_return"]
        market_return = float(market.iloc[0]) if not market.empty else float(group["daily_return"].mean())

        size_factor = 0.0
        if "market_cap" in group.columns and group["market_cap"].notna().sum() >= 4:
            ranked_size = group.dropna(subset=["market_cap"]).sort_values("market_cap")
            small = ranked_size.head(max(1, len(ranked_size) // 3))["daily_return"].mean()
            big = ranked_size.tail(max(1, len(ranked_size) // 3))["daily_return"].mean()
            size_factor = float(small - big)

        value_factor = 0.0
        if "pb_ratio" in group.columns and group["pb_ratio"].notna().sum() >= 4:
            ranked_value = group.dropna(subset=["pb_ratio"]).sort_values("pb_ratio")
            value = ranked_value.head(max(1, len(ranked_value) // 3))["daily_return"].mean()
            growth = ranked_value.tail(max(1, len(ranked_value) // 3))["daily_return"].mean()
            value_factor = float(value - growth)

        rows.append(
            {
                "date": date,
                "mkt_rf": market_return,
                "smb": size_factor,
                "hml": value_factor,
                "rf": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fama_french_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    hac_maxlags: int = 5,
):
    """Fit Fama-French OLS with Newey-West HAC standard errors."""

    import statsmodels.api as sm

    y = pd.Series(strategy_returns).rename("strategy_return")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("strategy_returns must have a DatetimeIndex")

    factor_frame = factors.copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"])
    factor_frame = factor_frame.set_index("date")
    required = ["mkt_rf", "smb", "hml"]
    missing = [column for column in required if column not in factor_frame.columns]
    if missing:
        raise ValueError(f"missing factor columns: {missing}")
    if "rf" not in factor_frame.columns:
        factor_frame["rf"] = 0.0

    aligned = pd.concat([y, factor_frame[["mkt_rf", "smb", "hml", "rf"]]], axis=1).dropna()
    if aligned.shape[0] < 20:
        raise ValueError("not enough observations for factor regression")

    dependent = aligned["strategy_return"] - aligned["rf"]
    independent = sm.add_constant(aligned[["mkt_rf", "smb", "hml"]])
    result = sm.OLS(dependent, independent).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_maxlags)})

    exposure = pd.DataFrame(
        {
            "coefficient": result.params,
            "standard_error": result.bse,
            "p_value": result.pvalues,
            "t_stat": result.tvalues,
        }
    )
    exposure.loc["r_squared", "coefficient"] = result.rsquared
    exposure.loc["hac_maxlags", "coefficient"] = int(hac_maxlags)
    return result, exposure
