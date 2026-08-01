"""Mean-reversion strategy using Bollinger Bands and Z-score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.technical_indicators import bollinger_bands


def _single_name_positions(
    group: pd.DataFrame,
    window: int,
    entry_z: float,
    exit_z: float,
    price_col: str,
) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    bands = bollinger_bands(g[price_col], window=window, num_std=abs(entry_z))
    g["zscore"] = bands["zscore"].to_numpy()
    g["bb_middle"] = bands["middle"].to_numpy()
    g["bb_lower"] = bands["lower"].to_numpy()

    position = 0
    positions = []
    signals = []
    for zvalue in g["zscore"]:
        if np.isnan(zvalue):
            signals.append("HOLD")
        elif position == 0 and zvalue <= entry_z:
            position = 1
            signals.append("BUY")
        elif position == 1 and zvalue >= exit_z:
            position = 0
            signals.append("SELL")
        else:
            signals.append("HOLD")
        positions.append(position)

    g["target_position"] = positions
    g["signal"] = signals
    return g


def generate_mean_reversion_weights(
    data: pd.DataFrame,
    window: int = 20,
    entry_z: float = -2.0,
    exit_z: float = 0.0,
    gross_exposure: float = 1.0,
    price_col: str = "adjusted_close",
) -> pd.DataFrame:
    """Buy oversold stocks and exit when price reverts to the rolling mean."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")

    frame = data[["date", "ticker", price_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pieces = [
        _single_name_positions(group, window, entry_z, exit_z, price_col)
        for _, group in frame.groupby("ticker", sort=False)
    ]
    signals = pd.concat(pieces, ignore_index=True)
    active_count = signals.groupby("date")["target_position"].transform("sum")
    signals["target_weight"] = np.where(
        active_count > 0,
        signals["target_position"].div(active_count).mul(gross_exposure),
        0.0,
    )
    return signals[
        ["date", "ticker", "target_weight", "signal", "target_position", "zscore", "bb_middle", "bb_lower"]
    ].sort_values(["date", "ticker"]).reset_index(drop=True)
