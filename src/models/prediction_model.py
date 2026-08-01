"""Machine learning models for forward return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "daily_return",
    "volatility_21d",
    "sma_20",
    "sma_60",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "zscore_20",
    "return_21d",
    "return_63d",
    "return_126d",
    "pe_ratio",
    "pb_ratio",
    "market_cap",
    "roe",
    "revenue_growth",
]


@dataclass(frozen=True)
class ModelResult:
    """Fitted model and validation diagnostics."""

    name: str
    model: Any
    cv_scores: np.ndarray
    test_r2: float
    test_mae: float
    test_rmse: float


def create_supervised_dataset(
    feature_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "future_21d_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create aligned X/y data for forward-return prediction."""

    features = DEFAULT_FEATURE_COLUMNS if feature_columns is None else feature_columns
    frame = feature_data.copy()
    for column in features:
        if column not in frame.columns:
            frame[column] = np.nan
    if target_column not in frame.columns:
        raise ValueError(f"missing target column: {target_column}")

    model_frame = frame[features + [target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if model_frame.empty:
        raise ValueError("no complete rows available for supervised learning")
    return model_frame[features], model_frame[target_column]


def chronological_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split observations chronologically to reduce time-series leakage."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split = int(len(x) * (1 - test_size))
    if split <= 0 or split >= len(x):
        raise ValueError("not enough observations for train/test split")
    return x.iloc[:split], x.iloc[split:], y.iloc[:split], y.iloc[split:]


def _model_specs(random_state: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "linear_regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=random_state,
        ),
    }


def train_return_models(
    feature_data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "future_21d_return",
    test_size: float = 0.25,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, ModelResult]:
    """Train linear, random forest, and gradient boosting return models."""

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    x, y = create_supervised_dataset(feature_data, feature_columns, target_column)
    x_train, x_test, y_train, y_test = chronological_train_test_split(x, y, test_size)
    n_splits = min(cv_splits, max(2, len(x_train) // 20))
    cv = TimeSeriesSplit(n_splits=n_splits)

    results: dict[str, ModelResult] = {}
    for name, model in _model_specs(random_state).items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        results[name] = ModelResult(
            name=name,
            model=model,
            cv_scores=cv_scores,
            test_r2=float(r2_score(y_test, prediction)),
            test_mae=float(mean_absolute_error(y_test, prediction)),
            test_rmse=float(np.sqrt(mean_squared_error(y_test, prediction))),
        )
    return results
