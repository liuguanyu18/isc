from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict
    predictions: pd.DataFrame


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
    }


def build_lag_features(df: pd.DataFrame, target_col: str, lag: int, exogenous_cols: list[str] | None = None):
    exogenous_cols = exogenous_cols or []

    lagged = pd.concat(
        {f"lag_{i}": df[target_col].shift(i) for i in range(1, lag + 1)},
        axis=1,
    )
    X = pd.concat([lagged, df[exogenous_cols]], axis=1) if exogenous_cols else lagged
    X = X.dropna()
    y = df.loc[X.index, target_col]
    return X, y


def train_naive(df: pd.DataFrame, target_col: str, test_ratio: float = 0.2) -> TrainingResult:
    series = df[target_col].dropna()
    split_idx = int(len(series) * (1 - test_ratio))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    last_value = train_series.iloc[-1] if not train_series.empty else test_series.iloc[0]
    y_pred = pd.Series([last_value] * len(test_series), index=test_series.index)

    metrics = _metrics(test_series, y_pred)
    predictions = pd.DataFrame({"y_true": test_series, "y_pred": y_pred})

    return TrainingResult(model_name="Naive", metrics=metrics, predictions=predictions)


def train_linear_regression(
    df: pd.DataFrame,
    target_col: str,
    exogenous_cols: list[str] | None = None,
    lag: int = 24,
    test_ratio: float = 0.2,
) -> TrainingResult:
    X, y = build_lag_features(df, target_col, lag, exogenous_cols)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    metrics = _metrics(y_test, y_pred)
    predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

    return TrainingResult(model_name="LinearRegression", metrics=metrics, predictions=predictions)


def train_arima(df: pd.DataFrame, target_col: str, test_ratio: float = 0.2, seasonal: bool = False) -> TrainingResult:
    if ARIMA is None:
        raise RuntimeError("statsmodels 未安装，无法训练 ARIMA")

    series = df[target_col].dropna()
    split_idx = int(len(series) * (1 - test_ratio))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 24) if seasonal else None

    if seasonal_order:
        model = ARIMA(train_series, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(train_series, order=order)

    fit = model.fit()
    forecast = fit.forecast(steps=len(test_series))
    forecast.index = test_series.index

    metrics = _metrics(test_series, forecast)
    predictions = pd.DataFrame({"y_true": test_series, "y_pred": forecast})

    name = "SARIMA" if seasonal_order else "ARIMA"
    return TrainingResult(model_name=name, metrics=metrics, predictions=predictions)


def train_prophet(df: pd.DataFrame, time_col: str, target_col: str, test_ratio: float = 0.2) -> TrainingResult:
    if Prophet is None:
        raise RuntimeError("prophet 未安装，无法训练 Prophet")

    series = df[[time_col, target_col]].dropna().rename(columns={time_col: "ds", target_col: "y"})
    split_idx = int(len(series) * (1 - test_ratio))
    train_df = series.iloc[:split_idx]
    test_df = series.iloc[split_idx:]

    model = Prophet()
    model.fit(train_df)
    future = test_df[["ds"]]
    forecast = model.predict(future)
    y_pred = forecast["yhat"].values

    metrics = _metrics(test_df["y"].values, y_pred)
    predictions = pd.DataFrame({"y_true": test_df["y"].values, "y_pred": y_pred}, index=test_df["ds"].values)

    return TrainingResult(model_name="Prophet", metrics=metrics, predictions=predictions)


def train_model(
    model_name: str,
    df: pd.DataFrame,
    time_col: Optional[str],
    target_col: str,
    exogenous_cols: list[str] | None = None,
    lag: int = 24,
    test_ratio: float = 0.2,
    seasonal: bool = False,
) -> TrainingResult:
    if model_name == "Naive":
        return train_naive(df, target_col, test_ratio)
    if model_name == "LinearRegression":
        return train_linear_regression(df, target_col, exogenous_cols, lag, test_ratio)
    if model_name in {"ARIMA", "SARIMA"}:
        return train_arima(df, target_col, test_ratio, seasonal=model_name == "SARIMA")
    if model_name == "Prophet":
        if time_col is None:
            raise RuntimeError("Prophet 需要时间列")
        return train_prophet(df, time_col, target_col, test_ratio)

    raise ValueError(f"未知模型: {model_name}")
