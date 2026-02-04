from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .utils import FrequencyInfo, detect_seasonality, infer_frequency, numeric_columns, top_n_by_abs

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except Exception:
    seasonal_decompose = None


@dataclass
class FeatureAnalysisResult:
    correlation: pd.Series
    key_variables: pd.Series
    frequency_info: FrequencyInfo
    decomposition: Optional[object]
    has_seasonality: bool


def analyze_features(df: pd.DataFrame, time_col: Optional[str], target_col: Optional[str]) -> FeatureAnalysisResult:
    numeric_cols = numeric_columns(df)
    correlation = pd.Series(dtype="float64")

    if target_col and target_col in df.columns and target_col in numeric_cols:
        correlation = df[numeric_cols].corr()[target_col].drop(labels=[target_col])

    key_variables = top_n_by_abs(correlation, n=5)

    frequency_info = FrequencyInfo(freq=None, period=None, description="未检测到频率")
    decomposition = None
    has_seasonality = False

    if time_col and time_col in df.columns:
        frequency_info = infer_frequency(df[time_col])

    if target_col and target_col in df.columns and frequency_info.period and seasonal_decompose:
        series = df[target_col].dropna()
        if len(series) >= frequency_info.period * 2:
            decomposition = seasonal_decompose(series, period=frequency_info.period, model="additive")

    if target_col and target_col in df.columns:
        has_seasonality = detect_seasonality(df[target_col].dropna(), frequency_info.period)

    return FeatureAnalysisResult(
        correlation=correlation,
        key_variables=key_variables,
        frequency_info=frequency_info,
        decomposition=decomposition,
        has_seasonality=has_seasonality,
    )
