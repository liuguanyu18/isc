from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

TIME_KEYWORDS = ["time", "timestamp", "date", "datetime", "ts", "record_time"]


@dataclass
class FrequencyInfo:
    freq: Optional[str]
    period: Optional[int]
    description: str


def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None

    candidates = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(key in col_lower for key in TIME_KEYWORDS):
            candidates.append(col)

    if not candidates:
        return None

    best_col = None
    best_score = -1
    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        score = parsed.notna().mean()
        if score > best_score:
            best_col = col
            best_score = score

    return best_col


def normalize_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
    return df


def infer_frequency(series: pd.Series) -> FrequencyInfo:
    if series is None or series.empty:
        return FrequencyInfo(freq=None, period=None, description="未检测到频率")

    series = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if series.empty or len(series) < 3:
        return FrequencyInfo(freq=None, period=None, description="样本太少，无法推断频率")

    freq = pd.infer_freq(series)
    if freq is None:
        diffs = series.diff().dropna().dt.total_seconds()
        if diffs.empty:
            return FrequencyInfo(freq=None, period=None, description="无法推断频率")
        most_common = diffs.mode().iloc[0]
        freq = f"{int(most_common)}S"

    period = map_frequency_to_period(freq)
    description = f"推断频率: {freq}, 建议周期: {period}" if period else f"推断频率: {freq}"
    return FrequencyInfo(freq=freq, period=period, description=description)


def map_frequency_to_period(freq: str) -> Optional[int]:
    if freq is None:
        return None

    freq_upper = freq.upper()
    if freq_upper.endswith("T") or freq_upper.endswith("MIN"):
        return 24 * 60
    if freq_upper.endswith("H"):
        return 24
    if freq_upper.endswith("D"):
        return 7
    if freq_upper.endswith("W"):
        return 52
    if freq_upper.endswith("M"):
        return 12
    return None


def numeric_columns(df: pd.DataFrame) -> Iterable[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def detect_seasonality(series: pd.Series, period: Optional[int]) -> bool:
    if series is None or series.empty or period is None:
        return False

    return len(series) >= period * 3


def top_n_by_abs(series: pd.Series, n: int = 5) -> pd.Series:
    if series is None or series.empty:
        return series

    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def clamp_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def compute_iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
