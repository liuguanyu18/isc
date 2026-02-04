from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import clamp_series, compute_iqr_bounds, numeric_columns


@dataclass
class CleaningResult:
    data: pd.DataFrame
    report: dict
    scaler: Optional[StandardScaler]


def clean_data(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    fill_strategy: str = "interpolate",
    outlier_strategy: str = "clip",
    scale_numeric: bool = False,
) -> CleaningResult:
    df = df.copy()
    report = {
        "missing_values": {},
        "outliers": {},
        "scaling": "none",
    }

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)

    numeric_cols = numeric_columns(df)

    for col in numeric_cols:
        missing_count = int(df[col].isna().sum())
        report["missing_values"][col] = missing_count

        if missing_count > 0:
            if fill_strategy == "interpolate" and time_col:
                df[col] = df[col].interpolate(method="linear")
                df[col] = df[col].fillna(method="bfill").fillna(method="ffill")
            elif fill_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(0)

        if outlier_strategy != "none":
            lower, upper = compute_iqr_bounds(df[col])
            outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
            report["outliers"][col] = {
                "count": outlier_count,
                "lower": float(lower),
                "upper": float(upper),
            }
            if outlier_strategy == "clip" and outlier_count > 0:
                df[col] = clamp_series(df[col], lower, upper)

    scaler = None
    if scale_numeric and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        report["scaling"] = "standard"

    return CleaningResult(data=df, report=report, scaler=scaler)
