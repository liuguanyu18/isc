from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class VisualizationBundle:
    forecast_fig: go.Figure
    error_fig: go.Figure
    suggestions: list[str]


def plot_forecast(history: pd.Series, predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="历史负荷"))
    fig.add_trace(
        go.Scatter(x=predictions.index, y=predictions["y_pred"].values, mode="lines", name="预测值")
    )
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["y_true"].values,
            mode="lines",
            name="真实值",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_error_distribution(predictions: pd.DataFrame) -> go.Figure:
    errors = predictions["y_true"].values - predictions["y_pred"].values
    fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=40)])
    fig.update_layout(title="预测误差分布", height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def generate_suggestions(predictions: pd.DataFrame) -> list[str]:
    suggestions = []
    if predictions.empty:
        return suggestions

    peak_idx = predictions["y_pred"].idxmax()
    peak_value = predictions.loc[peak_idx, "y_pred"]
    suggestions.append(f"预测峰值出现在 {peak_idx}，负荷约 {peak_value:.2f}。")

    error = (predictions["y_true"] - predictions["y_pred"]).abs().mean()
    suggestions.append(f"平均绝对误差约 {error:.2f}，建议关注高负荷时段的调度策略。")

    return suggestions
