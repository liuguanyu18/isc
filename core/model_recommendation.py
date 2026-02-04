from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class ModelRecommendation:
    name: str
    rationale: str
    requires: str


def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def recommend_models(data_length: int, has_seasonality: bool) -> list[ModelRecommendation]:
    recommendations = []

    recommendations.append(
        ModelRecommendation(
            name="Naive",
            rationale="快速基线，适合短序列或做对比",
            requires="无",
        )
    )

    recommendations.append(
        ModelRecommendation(
            name="LinearRegression",
            rationale="适合线性趋势与滞后特征",
            requires="scikit-learn",
        )
    )

    if _is_available("statsmodels") and data_length >= 50:
        recommendations.append(
            ModelRecommendation(
                name="ARIMA",
                rationale="适合稳定时间序列建模",
                requires="statsmodels",
            )
        )

    if _is_available("prophet") and data_length >= 100:
        recommendations.append(
            ModelRecommendation(
                name="Prophet",
                rationale="适合季节性明显且有节假日效应的数据",
                requires="prophet",
            )
        )

    if has_seasonality and _is_available("statsmodels"):
        recommendations.insert(
            2,
            ModelRecommendation(
                name="SARIMA",
                rationale="季节性明显时更稳健",
                requires="statsmodels",
            ),
        )

    return recommendations
