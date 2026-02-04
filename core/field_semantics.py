from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


SEMANTIC_DICT = {
    "time": ["time", "timestamp", "date", "datetime", "ts", "记录时间", "时间"],
    "power": ["power", "pwr", "load", "kw", "kwh", "demand", "负荷", "功率"],
    "voltage": ["voltage", "volt", "v", "电压"],
    "current": ["current", "amp", "a", "电流"],
    "temperature": ["temp", "temperature", "t", "温度"],
}


@dataclass
class SemanticResult:
    mapping: pd.DataFrame
    ambiguous: list[str]


def _score_match(column: str, keywords: list[str]) -> float:
    column_lower = column.lower()
    for kw in keywords:
        if kw.lower() == column_lower:
            return 0.95
    for kw in keywords:
        if kw.lower() in column_lower:
            return 0.75
    return 0.0


def infer_semantics(columns: list[str]) -> SemanticResult:
    rows = []
    ambiguous = []

    for col in columns:
        best_label = "unknown"
        best_score = 0.0
        for label, keywords in SEMANTIC_DICT.items():
            score = _score_match(str(col), keywords)
            if score > best_score:
                best_label = label
                best_score = score

        meaning = best_label if best_score > 0 else "unknown"
        if best_score < 0.6:
            ambiguous.append(col)

        rows.append(
            {
                "original": col,
                "canonical": meaning,
                "confidence": round(best_score, 2),
                "notes": "" if meaning != "unknown" else "需要确认字段含义",
            }
        )

    return SemanticResult(mapping=pd.DataFrame(rows), ambiguous=ambiguous)


def pick_best_column(mapping: pd.DataFrame, canonical: str) -> str | None:
    subset = mapping[mapping["canonical"] == canonical]
    if subset.empty:
        return None
    subset = subset.sort_values(["confidence"], ascending=False)
    return subset.iloc[0]["original"]
