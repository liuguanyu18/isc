from __future__ import annotations

STEP_ORDER = [
    ("数据接入", "data_ingestion"),
    ("字段语义", "field_semantics"),
    ("数据清洗", "data_cleaning"),
    ("特征分析", "feature_analysis"),
    ("模型推荐", "model_recommendation"),
    ("训练验证", "training_validation"),
    ("结果可视化", "result_visualization"),
    ("交互反馈", "interaction_feedback"),
]


def build_flow_dot(status: dict[str, str] | None = None) -> str:
    status = status or {}
    lines = ["digraph G {", "rankdir=LR;", "node [shape=box, style=filled, fontname=Helvetica];"]

    for label, key in STEP_ORDER:
        state = status.get(key, "pending")
        color = _status_color(state)
        lines.append(f'"{label}" [fillcolor="{color}"];')

    for i in range(len(STEP_ORDER) - 1):
        lines.append(f'"{STEP_ORDER[i][0]}" -> "{STEP_ORDER[i + 1][0]}";')

    lines.append("}")
    return "\n".join(lines)


def _status_color(state: str) -> str:
    return {
        "done": "#B9F6CA",
        "running": "#FFF9C4",
        "error": "#FFCDD2",
        "pending": "#ECEFF1",
    }.get(state, "#ECEFF1")
