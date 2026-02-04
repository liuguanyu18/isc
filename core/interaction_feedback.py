from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FeedbackEntry:
    user: str
    rating: int
    comments: str
    adjustments: dict[str, Any]


def save_feedback(entry: FeedbackEntry, path: str = "data/feedback.json") -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        existing = json.loads(file_path.read_text(encoding="utf-8"))
    else:
        existing = []

    existing.append(
        {
            "user": entry.user,
            "rating": entry.rating,
            "comments": entry.comments,
            "adjustments": entry.adjustments,
        }
    )

    file_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
