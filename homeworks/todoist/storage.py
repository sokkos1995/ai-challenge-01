from __future__ import annotations

import json
from pathlib import Path

from .models import Task


class JsonTaskStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def load_tasks(self) -> list[Task]:
        if not self.db_path.exists():
            return []
        raw = json.loads(self.db_path.read_text(encoding="utf-8"))
        return [Task.from_dict(item) for item in raw]

    def save_tasks(self, tasks: list[Task]) -> None:
        payload = [task.to_dict() for task in tasks]
        self.db_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
