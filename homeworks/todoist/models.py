from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Task:
    id: str
    title: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str
    due_date: str | None = None
    source: str = "manual"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        title: str,
        description: str = "",
        *,
        priority: str = "medium",
        due_date: str | None = None,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> "Task":
        now = utc_now_iso()
        return cls(
            id=f"tsk_{uuid4().hex[:10]}",
            title=title.strip(),
            description=description.strip(),
            status="todo",
            priority=priority,
            created_at=now,
            updated_at=now,
            due_date=due_date,
            source=source,
            tags=tags or [],
        )

    def with_status(self, status: str) -> "Task":
        data = asdict(self)
        data["status"] = status
        data["updated_at"] = utc_now_iso()
        return Task(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(**data)
