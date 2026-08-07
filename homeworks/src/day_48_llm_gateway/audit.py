"""JSONL audit log for the LLM gateway (no raw secrets)."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview_hash(text: str, *, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


@dataclass
class AuditRecord:
    audit_id: str
    ts: str
    ip: str
    mode: str
    blocked_stage: Optional[str]
    input_findings: list[str] = field(default_factory=list)
    output_findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_hash: str = ""
    answer_hash: str = ""
    prompt_preview: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def new_id(self) -> str:
        return f"aud_{uuid.uuid4().hex[:12]}"


def safe_preview(text: str, *, max_len: int = 80) -> str:
    """Short preview already expected to be redacted / non-secret."""
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"
