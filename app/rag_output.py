from __future__ import annotations

from app.services.rag_service import RagAnswer


def format_rag_answer(answer: RagAnswer, prefix: str = "agent> ") -> str:
    lines: list[str] = [f"{prefix}{answer.answer}", f"{prefix}Sources:"]
    for item in answer.sources:
        source = item.get("source", "unknown")
        section = item.get("section", "unknown")
        chunk_id = item.get("chunk_id", "")
        lines.append(f"{prefix}  - {source} :: {section} ({chunk_id})")
    lines.append(f"{prefix}Quotes:")
    for quote in answer.quotes:
        lines.append(f'{prefix}  - "{quote}"')
    return "\n".join(lines)
