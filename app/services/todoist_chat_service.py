import re
from dataclasses import dataclass
from typing import Optional

from app.services.todoist_reminder_service import TodoistMcpClient


@dataclass
class TodoistCreateIntent:
    content: str
    due_string: str = ""


def _normalize_time(hours: str, minutes: str) -> str:
    return f"{int(hours):02d}:{int(minutes):02d}"


def _normalize_due_phrase(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""

    normalized = re.sub(r"\s+", " ", raw).strip()
    ru_match = re.fullmatch(
        r"(сегодня|завтра)\s+в\s+(\d{1,2})(?:[:\s](\d{1,2}))",
        normalized,
        flags=re.IGNORECASE,
    )
    if ru_match:
        day, hours, minutes = ru_match.groups()
        day_map = {"сегодня": "today", "завтра": "tomorrow"}
        return f"{day_map[day.lower()]} at {_normalize_time(hours, minutes or '00')}"

    ru_match_reversed = re.fullmatch(
        r"в\s+(\d{1,2})(?:[:\s](\d{1,2}))\s+(сегодня|завтра)",
        normalized,
        flags=re.IGNORECASE,
    )
    if ru_match_reversed:
        hours, minutes, day = ru_match_reversed.groups()
        day_map = {"сегодня": "today", "завтра": "tomorrow"}
        return f"{day_map[day.lower()]} at {_normalize_time(hours, minutes or '00')}"

    en_match = re.fullmatch(
        r"(today|tomorrow)\s+at\s+(\d{1,2})(?:[:\s](\d{1,2}))",
        normalized,
        flags=re.IGNORECASE,
    )
    if en_match:
        day, hours, minutes = en_match.groups()
        return f"{day.lower()} at {_normalize_time(hours, minutes or '00')}"

    en_match_reversed = re.fullmatch(
        r"at\s+(\d{1,2})(?:[:\s](\d{1,2}))\s+(today|tomorrow)",
        normalized,
        flags=re.IGNORECASE,
    )
    if en_match_reversed:
        hours, minutes, day = en_match_reversed.groups()
        return f"{day.lower()} at {_normalize_time(hours, minutes or '00')}"

    return normalized


def _split_content_and_due(text: str) -> tuple[str, str]:
    patterns = [
        r"^(?P<content>.*?)(?:\s+|,\s*)(?P<due>(?:сегодня|завтра)\s+в\s+\d{1,2}(?:[:\s]\d{1,2})?)\s*$",
        r"^(?P<content>.*?)(?:\s+|,\s*)(?P<due>в\s+\d{1,2}(?:[:\s]\d{1,2})\s+(?:сегодня|завтра))\s*$",
        r"^(?P<content>.*?)(?:\s+|,\s*)(?P<due>(?:today|tomorrow)\s+at\s+\d{1,2}(?:[:\s]\d{1,2})?)\s*$",
        r"^(?P<content>.*?)(?:\s+|,\s*)(?P<due>at\s+\d{1,2}(?:[:\s]\d{1,2})\s+(?:today|tomorrow))\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("content").strip(), _normalize_due_phrase(match.group("due"))
    return text.strip(), ""


def parse_todoist_create_intent(user_input: str) -> Optional[TodoistCreateIntent]:
    text = user_input.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered.startswith("create_task"):
        payload = text[len("create_task") :].strip()
    elif "todoist" in lowered or "тудуист" in lowered:
        if "задач" not in lowered and "create" not in lowered:
            return None
        payload = re.sub(r"(?i)\b(?:в\s+)?(?:todoist|тудуист[еа]?)\b", "", text).strip()
    else:
        return None

    quoted_parts = re.findall(r'"([^"]+)"', payload)
    due_string = ""
    if quoted_parts:
        content = quoted_parts[0].strip()
        payload_without_quotes = re.sub(r'"([^"]+)"', "", payload)
        payload_without_quotes = re.sub(
            r'(?i)\b(?:создай|сделай|добавь|create|task|задач[ауеиыо]?)\b',
            "",
            payload_without_quotes,
        ).strip(" :,-")
        _, due_string = _split_content_and_due(payload_without_quotes)
        if not due_string:
            content, due_string = _split_content_and_due(content)
    else:
        payload = re.sub(r'(?i)\b(?:создай|сделай|добавь|create)\b', "", payload).strip()
        payload = re.sub(r"(?i)\btask\b", "", payload).strip()
        payload = re.sub(r"(?i)\bзадач[ауеиыо]?\b", "", payload).strip(" :,-")
        if not payload:
            return None
        content, due_string = _split_content_and_due(payload)

    clean_content = re.sub(r"\s+", " ", content).strip(" ,.-")
    if not clean_content:
        return None
    return TodoistCreateIntent(content=clean_content, due_string=due_string)


class TodoistChatService:
    def __init__(self, client: Optional[TodoistMcpClient] = None) -> None:
        self._client = client or TodoistMcpClient()

    def maybe_handle(self, user_input: str) -> Optional[str]:
        intent = parse_todoist_create_intent(user_input)
        if intent is None:
            return None

        created = self._client.create_task(intent.content, intent.due_string, "")
        content = str(created.get("content", intent.content)).strip() or intent.content
        due = created.get("due")
        due_text = ""
        if isinstance(due, dict):
            due_text = str(due.get("string", "")).strip()
            if not due_text:
                due_text = str(due.get("date", "")).strip()
        if due_text:
            return f'agent> Todoist: задача "{content}" создана, срок: {due_text}.'
        return f'agent> Todoist: задача "{content}" создана.'
