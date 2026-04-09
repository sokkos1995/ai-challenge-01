import json
import os
import queue
import threading
from datetime import datetime
from typing import Any, Callable, Optional

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client

from app.config import positive_int_from_env
from app.storage import has_todoist_notification, mark_todoist_notification_sent


def _normalize_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def parse_todoist_due_datetime(task: dict[str, Any]) -> Optional[datetime]:
    due = task.get("due")
    if not isinstance(due, dict):
        return None
    due_datetime = str(due.get("datetime", "")).strip()
    if due_datetime:
        return _normalize_iso_datetime(due_datetime)
    due_date = str(due.get("date", "")).strip()
    if "T" in due_date:
        return _normalize_iso_datetime(due_date)
    return None


def todoist_due_key(task: dict[str, Any]) -> str:
    due = task.get("due")
    if not isinstance(due, dict):
        return ""
    due_datetime = str(due.get("datetime", "")).strip()
    if due_datetime:
        return due_datetime
    due_date = str(due.get("date", "")).strip()
    if due_date:
        return due_date
    return ""


def format_todoist_reminder_message(task: dict[str, Any], now: Optional[datetime] = None) -> str:
    due_at = parse_todoist_due_datetime(task)
    content = str(task.get("content", "")).strip() or "без названия"
    if due_at is None:
        return f"agent> Todoist: сделать {content}"

    if now is None:
        local_now = datetime.now(tz=due_at.tzinfo) if due_at.tzinfo is not None else datetime.now()
    elif due_at.tzinfo is not None and now.tzinfo is not None:
        local_now = now.astimezone(due_at.tzinfo)
    else:
        local_now = now
    if local_now.date() == due_at.date():
        when_text = f"сделать сегодня в {due_at.strftime('%H %M')}"
    else:
        when_text = f"сделать {due_at.strftime('%Y-%m-%d')} в {due_at.strftime('%H %M')}"
    return f"agent> Todoist: {when_text} {content}"


def due_todoist_tasks(tasks: list[dict[str, Any]], now: Optional[datetime] = None) -> list[dict[str, Any]]:
    current = now or datetime.now().astimezone()
    due_items: list[dict[str, Any]] = []
    for task in tasks:
        due_at = parse_todoist_due_datetime(task)
        if due_at is None:
            continue
        comparable_now = current
        if due_at.tzinfo is None and current.tzinfo is not None:
            comparable_now = current.replace(tzinfo=None)
        elif due_at.tzinfo is not None and current.tzinfo is None:
            comparable_now = current.replace(tzinfo=due_at.tzinfo)
        if due_at <= comparable_now:
            due_items.append(task)
    due_items.sort(key=lambda item: (parse_todoist_due_datetime(item) or current).isoformat())
    return due_items


class TodoistMcpClient:
    _stdio_lock = threading.Lock()

    def __init__(self, command: str = "python3", args: Optional[list[str]] = None) -> None:
        self._command = command
        self._args = args or ["-m", "app.mcp_servers.todoist_server"]

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = None
        server_params = StdioServerParameters(command=self._command, args=self._args)
        with open(os.devnull, "w", encoding="utf-8") as errlog:
            async with stdio_client(server_params, errlog=errlog) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        if result is None:
            raise RuntimeError(f"Todoist MCP returned no result for tool: {tool_name}")
        if result.isError:
            raise RuntimeError("Todoist MCP call failed")
        structured = result.structuredContent
        if isinstance(structured, dict):
            return structured

        text_chunks: list[str] = []
        for chunk in result.content:
            if getattr(chunk, "type", "") == "text":
                text_chunks.append(str(getattr(chunk, "text", "")))
        if not text_chunks:
            return {}

        parsed = json.loads("\n".join(text_chunks))
        return parsed if isinstance(parsed, dict) else {}

    async def _list_tasks_async(self, limit: int, filter_query: str) -> list[dict[str, Any]]:
        structured = await self._call_tool_async(
            "list_tasks",
            {"limit": limit, "filter_query": filter_query},
        )
        tasks = structured.get("tasks")
        if not isinstance(tasks, list):
            return []
        return [task for task in tasks if isinstance(task, dict)]

    def list_tasks(
        self,
        limit: int = 0,
        filter_query: str = "",
        *,
        wait_for_lock: bool = True,
    ) -> list[dict[str, Any]]:
        safe_limit = 0 if limit <= 0 else min(limit, 5000)
        acquired = self._stdio_lock.acquire(blocking=wait_for_lock)
        if not acquired:
            return []
        try:
            return anyio.run(self._list_tasks_async, safe_limit, filter_query.strip())
        finally:
            self._stdio_lock.release()

    async def _create_task_async(self, content: str, due_string: str, project_id: str) -> dict[str, Any]:
        return await self._call_tool_async(
            "create_task",
            {
                "content": content,
                "due_string": due_string,
                "project_id": project_id,
            },
        )

    def create_task(self, content: str, due_string: str = "", project_id: str = "") -> dict[str, Any]:
        with self._stdio_lock:
            return anyio.run(self._create_task_async, content, due_string, project_id)


class TodoistReminderService:
    _DEFAULT_FILTER_QUERY = "today | overdue"

    def __init__(
        self,
        db_path: str,
        poll_interval_sec: int = 30,
        task_fetcher: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ) -> None:
        self._db_path = db_path
        self._poll_interval_sec = max(5, poll_interval_sec)
        client = TodoistMcpClient()
        self._task_fetcher = task_fetcher or (
            lambda: client.list_tasks(
                limit=0,
                filter_query=self._DEFAULT_FILTER_QUERY,
                wait_for_lock=False,
            )
        )
        self._messages: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: str = ""

    @classmethod
    def from_env(cls) -> Optional["TodoistReminderService"]:
        enabled = os.getenv("TODOIST_REMINDERS_ENABLED", "1").strip().lower()
        if enabled in {"0", "false", "off", "no"}:
            return None
        if not os.getenv("TODOIST_API_TOKEN", "").strip():
            return None
        poll_interval = positive_int_from_env("TODOIST_REMINDER_POLL_SECONDS", 30)
        db_path = os.getenv("TODOIST_REMINDER_DB_PATH", ".llm_todoist_reminders.db").strip()
        return cls(db_path=db_path or ".llm_todoist_reminders.db", poll_interval_sec=poll_interval)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="todoist-reminders", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def drain_messages(self) -> list[str]:
        messages: list[str] = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                break
        return messages

    def _pending_due_tasks(self, now: Optional[datetime] = None) -> tuple[datetime, list[tuple[str, str, dict[str, Any]]]]:
        current = now or datetime.now().astimezone()
        pending: list[tuple[str, str, dict[str, Any]]] = []
        for task in due_todoist_tasks(self._task_fetcher(), now=current):
            task_id = str(task.get("id", "")).strip()
            due_key = todoist_due_key(task)
            if not task_id or not due_key:
                continue
            if has_todoist_notification(self._db_path, task_id, due_key):
                continue
            pending.append((task_id, due_key, task))
        return current, pending

    def prime_existing_due_tasks(self, now: Optional[datetime] = None) -> int:
        current, pending = self._pending_due_tasks(now)
        for task_id, due_key, _ in pending:
            mark_todoist_notification_sent(self._db_path, task_id, due_key, current.isoformat())
        return len(pending)

    def poll_once(self, now: Optional[datetime] = None) -> list[str]:
        current, pending = self._pending_due_tasks(now)
        if not pending:
            return []

        notifications: list[str] = []
        for task_id, due_key, task in pending:
            message = format_todoist_reminder_message(task, now=current)
            mark_todoist_notification_sent(self._db_path, task_id, due_key, current.isoformat())
            notifications.append(message)
        return notifications

    def _run(self) -> None:
        try:
            self.prime_existing_due_tasks()
            self._last_error = ""
        except Exception as exc:
            current_error = str(exc)
            if current_error and current_error != self._last_error:
                self._messages.put(f"agent> Todoist reminder error: {current_error}")
            self._last_error = current_error

        while not self._stop_event.is_set():
            try:
                for message in self.poll_once():
                    self._messages.put(message)
                self._last_error = ""
            except Exception as exc:
                current_error = str(exc)
                if current_error and current_error != self._last_error:
                    self._messages.put(f"agent> Todoist reminder error: {current_error}")
                self._last_error = current_error
            self._stop_event.wait(self._poll_interval_sec)
