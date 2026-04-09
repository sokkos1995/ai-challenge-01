from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.services.todoist_reminder_service import (
    TodoistMcpClient,
    TodoistReminderService,
    due_todoist_tasks,
    format_todoist_reminder_message,
)


def _task(task_id: str, content: str, due_iso: str) -> dict:
    return {
        "id": task_id,
        "content": content,
        "due": {
            "datetime": due_iso,
            "timezone": "UTC",
            "string": "today at noon",
        },
    }


def test_filters_only_due_tasks_with_explicit_time() -> None:
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    tasks = [
        _task("1", "Сделать отчет", "2026-04-09T11:55:00+00:00"),
        _task("2", "Позвонить", "2026-04-09T12:10:00+00:00"),
        {"id": "3", "content": "Без времени", "due": {"date": "2026-04-09"}},
    ]

    due = due_todoist_tasks(tasks, now=now)

    assert [item["id"] for item in due] == ["1"]


def test_formats_same_day_reminder_message() -> None:
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)

    message = format_todoist_reminder_message(
        _task("1", "Сделать отчет", "2026-04-09T12:00:00+00:00"),
        now=now,
    )

    assert message == "agent> Todoist: сделать сегодня в 12 00 Сделать отчет"


def test_persists_sent_notifications_and_avoids_duplicates(tmp_path) -> None:
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    tasks = [_task("1", "Сделать отчет", "2026-04-09T12:00:00+00:00")]
    service = TodoistReminderService(
        db_path=str(tmp_path / "todoist_reminders.db"),
        poll_interval_sec=5,
        task_fetcher=lambda: tasks,
    )

    first = service.poll_once(now=now)
    second = service.poll_once(now=now + timedelta(minutes=1))

    assert first == ["agent> Todoist: сделать сегодня в 12 00 Сделать отчет"]
    assert second == []


def test_repeating_task_reappears_when_due_key_changes(tmp_path) -> None:
    state = {
        "tasks": [_task("1", "Полить цветы", "2026-04-09T12:00:00+00:00")],
    }
    service = TodoistReminderService(
        db_path=str(tmp_path / "todoist_recurring.db"),
        poll_interval_sec=5,
        task_fetcher=lambda: state["tasks"],
    )

    first = service.poll_once(now=datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc))
    state["tasks"] = [_task("1", "Полить цветы", "2026-04-10T12:00:00+00:00")]
    second = service.poll_once(now=datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc))

    assert first == ["agent> Todoist: сделать сегодня в 12 00 Полить цветы"]
    assert second == ["agent> Todoist: сделать сегодня в 12 00 Полить цветы"]


def test_accepts_due_date_with_time_component() -> None:
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    tasks = [
        {
            "id": "1",
            "content": "Сделать отчет",
            "due": {"date": "2026-04-09T11:59:00", "string": "today at 11:59"},
        }
    ]

    due = due_todoist_tasks(tasks, now=now)

    assert [item["id"] for item in due] == ["1"]


def test_default_reminder_fetch_uses_due_filter(tmp_path) -> None:
    calls: list[tuple[int, str]] = []

    def _fake_list_tasks(
        self,
        limit: int = 0,
        filter_query: str = "",
        *,
        wait_for_lock: bool = True,
    ) -> list[dict]:
        calls.append((limit, filter_query, wait_for_lock))
        return []

    with patch.object(TodoistMcpClient, "list_tasks", _fake_list_tasks):
        service = TodoistReminderService(
            db_path=str(tmp_path / "todoist_reminders.db"),
            poll_interval_sec=5,
        )
        assert service.poll_once() == []

    assert calls == [(0, "today | overdue", False)]


def test_prime_existing_due_tasks_suppresses_startup_backlog(tmp_path) -> None:
    now = datetime(2026, 4, 9, 12, 33, tzinfo=timezone.utc)
    state = {
        "tasks": [_task("1", "Старый reminder", "2026-04-09T12:00:00+00:00")],
    }
    service = TodoistReminderService(
        db_path=str(tmp_path / "todoist_reminders.db"),
        poll_interval_sec=5,
        task_fetcher=lambda: state["tasks"],
    )

    primed = service.prime_existing_due_tasks(now=now)
    startup_poll = service.poll_once(now=now)

    state["tasks"] = [_task("1", "Новый reminder", "2026-04-09T13:02:00+00:00")]
    future_poll = service.poll_once(now=datetime(2026, 4, 9, 13, 2, tzinfo=timezone.utc))

    assert primed == 1
    assert startup_poll == []
    assert future_poll == ["agent> Todoist: сделать сегодня в 13 02 Новый reminder"]


def test_list_tasks_can_skip_when_lock_is_busy() -> None:
    client = TodoistMcpClient()
    acquired = client._stdio_lock.acquire(blocking=False)
    assert acquired is True
    try:
        assert client.list_tasks(limit=1, filter_query="today", wait_for_lock=False) == []
    finally:
        client._stdio_lock.release()
