"""Unit tests for app.storage SQLite helpers (offline, tmp_path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models import LongTermMemory, ShortTermMemory, TaskState, WorkingMemory
from app.storage import (
    ensure_user_record,
    has_todoist_notification,
    load_chat_messages,
    load_chat_summary,
    load_long_term_memory,
    load_short_term_memory,
    load_user_profile,
    load_working_memory,
    mark_todoist_notification_sent,
    save_chat_state,
    save_long_term_memory,
    save_short_term_memory,
    save_working_memory,
    set_user_interview_completed,
    upsert_user_profile_entries,
)


def test_chat_state_roundtrip(tmp_path: Path) -> None:
    path = str(tmp_path / "chat.db")
    assert load_chat_messages(path) == []
    assert load_chat_summary(path) == ""

    save_chat_state(
        path,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "summary-1",
    )
    assert load_chat_messages(path) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert load_chat_summary(path) == "summary-1"

    save_chat_state(path, [{"role": "user", "content": "again"}], "summary-2")
    assert load_chat_messages(path) == [{"role": "user", "content": "again"}]
    assert load_chat_summary(path) == "summary-2"


def test_memory_layers_roundtrip(tmp_path: Path) -> None:
    base = str(tmp_path / "memory")

    short = ShortTermMemory(
        dialog_tail=[{"role": "user", "content": "q"}],
        notes=["f1"],
    )
    save_short_term_memory(base, short)
    loaded_short = load_short_term_memory(base)
    assert loaded_short.dialog_tail == short.dialog_tail
    assert loaded_short.notes == ["f1"]

    task = TaskState(
        task="Ship",
        state="PLANNING",
        plan_status="DRAFT",
        validation_status="PENDING",
        notes=["n"],
        plan=["step-1"],
    )
    work = WorkingMemory(current_task=task)
    save_working_memory(base, work)
    loaded_work = load_working_memory(base)
    assert loaded_work.current_task.task == "Ship"
    assert loaded_work.current_task.plan == ["step-1"]
    assert loaded_work.current_task.notes == ["n"]

    long = LongTermMemory(
        profile={"role": "dev"},
        decisions=["use sqlite"],
        knowledge={"k": "v"},
        invariants=["never leak secrets"],
    )
    save_long_term_memory(base, long)
    loaded_long = load_long_term_memory(base)
    assert loaded_long.profile == {"role": "dev"}
    assert loaded_long.decisions == ["use sqlite"]
    assert loaded_long.knowledge == {"k": "v"}
    assert loaded_long.invariants == ["never leak secrets"]


def test_user_profile_and_todoist_notifications(tmp_path: Path) -> None:
    users = str(tmp_path / "users.db")
    assert ensure_user_record(users, "alice") is True
    assert ensure_user_record(users, "alice") is False

    upsert_user_profile_entries(users, "alice", {"role": "eng", "stack": ""})
    profile, completed = load_user_profile(users, "alice")
    assert profile == {"role": "eng"}
    assert completed is False

    set_user_interview_completed(users, "alice", True)
    _, completed = load_user_profile(users, "alice")
    assert completed is True

    reminders = str(tmp_path / "reminders.db")
    assert has_todoist_notification(reminders, "1", "due-a") is False
    mark_todoist_notification_sent(reminders, "1", "due-a", "2026-07-23T10:00:00+00:00")
    assert has_todoist_notification(reminders, "1", "due-a") is True
    mark_todoist_notification_sent(reminders, "1", "due-a", "2026-07-23T11:00:00+00:00")
    assert has_todoist_notification(reminders, "1", "due-a") is True

    with pytest.raises(ValueError):
        ensure_user_record(users, "  ")
