import os
import sqlite3

from app.models import LongTermMemory, ShortTermMemory, TaskState, WorkingMemory
from app.task_state_machine import normalize_plan_status, normalize_task_stage, normalize_validation_status


def _ensure_db_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _init_chat_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            summary TEXT NOT NULL
        )
        """
    )


def load_chat_messages(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        rows = conn.execute("SELECT role, content FROM chat_messages ORDER BY id ASC").fetchall()
    finally:
        conn.close()
    return [{"role": str(role), "content": str(content)} for role, content in rows]


def load_chat_summary(path: str) -> str:
    if not os.path.exists(path):
        return ""
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        row = conn.execute("SELECT summary FROM chat_summary WHERE id = 1").fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] is not None else ""


def save_chat_state(path: str, messages: list[dict[str, str]], summary: str) -> None:
    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_chat_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM chat_messages")
        conn.executemany(
            "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
            [(m["role"], m["content"]) for m in messages],
        )
        conn.execute(
            """
            INSERT INTO chat_summary (id, summary) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET summary=excluded.summary
            """,
            (summary,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _short_memory_path(base_path: str) -> str:
    return f"{base_path}.short.db"


def _work_memory_path(base_path: str) -> str:
    return f"{base_path}.work.db"


def _long_memory_path(base_path: str) -> str:
    return f"{base_path}.long.db"


def _init_short_memory_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS short_term_dialog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS short_term_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL
        )
        """
    )


def _init_work_memory_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_task_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            task TEXT NOT NULL,
            state TEXT NOT NULL,
            plan_status TEXT NOT NULL DEFAULT 'DRAFT',
            validation_status TEXT NOT NULL DEFAULT 'PENDING',
            paused INTEGER NOT NULL DEFAULT 0,
            step INTEGER NOT NULL,
            total INTEGER NOT NULL,
            expected_action TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_plan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_done (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL
        )
        """
    )


def _table_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def _migrate_work_memory_db(conn: sqlite3.Connection) -> None:
    if not _table_has_column(conn, "work_task_state", "plan_status"):
        conn.execute(
            "ALTER TABLE work_task_state ADD COLUMN plan_status TEXT NOT NULL DEFAULT 'DRAFT'"
        )
    if not _table_has_column(conn, "work_task_state", "validation_status"):
        conn.execute(
            "ALTER TABLE work_task_state ADD COLUMN validation_status TEXT NOT NULL DEFAULT 'PENDING'"
        )
    if not _table_has_column(conn, "work_task_state", "paused"):
        conn.execute(
            "ALTER TABLE work_task_state ADD COLUMN paused INTEGER NOT NULL DEFAULT 0"
        )
    if not _table_has_column(conn, "work_task_state", "expected_action"):
        conn.execute(
            "ALTER TABLE work_task_state ADD COLUMN expected_action TEXT NOT NULL DEFAULT ''"
        )


def _init_long_memory_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_knowledge (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_invariants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invariant_text TEXT NOT NULL
        )
        """
    )


def _init_user_profile_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            interview_completed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile_entries (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
        """
    )


def _init_todoist_reminder_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS todoist_reminder_notifications (
            task_id TEXT NOT NULL,
            due_key TEXT NOT NULL,
            notified_at TEXT NOT NULL,
            PRIMARY KEY (task_id, due_key)
        )
        """
    )


def load_short_term_memory(base_path: str) -> ShortTermMemory:
    path = _short_memory_path(base_path)
    if not os.path.exists(path):
        return ShortTermMemory()
    conn = sqlite3.connect(path)
    try:
        _init_short_memory_db(conn)
        rows = conn.execute("SELECT role, content FROM short_term_dialog ORDER BY id ASC").fetchall()
        notes_rows = conn.execute("SELECT note FROM short_term_notes ORDER BY id ASC").fetchall()
    finally:
        conn.close()
    return ShortTermMemory(
        dialog_tail=[{"role": str(role), "content": str(content)} for role, content in rows],
        notes=[str(r[0]) for r in notes_rows],
    )


def save_short_term_memory(base_path: str, state: ShortTermMemory) -> None:
    path = _short_memory_path(base_path)
    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_short_memory_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM short_term_dialog")
        conn.executemany(
            "INSERT INTO short_term_dialog (role, content) VALUES (?, ?)",
            [(m["role"], m["content"]) for m in state.dialog_tail],
        )
        conn.execute("DELETE FROM short_term_notes")
        conn.executemany(
            "INSERT INTO short_term_notes (note) VALUES (?)",
            [(note,) for note in state.notes],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_working_memory(base_path: str) -> WorkingMemory:
    path = _work_memory_path(base_path)
    if not os.path.exists(path):
        return WorkingMemory()
    conn = sqlite3.connect(path)
    try:
        _init_work_memory_db(conn)
        _migrate_work_memory_db(conn)
        row = conn.execute(
            """
            SELECT task, state, plan_status, validation_status, paused, step, total, expected_action
            FROM work_task_state
            WHERE id = 1
            """
        ).fetchone()
        plan_rows = conn.execute("SELECT item FROM work_plan ORDER BY id ASC").fetchall()
        done_rows = conn.execute("SELECT item FROM work_done ORDER BY id ASC").fetchall()
        notes_rows = conn.execute("SELECT note FROM work_notes ORDER BY id ASC").fetchall()
    finally:
        conn.close()

    if row is None:
        task_state = TaskState()
    else:
        task_state = TaskState(
            task=str(row[0]),
            state=normalize_task_stage(str(row[1])),
            plan_status=normalize_plan_status(str(row[2])),
            validation_status=normalize_validation_status(str(row[3])),
            paused=bool(int(row[4])),
            step=int(row[5]),
            total=int(row[6]),
            expected_action=str(row[7]),
            plan=[str(r[0]) for r in plan_rows],
            done=[str(r[0]) for r in done_rows],
            notes=[str(r[0]) for r in notes_rows],
        )
    return WorkingMemory(current_task=task_state)


def save_working_memory(base_path: str, state: WorkingMemory) -> None:
    path = _work_memory_path(base_path)
    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_work_memory_db(conn)
        _migrate_work_memory_db(conn)
        task = state.current_task
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO work_task_state (
                id, task, state, plan_status, validation_status, paused, step, total, expected_action
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task=excluded.task,
                state=excluded.state,
                plan_status=excluded.plan_status,
                validation_status=excluded.validation_status,
                paused=excluded.paused,
                step=excluded.step,
                total=excluded.total,
                expected_action=excluded.expected_action
            """,
            (
                task.task,
                normalize_task_stage(task.state),
                normalize_plan_status(task.plan_status),
                normalize_validation_status(task.validation_status),
                1 if task.paused else 0,
                task.step,
                task.total,
                task.expected_action,
            ),
        )
        conn.execute("DELETE FROM work_plan")
        conn.executemany("INSERT INTO work_plan (item) VALUES (?)", [(item,) for item in task.plan])
        conn.execute("DELETE FROM work_done")
        conn.executemany("INSERT INTO work_done (item) VALUES (?)", [(item,) for item in task.done])
        conn.execute("DELETE FROM work_notes")
        conn.executemany("INSERT INTO work_notes (note) VALUES (?)", [(note,) for note in task.notes])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_long_term_memory(base_path: str) -> LongTermMemory:
    path = _long_memory_path(base_path)
    if not os.path.exists(path):
        return LongTermMemory()
    conn = sqlite3.connect(path)
    try:
        _init_long_memory_db(conn)
        profile_rows = conn.execute("SELECT key, value FROM long_profile ORDER BY key ASC").fetchall()
        decision_rows = conn.execute("SELECT decision FROM long_decisions ORDER BY id ASC").fetchall()
        knowledge_rows = conn.execute("SELECT key, value FROM long_knowledge ORDER BY key ASC").fetchall()
        invariant_rows = conn.execute(
            "SELECT invariant_text FROM long_invariants ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    return LongTermMemory(
        profile={str(k): str(v) for k, v in profile_rows},
        decisions=[str(r[0]) for r in decision_rows],
        knowledge={str(k): str(v) for k, v in knowledge_rows},
        invariants=[str(r[0]) for r in invariant_rows],
    )


def save_long_term_memory(base_path: str, state: LongTermMemory) -> None:
    path = _long_memory_path(base_path)
    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_long_memory_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM long_profile")
        conn.executemany(
            "INSERT INTO long_profile (key, value) VALUES (?, ?)",
            [(k, v) for k, v in state.profile.items()],
        )
        conn.execute("DELETE FROM long_decisions")
        conn.executemany(
            "INSERT INTO long_decisions (decision) VALUES (?)",
            [(d,) for d in state.decisions],
        )
        conn.execute("DELETE FROM long_knowledge")
        conn.executemany(
            "INSERT INTO long_knowledge (key, value) VALUES (?, ?)",
            [(k, v) for k, v in state.knowledge.items()],
        )
        conn.execute("DELETE FROM long_invariants")
        conn.executemany(
            "INSERT INTO long_invariants (invariant_text) VALUES (?)",
            [(item,) for item in state.invariants],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_user_record(path: str, user_id: str) -> bool:
    clean_user_id = user_id.strip()
    if not clean_user_id:
        raise ValueError("user_id must not be empty")

    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_user_profile_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT OR IGNORE INTO users (user_id, interview_completed) VALUES (?, 0)",
            (clean_user_id,),
        )
        conn.commit()
        return bool(cursor.rowcount)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_user_profile(path: str, user_id: str) -> tuple[dict[str, str], bool]:
    clean_user_id = user_id.strip()
    if not clean_user_id:
        raise ValueError("user_id must not be empty")
    if not os.path.exists(path):
        return {}, False

    conn = sqlite3.connect(path)
    try:
        _init_user_profile_db(conn)
        user_row = conn.execute(
            "SELECT interview_completed FROM users WHERE user_id = ?",
            (clean_user_id,),
        ).fetchone()
        if user_row is None:
            return {}, False
        profile_rows = conn.execute(
            """
            SELECT key, value
            FROM user_profile_entries
            WHERE user_id = ?
            ORDER BY key ASC
            """,
            (clean_user_id,),
        ).fetchall()
    finally:
        conn.close()

    return (
        {str(key): str(value) for key, value in profile_rows},
        bool(int(user_row[0])),
    )


def upsert_user_profile_entries(path: str, user_id: str, entries: dict[str, str]) -> None:
    clean_user_id = user_id.strip()
    if not clean_user_id:
        raise ValueError("user_id must not be empty")

    ensure_user_record(path, clean_user_id)
    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_user_profile_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        for key, value in entries.items():
            clean_key = key.strip()
            if not clean_key:
                continue
            clean_value = value.strip()
            if clean_value:
                conn.execute(
                    """
                    INSERT INTO user_profile_entries (user_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value
                    """,
                    (clean_user_id, clean_key, clean_value),
                )
            else:
                conn.execute(
                    "DELETE FROM user_profile_entries WHERE user_id = ? AND key = ?",
                    (clean_user_id, clean_key),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_user_interview_completed(path: str, user_id: str, completed: bool) -> None:
    clean_user_id = user_id.strip()
    if not clean_user_id:
        raise ValueError("user_id must not be empty")

    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_user_profile_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO users (user_id, interview_completed)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET interview_completed = excluded.interview_completed
            """,
            (clean_user_id, 1 if completed else 0),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def has_todoist_notification(path: str, task_id: str, due_key: str) -> bool:
    clean_task_id = task_id.strip()
    clean_due_key = due_key.strip()
    if not clean_task_id or not clean_due_key or not os.path.exists(path):
        return False

    conn = sqlite3.connect(path)
    try:
        _init_todoist_reminder_db(conn)
        row = conn.execute(
            """
            SELECT 1
            FROM todoist_reminder_notifications
            WHERE task_id = ? AND due_key = ?
            """,
            (clean_task_id, clean_due_key),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def mark_todoist_notification_sent(path: str, task_id: str, due_key: str, notified_at: str) -> None:
    clean_task_id = task_id.strip()
    clean_due_key = due_key.strip()
    if not clean_task_id:
        raise ValueError("task_id must not be empty")
    if not clean_due_key:
        raise ValueError("due_key must not be empty")
    if not notified_at.strip():
        raise ValueError("notified_at must not be empty")

    _ensure_db_parent(path)
    conn = sqlite3.connect(path)
    try:
        _init_todoist_reminder_db(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT OR IGNORE INTO todoist_reminder_notifications (task_id, due_key, notified_at)
            VALUES (?, ?, ?)
            """,
            (clean_task_id, clean_due_key, notified_at.strip()),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
