import os
import sqlite3

from app.models import LongTermMemory, ShortTermMemory, TaskState, WorkingMemory


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
            step INTEGER NOT NULL,
            total INTEGER NOT NULL
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
        row = conn.execute("SELECT task, state, step, total FROM work_task_state WHERE id = 1").fetchone()
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
            state=str(row[1]),
            step=int(row[2]),
            total=int(row[3]),
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
        task = state.current_task
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO work_task_state (id, task, state, step, total)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                task=excluded.task,
                state=excluded.state,
                step=excluded.step,
                total=excluded.total
            """,
            (task.task, task.state, task.step, task.total),
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
    finally:
        conn.close()
    return LongTermMemory(
        profile={str(k): str(v) for k, v in profile_rows},
        decisions=[str(r[0]) for r in decision_rows],
        knowledge={str(k): str(v) for k, v in knowledge_rows},
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
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
