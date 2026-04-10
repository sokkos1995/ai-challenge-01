from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


server = FastMCP(
    name="day-19-pipeline-tools-server",
    log_level="ERROR",
    instructions=(
        "Local MCP server for day 19 pipeline. "
        "Provides summarize_workload and save_summary_to_file tools."
    ),
)


def _normalize_tasks(tasks_payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = tasks_payload.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if isinstance(task, dict)]


def _task_due_bucket(task: dict[str, Any], today: date) -> str:
    due = task.get("due")
    if not isinstance(due, dict):
        return "no_due"

    due_datetime_raw = str(due.get("datetime", "")).strip()
    if due_datetime_raw:
        normalized = due_datetime_raw.replace("Z", "+00:00")
        try:
            due_day = datetime.fromisoformat(normalized).date()
        except ValueError:
            due_day = today
        if due_day < today:
            return "overdue"
        if due_day == today:
            return "today"
        return "future"

    due_date_raw = str(due.get("date", "")).strip()
    if due_date_raw:
        date_part = due_date_raw.split("T", 1)[0]
        try:
            due_day = date.fromisoformat(date_part)
        except ValueError:
            due_day = today
        if due_day < today:
            return "overdue"
        if due_day == today:
            return "today"
        return "future"

    return "no_due"


def _has_due(task: dict[str, Any]) -> bool:
    due = task.get("due")
    if not isinstance(due, dict):
        return False
    return bool(str(due.get("date", "")).strip() or str(due.get("datetime", "")).strip())


@server.tool()
def summarize_workload(
    tasks_payload: dict[str, Any],
    source_command: str = "",
    filter_query: str = "",
) -> dict[str, Any]:
    """
    Build a short markdown workload summary from Todoist list_tasks payload.
    """
    tasks = _normalize_tasks(tasks_payload)
    today = date.today()

    counters = {"overdue": 0, "today": 0, "future": 0, "no_due": 0}
    for task in tasks:
        counters[_task_due_bucket(task, today)] += 1

    tasks_with_due = [task for task in tasks if _has_due(task)]
    top_tasks = []
    for task in tasks_with_due[:7]:
        content = str(task.get("content", "")).strip() or "(empty content)"
        due = task.get("due")
        due_display = "без срока"
        if isinstance(due, dict):
            due_display = str(due.get("string") or due.get("date") or due.get("datetime") or due_display)
        top_tasks.append(f"- {content} — {due_display}")

    command_display = source_command.strip() or "(not provided)"
    filter_display = filter_query.strip() or "(none)"

    summary_markdown = "\n".join(
        [
            "# Todoist workload summary",
            "",
            f"- Command: `{command_display}`",
            f"- Filter: `{filter_display}`",
            f"- Total active tasks: **{len(tasks)}**",
            f"- Overdue: **{counters['overdue']}**",
            f"- Due today: **{counters['today']}**",
            f"- Future: **{counters['future']}**",
            f"- No due date: **{counters['no_due']}**",
            "",
            "## Top tasks",
            *(top_tasks or ["- No tasks found."]),
            "",
        ]
    )

    return {
        "total": len(tasks),
        "counters": counters,
        "summary_markdown": summary_markdown,
    }


@server.tool()
def save_summary_to_file(
    summary_markdown: str,
    output_dir: str = "workload",
    filename: str = "",
) -> dict[str, Any]:
    """
    Save markdown summary into output directory and return file path.
    """
    safe_name = filename.strip()
    if not safe_name:
        safe_name = f"todoist-summary-{date.today().isoformat()}.md"
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name

    destination.write_text(summary_markdown.strip() + "\n", encoding="utf-8")
    return {"ok": True, "path": str(destination), "bytes": destination.stat().st_size}


@server.tool()
def echo_json(payload: dict[str, Any]) -> str:
    """Debug helper: echo incoming JSON payload."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    server.run(transport="stdio")
