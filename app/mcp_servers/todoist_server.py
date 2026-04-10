import os
from datetime import date, datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import load_env_file
from app.mcp_servers._http import add_query, request_json

TODOIST_API_BASE = "https://api.todoist.com/rest/v2"
TODOIST_API_BASE_FALLBACK = "https://api.todoist.com/api/v1"

server = FastMCP(
    name="todoist-mcp-server",
    log_level="ERROR",
    instructions=(
        "MCP server for Todoist REST API. "
        "Exposes minimal tools to list, create, and complete tasks."
    ),
)


def _todoist_token() -> str:
    token = os.getenv("TODOIST_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TODOIST_API_TOKEN in environment or .env.")
    return token


def _todoist_request(
    *,
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, Any] | list[Any]:
    primary_url = add_query(f"{TODOIST_API_BASE}{path}", query or {})
    try:
        return request_json(method=method, url=primary_url, token=token, payload=payload)
    except RuntimeError as exc:
        if "HTTP 410" not in str(exc):
            raise
    fallback_url = add_query(f"{TODOIST_API_BASE_FALLBACK}{path}", query or {})
    return request_json(method=method, url=fallback_url, token=token, payload=payload)


def _unwrap_tasks(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        items = payload.get("tasks")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _next_cursor(payload: dict[str, Any] | list[Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    cursor = payload.get("next_cursor")
    return str(cursor).strip() if cursor else ""


def _extract_due_date(task: dict[str, Any]) -> date | None:
    due = task.get("due")
    if not isinstance(due, dict):
        return None

    due_datetime_raw = str(due.get("datetime", "")).strip()
    if due_datetime_raw:
        normalized = due_datetime_raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date()
        except ValueError:
            return None

    due_date_raw = str(due.get("date", "")).strip()
    if not due_date_raw:
        return None
    date_part = due_date_raw.split("T", 1)[0]
    try:
        return date.fromisoformat(date_part)
    except ValueError:
        return None


def _list_all_tasks(token: str, limit: int, filter_query: str = "") -> list[dict[str, Any]]:
    unlimited = limit <= 0
    capped_limit = max(1, min(limit, 5000)) if not unlimited else 5000
    aggregated: list[dict[str, Any]] = []
    cursor = ""
    clean_filter = filter_query.strip()

    while True:
        remaining = max(0, capped_limit - len(aggregated))
        page_limit = 100 if unlimited else max(1, min(100, remaining))
        query = {"limit": str(page_limit)}
        if clean_filter:
            query["filter"] = clean_filter
        if cursor:
            query["cursor"] = cursor
        payload = _todoist_request(method="GET", path="/tasks", token=token, query=query)
        page_items = _unwrap_tasks(payload)
        aggregated.extend(page_items)
        if not unlimited and len(aggregated) >= capped_limit:
            return aggregated[:capped_limit]
        cursor = _next_cursor(payload)
        if not cursor or not page_items:
            break

    return aggregated if unlimited else aggregated[:capped_limit]


@server.tool()
def list_tasks(project_id: str = "", limit: int = 20, filter_query: str = "") -> dict[str, Any]:
    """List active Todoist tasks (optionally scoped to a project)."""
    token = _todoist_token()
    clean_filter = filter_query.strip().lower()
    source_filter = "" if clean_filter == "today" else filter_query
    tasks = _list_all_tasks(token, limit, source_filter)
    filtered = tasks
    if project_id:
        filtered = [task for task in filtered if str(task.get("project_id", "")) == project_id]
    if clean_filter == "today":
        today = date.today()
        filtered = [task for task in filtered if _extract_due_date(task) == today]
    return {
        "count": len(filtered),
        "tasks": filtered,
    }


@server.tool()
def create_task(content: str, project_id: str = "", due_string: str = "") -> dict[str, Any]:
    """Create a Todoist task with optional project and due text."""
    token = _todoist_token()
    payload: dict[str, Any] = {"content": content}
    if project_id:
        payload["project_id"] = project_id
    if due_string:
        payload["due_string"] = due_string

    created = _todoist_request(method="POST", path="/tasks", token=token, payload=payload)
    return created if isinstance(created, dict) else {"raw": created}


@server.tool()
def complete_task(task_id: str) -> dict[str, Any]:
    """Close (complete) Todoist task by id."""
    token = _todoist_token()
    _todoist_request(method="POST", path=f"/tasks/{task_id}/close", token=token)
    return {"ok": True, "task_id": task_id}


if __name__ == "__main__":
    load_env_file()
    server.run(transport="stdio")
