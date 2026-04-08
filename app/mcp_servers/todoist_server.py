import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import load_env_file
from app.mcp_servers._http import request_json

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
) -> dict[str, Any] | list[Any]:
    primary_url = f"{TODOIST_API_BASE}{path}"
    try:
        return request_json(method=method, url=primary_url, token=token, payload=payload)
    except RuntimeError as exc:
        if "HTTP 410" not in str(exc):
            raise
    fallback_url = f"{TODOIST_API_BASE_FALLBACK}{path}"
    return request_json(method=method, url=fallback_url, token=token, payload=payload)


@server.tool()
def list_tasks(project_id: str = "", limit: int = 20) -> dict[str, Any]:
    """List active Todoist tasks (optionally scoped to a project)."""
    token = _todoist_token()
    capped_limit = max(1, min(limit, 100))
    tasks = _todoist_request(method="GET", path="/tasks", token=token)
    filtered = tasks if isinstance(tasks, list) else []
    if project_id:
        filtered = [task for task in filtered if str(task.get("project_id", "")) == project_id]
    return {
        "count": min(len(filtered), capped_limit),
        "tasks": filtered[:capped_limit],
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
