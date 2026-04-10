from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client

# Allow running as a plain script: `python3 homeworks/src/day_19_pipeline.py`
# by adding repository root to sys.path for `import app...`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import load_env_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 19 MCP pipeline: Todoist -> summarize -> saveToFile.")
    parser.add_argument(
        "--command",
        default="какие у меня задачи на сегодня",
        help="Natural-language command that triggers the pipeline.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Optional explicit Todoist filter. If empty, inferred from --command.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max tasks from Todoist list_tasks.")
    parser.add_argument(
        "--output-file",
        default="",
        help="Output markdown filename in workload/ directory.",
    )
    return parser


def _infer_filter_query(command: str, explicit_filter: str) -> str:
    clean_explicit = explicit_filter.strip()
    if clean_explicit:
        return clean_explicit

    normalized = command.lower()
    if "сегодня" in normalized or "today" in normalized:
        return "today"
    return ""


def _parse_text_as_json(content_text: str) -> dict[str, Any]:
    stripped = content_text.strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return {"raw_text": stripped}
    return loaded if isinstance(loaded, dict) else {"raw_payload": loaded}


def _extract_tool_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    if isinstance(result, dict):
        result_structured = result.get("structuredContent") or result.get("structured_content")
        if isinstance(result_structured, dict):
            return result_structured

    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if isinstance(text, str):
                return _parse_text_as_json(text)
    return {}


def _is_due_today(task: dict[str, Any], today: date) -> bool:
    due = task.get("due")
    if not isinstance(due, dict):
        return False

    due_datetime_raw = str(due.get("datetime", "")).strip()
    if due_datetime_raw:
        normalized = due_datetime_raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date() == today
        except ValueError:
            return False

    due_date_raw = str(due.get("date", "")).strip()
    if due_date_raw:
        date_part = due_date_raw.split("T", 1)[0]
        try:
            return date.fromisoformat(date_part) == today
        except ValueError:
            return False

    return False


def _enforce_due_filter(tasks_payload: dict[str, Any], filter_query: str) -> dict[str, Any]:
    tasks = tasks_payload.get("tasks", [])
    if not isinstance(tasks, list):
        return {"count": 0, "tasks": []}

    clean_filter = filter_query.strip().lower()
    if clean_filter != "today":
        return {
            "count": len([task for task in tasks if isinstance(task, dict)]),
            "tasks": [task for task in tasks if isinstance(task, dict)],
        }

    today = date.today()
    filtered = [task for task in tasks if isinstance(task, dict) and _is_due_today(task, today)]
    return {"count": len(filtered), "tasks": filtered}


async def _call_todoist_list_tasks(filter_query: str, limit: int) -> dict[str, Any]:
    server_params = StdioServerParameters(command="python3", args=["-m", "app.mcp_servers.todoist_server"])
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "list_tasks",
                {
                    "filter_query": filter_query,
                    "limit": limit if limit <= 0 else max(1, limit),
                },
            )
            return _extract_tool_payload(result)


async def _run_pipeline(args: argparse.Namespace) -> None:
    command = args.command.strip()
    filter_query = _infer_filter_query(command, args.filter)
    # For "today" we must inspect the full active set before strict due-date filtering,
    # otherwise recurring tasks can be truncated out by a small limit.
    todoist_limit = 0 if filter_query.strip().lower() == "today" else args.limit
    raw_tasks_payload = await _call_todoist_list_tasks(filter_query=filter_query, limit=todoist_limit)
    tasks_payload = _enforce_due_filter(raw_tasks_payload, filter_query)

    tools_server_path = Path(__file__).with_name("day_19_pipeline_tools_server.py")
    tools_params = StdioServerParameters(command="python3", args=[str(tools_server_path)])

    async with stdio_client(tools_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            summarize_result = await session.call_tool(
                "summarize_workload",
                {
                    "tasks_payload": tasks_payload,
                    "source_command": command,
                    "filter_query": filter_query,
                },
            )
            summary_payload = _extract_tool_payload(summarize_result)
            summary_markdown = str(summary_payload.get("summary_markdown", "")).strip()
            if not summary_markdown:
                raise RuntimeError("summarize_workload returned empty summary_markdown.")

            save_result = await session.call_tool(
                "save_summary_to_file",
                {
                    "summary_markdown": summary_markdown,
                    "output_dir": "workload",
                    "filename": args.output_file.strip(),
                },
            )
            save_payload = _extract_tool_payload(save_result)

    print("Pipeline completed successfully.")
    print(f"Command: {command}")
    print(f"Filter query: {filter_query or '(none)'}")
    print(f"Todoist tasks loaded: {raw_tasks_payload.get('count', 0)}")
    print(f"Tasks after due filter: {tasks_payload.get('count', 0)}")
    print(f"Saved file: {save_payload.get('path', '(unknown)')}")
    print()
    print(summary_markdown)


def main() -> None:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args()
    anyio.run(_run_pipeline, args)


if __name__ == "__main__":
    main()
