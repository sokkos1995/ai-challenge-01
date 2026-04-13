from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client


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


def _infer_filter_query(command: str) -> str:
    normalized = command.lower()
    if "сегодня" in normalized or "today" in normalized:
        return "today"
    return ""


@dataclass
class ToolCallPlan:
    server_id: str
    tool_name: str
    args: dict[str, Any]
    reason: str


@dataclass
class OrchestrationState:
    command: str
    todoist_payload: dict[str, Any] | None = None
    metric_sum: int | None = None
    created_task: dict[str, Any] | None = None


class MultiServerOrchestrator:
    def __init__(self, tool_to_server: dict[str, str]) -> None:
        self._tool_to_server = dict(tool_to_server)

    @staticmethod
    def build_tool_registry(server_tools: dict[str, list[str]]) -> dict[str, str]:
        registry: dict[str, str] = {}
        duplicates: list[str] = []
        for server_id, tools in server_tools.items():
            for tool_name in tools:
                if tool_name in registry and registry[tool_name] != server_id:
                    duplicates.append(tool_name)
                    continue
                registry[tool_name] = server_id
        if duplicates:
            dup_display = ", ".join(sorted(set(duplicates)))
            raise RuntimeError(f"Duplicate tool names across servers: {dup_display}")
        return registry

    def _resolve_server(self, tool_name: str) -> str:
        server_id = self._tool_to_server.get(tool_name)
        if not server_id:
            raise RuntimeError(f"Unknown tool '{tool_name}' in routing registry.")
        return server_id

    def choose_next(self, state: OrchestrationState) -> ToolCallPlan | None:
        if state.todoist_payload is None:
            filter_query = _infer_filter_query(state.command)
            return ToolCallPlan(
                server_id=self._resolve_server("list_tasks"),
                tool_name="list_tasks",
                args={"filter_query": filter_query, "limit": 20},
                reason="Need source data from Todoist before any downstream steps.",
            )

        if state.metric_sum is None:
            task_count = int(state.todoist_payload.get("count", 0))
            return ToolCallPlan(
                server_id=self._resolve_server("add"),
                tool_name="add",
                args={"a": task_count, "b": 2},
                reason="Compute derived metric using second MCP server.",
            )

        if state.created_task is None:
            content = (
                f"[day20] Tasks analyzed: {state.todoist_payload.get('count', 0)}; "
                f"metric={state.metric_sum}"
            )
            return ToolCallPlan(
                server_id=self._resolve_server("create_task"),
                tool_name="create_task",
                args={"content": content, "due_string": "today"},
                reason="Persist orchestration result back to Todoist.",
            )

        return None

    async def run(self, state: OrchestrationState, invoker: "ToolInvoker") -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        while True:
            plan = self.choose_next(state)
            if plan is None:
                break
            payload = await invoker.call(plan.server_id, plan.tool_name, plan.args)
            trace.append(
                {
                    "server_id": plan.server_id,
                    "tool_name": plan.tool_name,
                    "args": plan.args,
                    "reason": plan.reason,
                }
            )

            if plan.tool_name == "list_tasks":
                state.todoist_payload = payload
            elif plan.tool_name == "add":
                state.metric_sum = int(payload.get("result", 0))
            elif plan.tool_name == "create_task":
                state.created_task = payload

        return {
            "command": state.command,
            "todoist_count": int((state.todoist_payload or {}).get("count", 0)),
            "metric_sum": state.metric_sum,
            "created_task_id": str((state.created_task or {}).get("id", "")),
            "trace": trace,
        }


class ToolInvoker:
    async def call(self, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class McpSessionInvoker(ToolInvoker):
    def __init__(self, sessions: dict[str, ClientSession]) -> None:
        self._sessions = sessions

    async def call(self, server_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        session = self._sessions.get(server_id)
        if session is None:
            raise RuntimeError(f"Session is not available for server '{server_id}'.")
        result = await session.call_tool(tool_name, args)
        return _extract_tool_payload(result)


async def _connect_and_run(command: str) -> dict[str, Any]:
    day16_script = Path(__file__).with_name("day_16_server.py")
    server_params = {
        "user-day-16-local-server": StdioServerParameters(command="python3", args=[str(day16_script)]),
        "user-todoist-local": StdioServerParameters(
            command="python3",
            args=["-m", "app.mcp_servers.todoist_server"],
        ),
    }

    async with stdio_client(server_params["user-day-16-local-server"]) as (
        day16_read,
        day16_write,
    ):
        async with ClientSession(day16_read, day16_write) as day16_session:
            await day16_session.initialize()
            day16_tools = await day16_session.list_tools()

            async with stdio_client(server_params["user-todoist-local"]) as (todo_read, todo_write):
                async with ClientSession(todo_read, todo_write) as todo_session:
                    await todo_session.initialize()
                    todo_tools = await todo_session.list_tools()

                    server_tools = {
                        "user-day-16-local-server": [tool.name for tool in day16_tools.tools],
                        "user-todoist-local": [tool.name for tool in todo_tools.tools],
                    }
                    registry = MultiServerOrchestrator.build_tool_registry(server_tools)
                    orchestrator = MultiServerOrchestrator(registry)
                    state = OrchestrationState(command=command)
                    invoker = McpSessionInvoker(
                        {
                            "user-day-16-local-server": day16_session,
                            "user-todoist-local": todo_session,
                        }
                    )
                    return await orchestrator.run(state, invoker)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 20 MCP orchestration across multiple servers.")
    parser.add_argument(
        "--command",
        default="какие у меня задачи на сегодня и создай задачу-резюме",
        help="Natural language command for the orchestration flow.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = anyio.run(_connect_and_run, args.command.strip())
    print("Day 20 orchestration completed.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
