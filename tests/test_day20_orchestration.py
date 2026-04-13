import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeworks.src.day_20_orchestration import (
    MultiServerOrchestrator,
    OrchestrationState,
    ToolInvoker,
)


class _FakeInvoker(ToolInvoker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, server_id: str, tool_name: str, args: dict) -> dict:
        self.calls.append((server_id, tool_name, args))
        if tool_name == "list_tasks":
            return {"count": 3, "tasks": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}
        if tool_name == "add":
            return {"result": int(args["a"]) + int(args["b"])}
        if tool_name == "create_task":
            return {"id": "task-42", "content": args["content"]}
        return {}


def test_build_tool_registry_rejects_duplicate_tool_names() -> None:
    with pytest.raises(RuntimeError, match="Duplicate tool names"):
        MultiServerOrchestrator.build_tool_registry(
            {
                "server-a": ["list_tasks", "add"],
                "server-b": ["add"],
            }
        )


@pytest.mark.anyio
async def test_orchestration_routes_tools_to_expected_servers_and_order() -> None:
    registry = MultiServerOrchestrator.build_tool_registry(
        {
            "user-day-16-local-server": ["hello", "add"],
            "user-todoist-local": ["list_tasks", "create_task", "complete_task"],
        }
    )
    orchestrator = MultiServerOrchestrator(registry)
    invoker = _FakeInvoker()
    state = OrchestrationState(command="какие у меня задачи на сегодня")

    result = await orchestrator.run(state, invoker)

    assert [(server, tool) for server, tool, _ in invoker.calls] == [
        ("user-todoist-local", "list_tasks"),
        ("user-day-16-local-server", "add"),
        ("user-todoist-local", "create_task"),
    ]
    assert result["todoist_count"] == 3
    assert result["metric_sum"] == 5
    assert result["created_task_id"] == "task-42"
    assert [item["tool_name"] for item in result["trace"]] == ["list_tasks", "add", "create_task"]
