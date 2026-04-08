import argparse
import json
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client

from app.config import load_env_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo MCP client for Todoist server in app/mcp_servers/todoist_server.py."
    )
    parser.add_argument(
        "--action",
        choices=["list", "create", "complete"],
        default="list",
        help="Which MCP tool to call after initialize/list_tools.",
    )
    parser.add_argument("--project-id", default="", help="Todoist project id for list/create.")
    parser.add_argument("--limit", type=int, default=5, help="Limit for list action.")
    parser.add_argument("--content", default="", help="Task content for create action.")
    parser.add_argument("--due-string", default="", help="Natural language due date for create action.")
    parser.add_argument("--task-id", default="", help="Task id for complete action.")
    return parser


def _build_tool_call(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.action == "list":
        return "list_tasks", {"project_id": args.project_id, "limit": args.limit}
    if args.action == "create":
        if not args.content.strip():
            raise ValueError("--content is required for --action create")
        return "create_task", {
            "content": args.content.strip(),
            "project_id": args.project_id.strip(),
            "due_string": args.due_string.strip(),
        }
    if not args.task_id.strip():
        raise ValueError("--task-id is required for --action complete")
    return "complete_task", {"task_id": args.task_id.strip()}


async def _run(args: argparse.Namespace) -> None:
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "app.mcp_servers.todoist_server"],
    )

    print("1) Starting Todoist MCP server via stdio...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        print("2) Opening MCP session...")
        async with ClientSession(read_stream, write_stream) as session:
            print("3) Calling initialize()...")
            init_result = await session.initialize()
            print(f"   server={init_result.serverInfo.name}, protocol={init_result.protocolVersion}")

            print("4) Calling list_tools()...")
            tools_result = await session.list_tools()
            for idx, tool in enumerate(tools_result.tools, start=1):
                print(f"   {idx}. {tool.name} - {tool.description}")

            tool_name, tool_args = _build_tool_call(args)
            print(f"5) Calling tool: {tool_name} with args={tool_args}")
            result = await session.call_tool(tool_name, tool_args)
            print("6) Tool result:")
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        anyio.run(_run, args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
