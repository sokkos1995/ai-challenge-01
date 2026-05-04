from __future__ import annotations

import argparse
from pathlib import Path

from .service import TaskTrackerService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI task tracker demo service.")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parent / "data" / "tasks.json"),
        help="Path to JSON database file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add_cmd = sub.add_parser("add", help="Add manual task.")
    add_cmd.add_argument("title", help="Task title.")
    add_cmd.add_argument("--description", default="", help="Task description.")
    add_cmd.add_argument("--priority", default="medium", choices=["low", "medium", "high"])
    add_cmd.add_argument("--due-date", default=None, help="Due date in YYYY-MM-DD.")

    sub.add_parser("list", help="List all tasks.")

    done_cmd = sub.add_parser("done", help="Mark task as done.")
    done_cmd.add_argument("task_id", help="Task ID.")

    ai_cmd = sub.add_parser("ai_plan", help="Use AI to decompose goal.")
    ai_cmd.add_argument("goal", help="Real goal to automate.")

    sub.add_parser("demo", help="Run end-to-end demo scenario.")
    return parser


def print_tasks(tasks: list) -> None:
    if not tasks:
        print("No tasks yet.")
        return
    for task in tasks:
        print(
            f"- {task.id} [{task.status}] ({task.priority}) {task.title}"
            f" | due={task.due_date or '-'} | source={task.source}"
        )


def run_demo(service: TaskTrackerService) -> None:
    print("=== Demo: AI task tracker ===")
    created = service.plan_goal_with_ai("Подготовить и выпустить pet-проект в стор")
    print(f"Created by AI: {len(created)} tasks")
    print_tasks(created)
    print("")
    all_tasks = service.list_tasks()
    print("Current backlog:")
    print_tasks(all_tasks)
    print("")
    first = next((task for task in all_tasks if task.status != "done"), None)
    if first:
        updated = service.complete_task(first.id)
        print(f"Marked as done: {updated.id} - {updated.title}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = TaskTrackerService(db_path=Path(args.db))

    if args.command == "add":
        task = service.add_task(
            title=args.title,
            description=args.description,
            priority=args.priority,
            due_date=args.due_date,
            source="manual",
            tags=["manual"],
        )
        print(f"Created: {task.id}")
        return

    if args.command == "list":
        print_tasks(service.list_tasks())
        return

    if args.command == "done":
        updated = service.complete_task(args.task_id)
        print(f"Done: {updated.id}")
        return

    if args.command == "ai_plan":
        created = service.plan_goal_with_ai(args.goal)
        print(f"AI created: {len(created)} tasks")
        print_tasks(created)
        return

    if args.command == "demo":
        run_demo(service)
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
