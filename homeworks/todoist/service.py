from __future__ import annotations

from pathlib import Path

from .ai import fallback_plan, generate_plan
from .models import Task
from .storage import JsonTaskStorage


class TaskTrackerService:
    def __init__(self, db_path: Path) -> None:
        self.storage = JsonTaskStorage(db_path)

    def add_task(
        self,
        title: str,
        description: str = "",
        *,
        priority: str = "medium",
        due_date: str | None = None,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> Task:
        tasks = self.storage.load_tasks()
        task = Task.create(
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            source=source,
            tags=tags,
        )
        tasks.append(task)
        self.storage.save_tasks(tasks)
        return task

    def list_tasks(self) -> list[Task]:
        tasks = self.storage.load_tasks()
        return sorted(tasks, key=lambda item: (item.status == "done", item.created_at))

    def complete_task(self, task_id: str) -> Task:
        tasks = self.storage.load_tasks()
        updated: list[Task] = []
        changed: Task | None = None
        for task in tasks:
            if task.id == task_id:
                changed = task.with_status("done")
                updated.append(changed)
            else:
                updated.append(task)
        if changed is None:
            raise RuntimeError(f"Task not found: {task_id}")
        self.storage.save_tasks(updated)
        return changed

    def delete_task(self, task_id: str) -> Task:
        tasks = self.storage.load_tasks()
        remaining: list[Task] = []
        removed: Task | None = None
        for task in tasks:
            if task.id == task_id:
                removed = task
            else:
                remaining.append(task)
        if removed is None:
            raise RuntimeError(f"Task not found: {task_id}")
        self.storage.save_tasks(remaining)
        return removed

    def plan_goal_with_ai(self, goal: str) -> list[Task]:
        try:
            plan = generate_plan(goal)
            source = "ai_llm"
        except Exception:
            plan = fallback_plan(goal)
            source = "ai_fallback"

        parent = self.add_task(
            title=plan.title,
            description=f"AI decomposition for goal: {goal}",
            priority=plan.priority if plan.priority in {"low", "medium", "high"} else "medium",
            due_date=plan.due_date,
            source=source,
            tags=plan.tags or ["ai"],
        )
        created = [parent]
        for subtask in plan.subtasks:
            child = self.add_task(
                title=subtask,
                description=f"Subtask for {parent.id}",
                priority=parent.priority,
                due_date=parent.due_date,
                source=source,
                tags=(plan.tags or []) + ["subtask", parent.id],
            )
            created.append(child)
        return created
