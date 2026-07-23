"""Minimal demo web UI for day_38 Playwright smoke (stdlib only)."""

from __future__ import annotations

import argparse
import html
import json
import secrets
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .service import TaskTrackerService

DEMO_USER = "demo"
DEMO_PASSWORD = "demo123"
SESSION_COOKIE = "hw01_session"


def _page(title: str, body: str, *, flash: str = "") -> bytes:
    flash_html = (
        f'<p class="flash" data-testid="flash">{html.escape(flash)}</p>' if flash else ""
    )
    doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
    input, button {{ font: inherit; padding: 0.4rem 0.6rem; margin: 0.2rem 0; }}
    .task {{ display: flex; gap: 0.75rem; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #ddd; }}
    .flash {{ background: #eef6ff; padding: 0.5rem 0.75rem; border-radius: 4px; }}
    .error {{ color: #a40000; }}
    form.inline {{ display: inline; }}
  </style>
</head>
<body>
  <h1 data-testid="app-title">AI Task Tracker (day_38 demo)</h1>
  {flash_html}
  {body}
</body>
</html>"""
    return doc.encode("utf-8")


class TrackerWebApp:
    def __init__(self, db_path: Path) -> None:
        self.service = TaskTrackerService(db_path)
        self.sessions: dict[str, str] = {}

    def is_authed(self, token: str | None) -> bool:
        return bool(token and token in self.sessions)


def make_handler(app: TrackerWebApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _session_token(self) -> str | None:
            raw = self.headers.get("Cookie", "")
            cookie = SimpleCookie()
            cookie.load(raw)
            morsel = cookie.get(SESSION_COOKIE)
            return morsel.value if morsel else None

        def _redirect(self, location: str, *, set_cookie: str | None = None) -> None:
            self.send_response(303)
            self.send_header("Location", location)
            if set_cookie is not None:
                self.send_header("Set-Cookie", set_cookie)
            self.end_headers()

        def _redirect_flash(self, path: str, flash: str, *, set_cookie: str | None = None) -> None:
            self._redirect(f"{path}?flash={quote(flash)}", set_cookie=set_cookie)

        def _html(self, status: int, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_form(self) -> dict[str, str]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            parsed = parse_qs(raw, keep_blank_values=True)
            return {k: (v[0] if v else "") for k, v in parsed.items()}

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            token = self._session_token()
            query = parse_qs(urlparse(self.path).query)
            flash = (query.get("flash") or [""])[0]

            if path == "/health":
                body = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path in {"/", "/login"}:
                if app.is_authed(token):
                    self._redirect("/tasks")
                    return
                body = """
                <form method="post" action="/login" data-testid="login-form">
                  <label>Login<br/><input name="username" data-testid="login-username" value="demo"/></label><br/>
                  <label>Password<br/><input name="password" type="password" data-testid="login-password" value="demo123"/></label><br/>
                  <button type="submit" data-testid="login-submit">Войти</button>
                </form>
                <p data-testid="login-hint">Demo: demo / demo123</p>
                """
                self._html(200, _page("Login", body, flash=flash))
                return

            if path == "/tasks":
                if not app.is_authed(token):
                    self._redirect_flash("/login", "Требуется вход")
                    return
                tasks = app.service.list_tasks()
                rows = []
                for task in tasks:
                    complete_form = ""
                    if task.status != "done":
                        complete_form = f"""
                          <form class="inline" method="post" action="/tasks/complete">
                            <input type="hidden" name="task_id" value="{html.escape(task.id)}"/>
                            <button type="submit" data-testid="task-complete">Завершить</button>
                          </form>
                        """
                    rows.append(
                        f"""
                        <div class="task" data-testid="task-row" data-task-id="{html.escape(task.id)}">
                          <span data-testid="task-title">{html.escape(task.title)}</span>
                          <span data-testid="task-status">[{html.escape(task.status)}]</span>
                          {complete_form}
                          <form class="inline" method="post" action="/tasks/delete">
                            <input type="hidden" name="task_id" value="{html.escape(task.id)}"/>
                            <button type="submit" data-testid="task-delete">Удалить</button>
                          </form>
                        </div>
                        """
                    )
                empty = (
                    '<p data-testid="tasks-empty">Задач пока нет.</p>'
                    if not rows
                    else ""
                )
                body = f"""
                <p data-testid="user-badge">Вы вошли как <strong>{html.escape(app.sessions[token or ""])}</strong>
                  <a href="/logout" data-testid="logout-link">Выйти</a>
                </p>
                <form method="post" action="/tasks/create" data-testid="create-form">
                  <label>Новая задача<br/>
                    <input name="title" data-testid="task-title-input" placeholder="Название" required/>
                  </label>
                  <button type="submit" data-testid="task-create">Создать</button>
                </form>
                <h2>Список</h2>
                <div data-testid="task-list">{empty}{''.join(rows)}</div>
                """
                self._html(200, _page("Tasks", body, flash=flash))
                return

            if path == "/logout":
                if token and token in app.sessions:
                    del app.sessions[token]
                self._redirect_flash(
                    "/login",
                    "Вы вышли",
                    set_cookie=f"{SESSION_COOKIE}=; Path=/; Max-Age=0",
                )
                return

            self._html(404, _page("Not found", "<p class='error'>404</p>"))

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            token = self._session_token()
            form = self._read_form()

            if path == "/login":
                username = form.get("username", "").strip()
                password = form.get("password", "")
                if username == DEMO_USER and password == DEMO_PASSWORD:
                    new_token = secrets.token_hex(16)
                    app.sessions[new_token] = username
                    self._redirect_flash(
                        "/tasks",
                        "Вход выполнен",
                        set_cookie=f"{SESSION_COOKIE}={new_token}; Path=/; HttpOnly",
                    )
                    return
                self._redirect_flash("/login", "Неверный логин или пароль")
                return

            if not app.is_authed(token):
                self._redirect_flash("/login", "Требуется вход")
                return

            if path == "/tasks/create":
                title = form.get("title", "").strip()
                if not title:
                    self._redirect_flash("/tasks", "Укажите название")
                    return
                task = app.service.add_task(title, source="web", tags=["web"])
                self._redirect_flash("/tasks", f"Создано: {task.id}")
                return

            if path == "/tasks/complete":
                task_id = form.get("task_id", "").strip()
                try:
                    done = app.service.complete_task(task_id)
                except RuntimeError:
                    self._redirect_flash("/tasks", "Задача не найдена")
                    return
                self._redirect_flash("/tasks", f"Завершено: {done.id}")
                return

            if path == "/tasks/delete":
                task_id = form.get("task_id", "").strip()
                try:
                    removed = app.service.delete_task(task_id)
                except RuntimeError:
                    self._redirect_flash("/tasks", "Задача не найдена")
                    return
                self._redirect_flash("/tasks", f"Удалено: {removed.id}")
                return

            self._html(404, _page("Not found", "<p class='error'>404</p>"))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 38 demo web UI for task tracker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parent / "data" / "web_tasks.json"),
    )
    args = parser.parse_args()
    db_path = Path(args.db)
    app = TrackerWebApp(db_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving on http://{args.host}:{args.port}/login (demo/demo123)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
